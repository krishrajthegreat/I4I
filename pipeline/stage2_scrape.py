"""
pipeline/stage2_scrape.py
==========================
Stage 2 -- DuckDuckGo image scraper with query expansion.

For each of the 22 classes, expands the class name across 5 variation-axis
query templates (PRD Section 3.1), scrapes raw images from DuckDuckGo, and
downloads them into dataset/train/<class>/.

Usage
-----
    python -m pipeline.stage2_scrape --classes lathe            # single class
    python -m pipeline.stage2_scrape --classes lathe table_saw  # multiple classes
    python -m pipeline.stage2_scrape --classes lathe --resume   # skip if already at cap

Rate-limit handling
-------------------
DuckDuckGo's DDGS.images() raises duckduckgo_search.exceptions.RatelimitException
when the backend throttles the client.  This script retries with exponential
back-off (base 10 s, max DDGS_MAX_RETRIES attempts per query) before giving up
on that query and moving on.

Output
------
- Images -> dataset/train/<class>/img_<hash>.jpg
- Log    -> logs/stage2_scrape_log.json  (per-class, per-template counts)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

# Add repo root to sys.path so we can import pipeline.config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import (
    ALL_CLASS_NAMES,
    CLASSES,
    DATASET_TRAIN_DIR,
    DDGS_MAX_RETRIES,
    DDGS_SLEEP_BETWEEN_QUERIES,
    DOWNLOAD_TIMEOUT,
    LOGS_DIR,
    MIN_RESOLUTION,
    QUERY_TEMPLATES,
    SCRAPE_CAP_PER_CLASS,
)

# ---------------------------------------------------------------------------
# DuckDuckGo import  (graceful fallback between ddgs and duckduckgo_search)
# ---------------------------------------------------------------------------
try:
    from ddgs import DDGS
    from ddgs.exceptions import RatelimitException, DDGSException as DuckDuckGoSearchException
except ImportError:
    try:
        from duckduckgo_search import DDGS
        from duckduckgo_search.exceptions import RatelimitException, DuckDuckGoSearchException
    except ImportError:
        print("[ERROR] Neither `ddgs` nor `duckduckgo-search` is installed.")
        print("        Run: pip install ddgs")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXCLUDE_KEYWORDS = {
    "cctv", "surveillance", "network router", "wifi router", "router switch",
    "vector", "stock illustration", "diagram", "schematic", "clipart",
    "logo", "icon", "drawing", "sketch", "blueprint", "rendering", "3d model"
}


def _is_relevant_result(result: dict) -> bool:
    """Filter out non-machine / off-target results based on title and URL."""
    title = (result.get("title") or "").lower()
    image_url = (result.get("image") or "").lower()
    
    if any(kw in title for kw in EXCLUDE_KEYWORDS):
        return False
    if any(kw in image_url for kw in EXCLUDE_KEYWORDS):
        return False
    return True


def _url_to_filename(url: str, class_name: str, idx: int) -> str:
    """Derive a stable, filesystem-safe filename from a URL."""
    digest = hashlib.md5(url.encode()).hexdigest()[:10]
    return f"img_{class_name}_{idx:05d}_{digest}.jpg"


def _existing_count(class_dir: Path) -> int:
    """Count already-downloaded images for a class."""
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(1 for p in class_dir.iterdir() if p.suffix.lower() in exts) if class_dir.exists() else 0


def _download_image(url: str, dest: Path, timeout: int = DOWNLOAD_TIMEOUT) -> bool:
    """
    Download a single image URL to dest.  Returns True on success.
    Enforces minimum image dimensions (MIN_RESOLUTION).
    Converts any format to JPEG on save.
    """
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        if resp.status_code != 200:
            return False
        data = resp.content
        # Validate it's actually an image
        img = Image.open(io.BytesIO(data))
        img.verify()                     # raises on corrupt
        # Re-open after verify (verify closes the file)
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        min_dim = max(MIN_RESOLUTION, 200)
        if w < min_dim or h < min_dim:
            return False
        img = img.convert("RGB")
        img.save(dest, "JPEG", quality=90)
        return True
    except Exception:
        return False


def _search_with_retry(
    query: str,
    max_results: int,
    retries: int = DDGS_MAX_RETRIES,
) -> list[dict]:
    """
    Run DDGS.images() with exponential back-off on rate-limit errors.
    Returns a list of result dicts, or [] on persistent failure.
    """
    delay = 10  # seconds; doubles on each retry
    for attempt in range(1, retries + 1):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=max_results))
            return results
        except RatelimitException:
            if attempt == retries:
                print(f"      [RATE-LIMIT] giving up on query after {retries} retries: {query!r}")
                return []
            wait = delay * (2 ** (attempt - 1))
            print(f"      [RATE-LIMIT] attempt {attempt}/{retries}, waiting {wait}s ...")
            time.sleep(wait)
        except Exception as exc:
            if attempt == retries:
                print(f"      [ERROR] giving up on query after {retries} retries: {query!r} ({exc})")
                return []
            wait = delay * (2 ** (attempt - 1))
            print(f"      [ERROR] attempt {attempt}/{retries} ({exc}), waiting {wait}s ...")
            time.sleep(wait)
    return []


# ---------------------------------------------------------------------------
# Per-class scraping logic
# ---------------------------------------------------------------------------

def scrape_class(
    cls: dict,
    resume: bool = False,
    cap: int = SCRAPE_CAP_PER_CLASS,
) -> dict:
    """
    Scrape images for one class across all 5 query templates.

    Returns a summary dict with per-template counts and total.
    """
    name = cls["name"]
    display = cls["display"]
    class_dir = DATASET_TRAIN_DIR / name
    class_dir.mkdir(parents=True, exist_ok=True)

    already = _existing_count(class_dir)
    if resume and already >= cap:
        print(f"  [{name}] already at cap ({already}), skipping.")
        return {
            "class": name,
            "skipped_resume": True,
            "existing": already,
            "templates": {},
            "total_downloaded": 0,
        }

    remaining_cap = cap - already
    # Distribute cap evenly across templates; remainder goes to first template
    per_template_base = remaining_cap // len(QUERY_TEMPLATES)
    per_template_extra = remaining_cap % len(QUERY_TEMPLATES)

    template_counts: dict[str, dict] = {}
    total_downloaded = 0
    global_idx = already  # continue numbering from where we left off

    # We ask DuckDuckGo for slightly more than per-template quota because
    # some URLs will fail to download.  Ask for 1.5x and discard excess.
    fetch_multiplier = 1.5

    print(f"\n  [{name}] target={SCRAPE_CAP_PER_CLASS} | existing={already} | need={remaining_cap}")

    for t_idx, template in enumerate(QUERY_TEMPLATES):
        quota = per_template_base + (1 if t_idx < per_template_extra else 0)
        if total_downloaded >= remaining_cap or quota == 0:
            template_counts[template] = {"fetched": 0, "downloaded": 0, "quota": quota}
            continue

        query = template.format(machine=display)
        fetch_count = min(max(int(quota * 3), 60), 100)  # DDGS cap awareness

        print(f"    template {t_idx + 1}/{len(QUERY_TEMPLATES)}: {query!r} | target={remaining_cap}, fetching up to {fetch_count}")

        results = _search_with_retry(query, max_results=fetch_count)
        time.sleep(DDGS_SLEEP_BETWEEN_QUERIES)

        downloaded_this_template = 0

        for result in results:
            if total_downloaded >= remaining_cap:
                break
            if not _is_relevant_result(result):
                continue

            # Try primary image URL, fall back to thumbnail
            for url_key in ("image", "thumbnail"):
                url = result.get(url_key, "")
                if not url:
                    continue

                dest_filename = _url_to_filename(url, name, global_idx)
                dest_path = class_dir / dest_filename

                if dest_path.exists():
                    continue  # already downloaded (e.g. from a previous partial run)

                success = _download_image(url, dest_path)
                if success:
                    global_idx += 1
                    downloaded_this_template += 1
                    total_downloaded += 1
                    break  # move to next result

        template_counts[template] = {
            "query": query,
            "fetched_from_ddgs": len(results),
            "downloaded": downloaded_this_template,
            "quota": quota,
        }
        print(f"      -> fetched {len(results)}, downloaded {downloaded_this_template}/{quota}")

    final_count = _existing_count(class_dir)
    return {
        "class": name,
        "skipped_resume": False,
        "existing_before": already,
        "total_downloaded": total_downloaded,
        "final_count": final_count,
        "templates": template_counts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: DuckDuckGo image scraper with 5-template query expansion."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        metavar="CLASS",
        help=(
            "One or more class slugs to scrape (required). "
            "Example: --classes lathe   or   --classes lathe table_saw. "
            f"Valid options: {ALL_CLASS_NAMES}"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip any class that already has >= cap images.",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=SCRAPE_CAP_PER_CLASS,
        help=f"Maximum total raw images per class (default: {SCRAPE_CAP_PER_CLASS}).",
    )
    args = parser.parse_args()

    # Validate provided class slugs
    invalid = [c for c in args.classes if c not in ALL_CLASS_NAMES]
    if invalid:
        print(f"[ERROR] Unknown class slug(s): {invalid}")
        print(f"        Valid slugs: {ALL_CLASS_NAMES}")
        sys.exit(1)
    target_classes = [c for c in CLASSES if c["name"] in args.classes]

    print("=" * 65)
    print("STAGE 2 -- DuckDuckGo Image Scraper")
    print(f"Classes    : {len(target_classes)}")
    print(f"Cap/class  : {args.cap}")
    print(f"Templates  : {len(QUERY_TEMPLATES)}")
    print(f"Resume mode: {args.resume}")
    print("=" * 65)

    results = []
    for cls in tqdm(target_classes, desc="Classes", unit="class", position=0, leave=True):
        result = scrape_class(cls, resume=args.resume, cap=args.cap)
        results.append(result)

    # Save log
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "stage2_scrape_log.json"
    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cap_per_class": args.cap,
        "results": results,
    }
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    # Print summary table
    print("\n" + "=" * 65)
    print("SCRAPE SUMMARY")
    print("=" * 65)
    print(f"  {'Class':<25} {'Before':>7} {'Downloaded':>11} {'Final':>7}")
    print("-" * 65)
    total_imgs = 0
    for r in results:
        before   = r.get("existing_before", r.get("existing", 0))
        new_dl   = r.get("total_downloaded", 0)
        final    = r.get("final_count", before)
        skipped  = "  (skipped)" if r.get("skipped_resume") else ""
        print(f"  {r['class']:<25} {before:>7} {new_dl:>11} {final:>7}{skipped}")
        total_imgs += final
    print("-" * 65)
    print(f"  {'TOTAL':<25} {'':>7} {'':>11} {total_imgs:>7}")
    print("=" * 65)
    print(f"\nLog saved -> {log_path}")

    # Flag classes below 50 raw images (likely narrow search terms per PRD)
    low_classes = [r for r in results if r.get("final_count", 0) < 50 and not r.get("skipped_resume")]
    if low_classes:
        print("\nWARNING: The following classes returned fewer than 50 raw images.")
        print("         These are expected narrow search terms (PRD Section 3.4).")
        print("         Consider alternate queries or a lower accepted floor:")
        for r in low_classes:
            print(f"  * {r['class']} -> {r.get('final_count', 0)} images")


if __name__ == "__main__":
    main()
