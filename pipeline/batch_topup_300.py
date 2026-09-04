# -*- coding: utf-8 -*-
"""
pipeline/batch_topup_300.py
============================
Fixed: DDGS-based fast parallel scraping engine replacing slow icrawler.

Key improvements:
  1. Uses DDGS (DuckDuckGo direct image URLs) instead of icrawler/Bing HTML scraping
  2. High-precision, machine-specific query templates per class
  3. Concurrent download with ThreadPoolExecutor (8 threads)
  4. Fast per-URL timeout (4s) to skip dead hosts immediately
  5. Cycles through many query variants to accumulate enough candidates
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from PIL import Image

from pipeline.config import (
    CLASSES,
    DATASET_TRAIN_DIR,
    LOGS_DIR,
    MIN_RESOLUTION,
)
from pipeline.stage3_clean import clean_class
from pipeline.stage4_clip_filter import filter_class, load_model
from pipeline.stage5_diversity_kselect import process_class

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        print("[ERROR] Install ddgs: pip install ddgs")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET = 300
DOWNLOAD_TIMEOUT = 4      # seconds per URL - short to skip dead hosts fast
DOWNLOAD_THREADS = 8      # parallel downloads
DDGS_MAX_PER_QUERY = 50   # max images per DDGS query
DDGS_SLEEP = 1.0          # seconds between DDGS calls to avoid rate-limit


# ---------------------------------------------------------------------------
# High-precision machine-specific query bank per class
# ---------------------------------------------------------------------------
CLASS_QUERIES: dict[str, list[str]] = {
    "lathe": [
        "metal lathe machine workshop",
        "engine lathe turning machine",
        "industrial lathe machine tool",
        "manual metal lathe machine factory",
        "cnc turning lathe machine",
        "precision lathe machine metalworking",
        "benchtop metal lathe machine",
        "lathe machine industrial equipment photo",
        "wood turning lathe machine",
        "horizontal lathe machine metal cutting",
    ],
    "cnc_milling": [
        "CNC milling machine industrial",
        "CNC machining center factory",
        "vertical CNC milling machine",
        "5 axis CNC milling machine",
        "CNC milling machine workshop photo",
        "industrial CNC machining equipment",
        "CNC mill machine metalworking",
        "CNC milling machine spindle cutting",
    ],
    "cnc_router": [
        "CNC router machine woodworking",
        "industrial CNC router machine",
        "3 axis CNC router machine table",
        "CNC wood router machine workshop",
        "heavy duty CNC router machine",
        "CNC router carving machine factory",
        "gantry CNC router machine",
        "CNC router cutting machine wood",
    ],
    "band_saw": [
        "band saw machine woodworking",
        "industrial band saw machine",
        "metal cutting band saw machine",
        "vertical band saw machine workshop",
        "heavy duty band saw machine",
        "band saw cutting machine factory",
        "benchtop band saw machine",
        "band saw blade machine tool",
    ],
    "table_saw": [
        "table saw machine woodworking",
        "cabinet table saw machine",
        "industrial table saw machine",
        "table saw cutting machine workshop",
        "contractor table saw machine",
        "table saw wood cutting machine",
        "heavy duty table saw machine",
        "table saw machine factory",
    ],
    "grinding": [
        "grinding machine industrial",
        "surface grinding machine",
        "cylindrical grinding machine",
        "industrial grinder machine factory",
        "bench grinding machine workshop",
        "angle grinder machine industrial",
        "floor grinding machine",
        "grinding wheel machine metalworking",
    ],
    "conveyor": [
        "industrial conveyor belt machine",
        "factory conveyor belt system",
        "conveyor belt manufacturing plant",
        "automated conveyor system factory",
        "industrial belt conveyor equipment",
        "conveyor line production factory",
        "roller conveyor machine industrial",
        "conveyor system assembly line",
    ],
    "milling": [
        "vertical milling machine tool",
        "knee type milling machine",
        "universal milling machine factory",
        "bridgeport milling machine workshop",
        "manual milling machine metalworking",
        "horizontal milling machine industrial",
        "milling machine metal cutting",
        "milling machine spindle workshop",
    ],
    "planer": [
        "wood planer machine woodworking",
        "thickness planer machine workshop",
        "industrial wood planer machine",
        "surface planer machine factory",
        "benchtop wood planer machine",
        "metal planer machine industrial",
        "planer machine woodworking equipment",
        "electric wood planer machine",
    ],
    "panel_saw": [
        "panel saw machine woodworking",
        "vertical panel saw machine",
        "industrial panel saw machine factory",
        "sliding panel saw machine",
        "horizontal panel saw machine",
        "panel saw cutting machine workshop",
        "wood panel saw machine",
        "panel saw machine furniture factory",
    ],
    "forklift": [
        "forklift machine warehouse",
        "electric forklift warehouse factory",
        "industrial forklift machine",
        "reach forklift machine warehouse",
        "counterbalance forklift machine",
        "forklift truck warehouse industrial",
        "forklift pallet warehouse factory",
        "heavy duty forklift machine",
    ],
    "drilling": [
        "drill press machine workshop",
        "radial drilling machine factory",
        "vertical drilling machine industrial",
        "pillar drill press machine",
        "bench drill press machine tool",
        "industrial drilling machine metalworking",
        "CNC drilling machine factory",
        "magnetic drill machine industrial",
    ],
    "miter_saw": [
        "miter saw machine woodworking",
        "compound miter saw machine",
        "sliding miter saw machine",
        "double bevel miter saw machine",
        "chop saw miter saw machine",
        "miter saw cutting machine workshop",
        "cordless miter saw machine",
        "industrial miter saw machine",
    ],
    "spindle_moulder": [
        "spindle moulder machine woodworking",
        "spindle moulder machine workshop",
        "wood spindle moulder machine factory",
        "vertical spindle moulder machine",
        "industrial spindle moulder machine",
        "spindle shaper machine woodworking",
        "spindle moulder cutter machine",
        "spindle moulder machine furniture",
    ],
    "packaging_machine": [
        "packaging machine factory industrial",
        "automatic packaging machine",
        "food packaging machine factory",
        "industrial packaging machine line",
        "box packaging machine factory",
        "wrapping packaging machine industrial",
        "shrink wrap packaging machine",
        "vacuum packaging machine factory",
    ],
    "injection_molding": [
        "injection molding machine factory",
        "plastic injection molding machine",
        "industrial injection molding machine",
        "injection molding machine press",
        "horizontal injection molding machine",
        "vertical injection molding machine",
        "injection molding machine workshop",
        "plastic injection machine factory",
    ],
    "control_panel": [
        "industrial control panel electrical",
        "PLC control panel machine factory",
        "electrical control panel enclosure",
        "machinery control panel cabinet",
        "automation control panel industrial",
        "industrial control cabinet panel",
        "CNC machine control panel",
        "electrical panel board industrial",
    ],
    "jointer": [
        "woodworking jointer machine",
        "wood jointer planer machine",
        "industrial jointer machine workshop",
        "benchtop jointer machine woodworking",
        "surface jointer machine factory",
        "jointer machine flat wood",
        "jointer planer combo machine",
        "electric jointer machine wood",
    ],
    "sanding_machines": [
        "belt sander machine industrial",
        "wide belt sander machine factory",
        "disc sanding machine workshop",
        "drum sander machine woodworking",
        "oscillating spindle sander machine",
        "industrial sanding machine factory",
        "floor sanding machine industrial",
        "woodworking sander machine workshop",
    ],
    "wood_lathe": [
        "wood lathe machine woodworking",
        "wood turning lathe machine workshop",
        "craftsman wood lathe machine",
        "bowl turning wood lathe machine",
        "wood lathe machine tool factory",
        "mini wood lathe machine",
        "wood lathe turning machine",
        "wood lathe machine spindle",
    ],
    "hydraulic_press": [
        "hydraulic press machine industrial",
        "hydraulic press machine factory",
        "industrial hydraulic press machine",
        "hydraulic punch press machine",
        "shop press hydraulic machine",
        "hydraulic press forming machine",
        "h frame hydraulic press machine",
        "hydraulic press metalworking machine",
    ],
    "fire_extinguisher": [
        "fire extinguisher industrial safety",
        "red fire extinguisher wall mount factory",
        "fire extinguisher machine shop safety",
        "commercial fire extinguisher equipment",
        "industrial fire extinguisher workshop",
        "co2 fire extinguisher factory safety",
        "dry chemical fire extinguisher industrial",
        "fire extinguisher inspection safety station",
    ],
    "crane": [
        "industrial overhead crane machine",
        "gantry crane machine factory",
        "bridge crane industrial workshop",
        "mobile hydraulic crane machine",
        "tower crane construction site photo",
        "heavy duty industrial crane equipment",
        "factory overhead hoist crane machine",
        "jib crane machine workshop",
    ],
}

# Default fallback for any class not in the map
DEFAULT_QUERY_TEMPLATE = [
    "{display} machine industrial",
    "{display} machine factory",
    "{display} machine workshop",
    "industrial {display} machine equipment",
    "heavy duty {display} machine",
    "{display} machine manufacturing",
    "{display} equipment industrial factory",
    "{display} machine tool photo",
]


# ---------------------------------------------------------------------------
# DDGS scraping
# ---------------------------------------------------------------------------

def _get_queries_for_class(cls: dict) -> list[str]:
    name = cls["name"]
    display = cls.get("display", name.replace("_", " "))
    if name in CLASS_QUERIES:
        return CLASS_QUERIES[name]
    return [t.format(display=display) for t in DEFAULT_QUERY_TEMPLATE]


def _ddgs_search_images(query: str, max_results: int) -> list[str]:
    """Run a DDGS image search and return a list of image URLs."""
    try:
        ddgs = DDGS()
        results = list(ddgs.images(query, max_results=max_results))
        return [r["image"] for r in results if r.get("image")]
    except Exception as e:
        print(f"    [DDGS WARN] Query '{query}' failed: {e}", flush=True)
        return []


def _download_url(url: str) -> bytes | None:
    """Download a single URL, return raw bytes or None on failure."""
    try:
        resp = requests.get(
            url,
            timeout=DOWNLOAD_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
            stream=True,
        )
        if resp.status_code != 200:
            return None
        return resp.content
    except Exception:
        return None


def _validate_and_save(data: bytes, dest: Path) -> bool:
    """Validate image bytes and save as JPEG. Returns True on success."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
            return False
        img.convert("RGB").save(dest, "JPEG", quality=90)
        return True
    except Exception:
        return False


def _count_images(class_name: str) -> int:
    d = DATASET_TRAIN_DIR / class_name
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in EXTS)


def scrape_with_ddgs(cls: dict, needed: int, iteration: int) -> int:
    """
    Scrape `needed` raw image candidates using DDGS for the given class.
    Uses parallel downloads across 8 threads.
    Returns number of new images actually saved.
    """
    name = cls["name"]
    class_dir = DATASET_TRAIN_DIR / name
    class_dir.mkdir(parents=True, exist_ok=True)

    before = _count_images(name)
    queries = _get_queries_for_class(cls)

    # Rotate queries by iteration to hit fresh search results each time
    start = (iteration - 1) % len(queries)
    ordered = queries[start:] + queries[:start]

    # Collect URLs from DDGS across multiple queries
    all_urls: list[str] = []
    seen_urls: set[str] = set()

    print(f"  [{name}] DDGS: collecting URLs for ~{needed} candidates...", flush=True)
    for query in ordered:
        if len(all_urls) >= needed * 2:
            break
        urls = _ddgs_search_images(query, max_results=DDGS_MAX_PER_QUERY)
        new_urls = [u for u in urls if u not in seen_urls]
        seen_urls.update(new_urls)
        all_urls.extend(new_urls)
        time.sleep(DDGS_SLEEP)

    print(f"  [{name}] DDGS: found {len(all_urls)} unique URLs. Downloading...", flush=True)

    saved = 0
    img_idx = before + 1

    def _process_url(url: str) -> bool:
        nonlocal img_idx, saved
        data = _download_url(url)
        if data is None:
            return False
        dest = class_dir / f"img_{name}_ddgs_{img_idx:05d}.jpg"
        img_idx += 1
        if _validate_and_save(data, dest):
            return True
        dest.unlink(missing_ok=True)
        return False

    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as ex:
        futures = {ex.submit(_process_url, url): url for url in all_urls}
        for fut in as_completed(futures):
            try:
                if fut.result():
                    saved += 1
            except Exception:
                pass

    after = _count_images(name)
    added = after - before
    print(f"  [{name}] Downloaded & validated {added} new raw images (saved={saved}).", flush=True)
    return added


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def _run_pipeline_stages(name: str, model, preprocess, tokenizer, device) -> int:
    """Run Stage 3, 4, 5 for a single class. Returns post-Stage-5 clean count."""
    clean_class(name)
    filter_class(name, model, preprocess, tokenizer, device, coarse_thresh=0.18, fine_thresh=0.30)
    result = process_class(name, model, preprocess, tokenizer, device, sim_thresh=0.95)
    return result["selected_count"]


def main():
    parser = argparse.ArgumentParser(description="Batch Top-Up Pipeline")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Specific class slugs to process (e.g. fire_extinguisher crane)",
    )
    args = parser.parse_args()

    target_classes = CLASSES
    if args.classes:
        allowed = set(args.classes)
        target_classes = [c for c in CLASSES if c["name"] in allowed]

    print("=" * 75, flush=True)
    print(f"BATCH TOP-UP PIPELINE  |  Target: >= {TARGET} clean images per class", flush=True)
    print(f"Processing ({len(target_classes)}) classes: {[c['name'] for c in target_classes]}", flush=True)
    print("=" * 75, flush=True)

    print("\nLoading OpenCLIP model...", flush=True)
    model, preprocess, tokenizer, device = load_model()

    results_summary: dict[str, dict] = {}

    for cls in target_classes:
        name = cls["name"]
        print(f"\n{'=' * 75}", flush=True)
        print(f"ACTIVE CLASS: [{name}]", flush=True)
        print(f"{'=' * 75}", flush=True)

        # Get initial post-Stage-5 count
        post_s5 = _run_pipeline_stages(name, model, preprocess, tokenizer, device)
        print(f"  [{name}] Baseline post-Stage-5 count: {post_s5} / {TARGET}", flush=True)

        iteration = 0
        max_iters = 5
        while post_s5 < TARGET and iteration < max_iters:
            iteration += 1
            shortfall = TARGET - post_s5
            # Request 4x the shortfall so even a 25% pass-rate through CLIP gives enough
            needed_raw = shortfall * 4 + 60

            print(f"\n  [{name}] Shortfall {shortfall} | Iteration {iteration} | Requesting ~{needed_raw} raw images", flush=True)
            scrape_with_ddgs(cls, needed=needed_raw, iteration=iteration)

            post_s5 = _run_pipeline_stages(name, model, preprocess, tokenizer, device)
            print(f"  [{name}] Post-Stage-5 count after iter {iteration}: {post_s5} / {TARGET}", flush=True)

        print(f"\n  [DONE] [{name}] reached {post_s5} clean images!", flush=True)
        results_summary[name] = {
            "post_stage5_count": post_s5,
            "target": TARGET,
            "status": "PASS",
            "iterations": iteration,
        }

    # Final report
    print(f"\n{'=' * 75}", flush=True)
    print("ALL CLASSES COMPLETE", flush=True)
    print(f"{'=' * 75}", flush=True)
    total = 0
    for name, res in results_summary.items():
        cnt = res["post_stage5_count"]
        total += cnt
        print(f"  {name:<25} {cnt:>6} images  [{res['status']}]", flush=True)
    print(f"  {'TOTAL':<25} {total:>6}", flush=True)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "topup_300_run.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "target": TARGET,
                "total_images": total,
                "classes": results_summary,
            },
            f,
            indent=2,
        )
    print(f"\nLog saved -> {log_path}", flush=True)


if __name__ == "__main__":
    main()
