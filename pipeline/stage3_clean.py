"""
pipeline/stage3_clean.py
========================
Stage 3 -- Image Cleaning & Deduplication.

Performs three cleaning steps on raw scraped images in dataset/train/<class>/:
1. Format & Corrupt File Check: verifies PIL can open and read image headers/data.
2. Minimum Resolution Filter: discards any image with width < 150px or height < 150px.
3. Perceptual Hashing Deduplication: computes pHash (perceptual hash) for each image
   and removes duplicates (Hamming distance <= PHASH_DUPLICATE_THRESHOLD, default=8).

Usage
-----
    python -m pipeline.stage3_clean                    # clean all classes
    python -m pipeline.stage3_clean --classes lathe     # clean specific class

Output
------
- Removes corrupt/small/duplicate files in place from dataset/train/<class>/
- Log -> logs/stage3_clean_log.json  (before vs. after counts per class)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import (
    ALL_CLASS_NAMES,
    CLASSES,
    DATASET_TRAIN_DIR,
    LOGS_DIR,
    MIN_RESOLUTION,
    PHASH_DUPLICATE_THRESHOLD,
)


def clean_class(cls_name: str) -> dict:
    class_dir = DATASET_TRAIN_DIR / cls_name
    if not class_dir.exists():
        return {
            "class": cls_name,
            "raw_count": 0,
            "corrupt_removed": 0,
            "low_res_removed": 0,
            "duplicates_removed": 0,
            "final_count": 0,
        }

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])
    raw_count = len(files)

    corrupt_count = 0
    low_res_count = 0
    duplicate_count = 0

    valid_images: list[tuple[Path, Image.Image, imagehash.ImageHash]] = []
    seen_hashes: list[imagehash.ImageHash] = []

    for path in files:
        # Step 1: Corrupt file check
        try:
            with Image.open(path) as img:
                img.verify()
            # Re-open after verify
            img = Image.open(path).convert("RGB")
        except Exception:
            corrupt_count += 1
            path.unlink(missing_ok=True)
            continue

        # Step 2: Resolution check
        width, height = img.size
        if width < MIN_RESOLUTION or height < MIN_RESOLUTION:
            low_res_count += 1
            path.unlink(missing_ok=True)
            continue

        # Step 3: pHash near-duplicate check
        try:
            h = imagehash.phash(img)
        except Exception:
            corrupt_count += 1
            path.unlink(missing_ok=True)
            continue

        is_duplicate = False
        for existing_h in seen_hashes:
            if (h - existing_h) <= PHASH_DUPLICATE_THRESHOLD:
                is_duplicate = True
                break

        if is_duplicate:
            duplicate_count += 1
            path.unlink(missing_ok=True)
        else:
            seen_hashes.append(h)
            valid_images.append((path, img, h))

    final_count = len(valid_images)
    return {
        "class": cls_name,
        "raw_count": raw_count,
        "corrupt_removed": corrupt_count,
        "low_res_removed": low_res_count,
        "duplicates_removed": duplicate_count,
        "final_count": final_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: Image cleaning & pHash deduplication."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Clean only these class slugs.",
    )
    args = parser.parse_args()

    if args.classes:
        target_names = [c for c in args.classes if c in ALL_CLASS_NAMES]
    else:
        target_names = ALL_CLASS_NAMES

    print("=" * 65)
    print("STAGE 3 -- Image Cleaning & Deduplication")
    print(f"Classes        : {len(target_names)}")
    print(f"Min Resolution : {MIN_RESOLUTION}x{MIN_RESOLUTION} px")
    print(f"pHash Threshold: Hamming distance <= {PHASH_DUPLICATE_THRESHOLD}")
    print("=" * 65)

    results = []
    for cls_name in tqdm(target_names, desc="Cleaning", unit="class"):
        res = clean_class(cls_name)
        results.append(res)

    # Save log
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "stage3_clean_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "min_resolution": MIN_RESOLUTION,
                "phash_threshold": PHASH_DUPLICATE_THRESHOLD,
                "results": results,
            },
            f,
            indent=2,
        )

    # Print summary table
    print("\n" + "=" * 65)
    print("CLEANING SUMMARY")
    print("=" * 65)
    print(f"  {'Class':<22} {'Raw':>6} {'Corrupt':>8} {'LowRes':>7} {'Dupes':>6} {'Clean':>6}")
    print("-" * 65)
    total_raw = total_corrupt = total_lowres = total_dupes = total_clean = 0
    for r in results:
        print(
            f"  {r['class']:<22} {r['raw_count']:>6} {r['corrupt_removed']:>8} "
            f"{r['low_res_removed']:>7} {r['duplicates_removed']:>6} {r['final_count']:>6}"
        )
        total_raw += r["raw_count"]
        total_corrupt += r["corrupt_removed"]
        total_lowres += r["low_res_removed"]
        total_dupes += r["duplicates_removed"]
        total_clean += r["final_count"]
    print("-" * 65)
    print(
        f"  {'TOTAL':<22} {total_raw:>6} {total_corrupt:>8} "
        f"{total_lowres:>7} {total_dupes:>6} {total_clean:>6}"
    )
    print("=" * 65)
    print(f"\nLog saved -> {log_path}")


if __name__ == "__main__":
    main()
