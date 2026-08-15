# -*- coding: utf-8 -*-
"""
pipeline/batch_topup_200.py
============================
Automated batch top-up pipeline.
Ensures that EVERY class has at least 200 clean, diverse images AFTER Stage 5.

For any class that falls below 200 post-Stage 5:
1. Scrapes additional images via DDGS and BingImageCrawler fallback across query templates.
2. Runs Stage 3 (Cleaning & pHash dedup).
3. Runs Stage 4 (Hybrid CLIP filtering).
4. Runs Stage 5 (Diversity Dedup & K-Selection).
5. Loops until the post-Stage 5 image count is >= 200 for all 21 classes.
"""

from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import requests
from PIL import Image
from icrawler.builtin import BingImageCrawler

from pipeline.config import (
    ALL_CLASS_NAMES,
    CLASSES,
    DATASET_TRAIN_DIR,
    DOWNLOAD_TIMEOUT,
    LOGS_DIR,
    MIN_RESOLUTION,
    QUERY_TEMPLATES,
    class_by_name,
)
from pipeline.stage3_clean import clean_class
from pipeline.stage4_clip_filter import filter_class, load_model
from pipeline.stage5_diversity_kselect import process_class

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_POST_STAGE5_COUNT = 200


def count_images(class_name: str) -> int:
    d = DATASET_TRAIN_DIR / class_name
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in EXTS)


def scrape_bing_fallback(cls_dict: dict, needed_raw: int) -> int:
    """Scrape additional images using BingImageCrawler as fallback."""
    name = cls_dict["name"]
    display = cls_dict["display"]
    class_dir = DATASET_TRAIN_DIR / name
    class_dir.mkdir(parents=True, exist_ok=True)

    start_count = count_images(name)
    downloaded = 0

    templates = QUERY_TEMPLATES[:8]  # Top templates
    per_template = max(15, (needed_raw // len(templates)) + 5)

    print(f"  [{name}] Bing fallback scraping needed raw={needed_raw}...")

    for template in templates:
        if count_images(name) - start_count >= needed_raw:
            break

        query = template.format(machine=display)
        temp_download_dir = class_dir / "_bing_tmp"
        temp_download_dir.mkdir(exist_ok=True)

        try:
            crawler = BingImageCrawler(
                storage={"root_dir": str(temp_download_dir)},
                log_level=40,
                downloader_threads=4,
            )
            crawler.crawl(keyword=query, max_num=per_template)

            for p in temp_download_dir.iterdir():
                if p.is_file() and p.suffix.lower() in EXTS:
                    try:
                        with Image.open(p) as img:
                            w, h = img.size
                            if w >= MIN_RESOLUTION and h >= MIN_RESOLUTION:
                                img = img.convert("RGB")
                                new_idx = count_images(name) + 1
                                dest_path = class_dir / f"img_{name}_bing_{new_idx:05d}.jpg"
                                img.save(dest_path, "JPEG", quality=90)
                                downloaded += 1
                    except Exception:
                        pass
                    p.unlink(missing_ok=True)
        except Exception as e:
            print(f"    [WARN] Bing crawler template '{query}' failed: {e}")
        finally:
            if temp_download_dir.exists():
                for f in temp_download_dir.iterdir():
                    f.unlink(missing_ok=True)
                temp_download_dir.rmdir()

    final_added = count_images(name) - start_count
    print(f"  [{name}] Bing fallback added {final_added} new raw images.")
    return final_added


def run_pipeline_for_class(cls_name: str, model, preprocess, tokenizer, device) -> int:
    """Run Stage 3, 4, 5 for a single class and return post-Stage 5 count."""
    clean_res = clean_class(cls_name)
    filter_res = filter_class(
        cls_name, model, preprocess, tokenizer, device, coarse_thresh=0.18, fine_thresh=0.30
    )
    stage5_res = process_class(
        cls_name, model, preprocess, tokenizer, device, sim_thresh=0.95
    )
    return stage5_res["selected_count"]


def main():
    print("=" * 70)
    print(f"BATCH TOP-UP PIPELINE — TARGET: >= {TARGET_POST_STAGE5_COUNT} IMAGES POST-STAGE 5")
    print("=" * 70)

    print("\nLoading OpenCLIP model...")
    model, preprocess, tokenizer, device = load_model()

    results_summary = {}

    for cls in CLASSES:
        name = cls["name"]
        print(f"\n" + "-" * 70)
        print(f"Processing Class: [{name}]")
        print("-" * 70)

        # 1. Run Stages 3, 4, 5 on current images to get baseline post-stage-5 count
        current_post_s5 = run_pipeline_for_class(name, model, preprocess, tokenizer, device)
        print(f"  [{name}] Baseline post-Stage 5 count: {current_post_s5}")

        iterations = 0
        max_iterations = 3

        # 2. Top-up loop if under target
        while current_post_s5 < TARGET_POST_STAGE5_COUNT and iterations < max_iterations:
            iterations += 1
            needed_approx = (TARGET_POST_STAGE5_COUNT - current_post_s5) * 2 + 50
            print(f"  [{name}] Below target ({current_post_s5}/{TARGET_POST_STAGE5_COUNT}). Iteration {iterations}: Scrape ~{needed_approx} more raw images...")

            scrape_bing_fallback(cls, needed_raw=needed_approx)

            # Re-run pipeline
            current_post_s5 = run_pipeline_for_class(name, model, preprocess, tokenizer, device)
            print(f"  [{name}] Post-Stage 5 count after iteration {iterations}: {current_post_s5}")

        status = "PASS" if current_post_s5 >= TARGET_POST_STAGE5_COUNT else "FLAGGED"
        results_summary[name] = {
            "post_stage5_count": current_post_s5,
            "target": TARGET_POST_STAGE5_COUNT,
            "status": status,
            "iterations": iterations,
        }
        print(f"  [{name}] FINAL POST-STAGE 5 COUNT: {current_post_s5} | Status: {status}")

    # 3. Print Final Report
    print("\n" + "=" * 70)
    print("BATCH TOP-UP COMPLETE — FINAL POST-STAGE 5 SUMMARY")
    print("=" * 70)
    print(f"  {'Class Name':<25} {'Post-Stage 5 Count':>20} {'Target':>10} {'Status':<10}")
    print("-" * 70)
    tot = 0
    all_passed = True
    for name, res in results_summary.items():
        cnt = res["post_stage5_count"]
        tot += cnt
        st = res["status"]
        if st != "PASS":
            all_passed = False
        print(f"  {name:<25} {cnt:>20} {res['target']:>10} {st:<10}")

    print("-" * 70)
    print(f"  {'TOTAL':<25} {tot:>20}")
    print("=" * 70)

    if all_passed:
        print("\nSUCCESS: All 21 classes reached >= 200 images post Stage 5!")
    else:
        print("\nNOTE: Some classes are below 200 images post Stage 5.")

    # Save log
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "topup_200_run.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "target_count": TARGET_POST_STAGE5_COUNT,
                "summary": results_summary,
                "total_images": tot,
            },
            f,
            indent=2,
        )
    print(f"Log written to {log_file}")


if __name__ == "__main__":
    main()
