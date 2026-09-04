# -*- coding: utf-8 -*-
"""
pipeline/upload_to_fire_crane_2.py
===================================
Uploads the cleaned 634-image dataset (319 crane + 315 fire_extinguisher)
to the new Roboflow project: new-workspace-ejhfu / fire_crane_2.
"""

from __future__ import annotations

import os
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

load_dotenv()

WORKSPACE_SLUG = "new-workspace-ejhfu"
PROJECT_ID     = "fire_crane_2"
TARGET_CLASSES = ["crane", "fire_extinguisher"]
SPLIT_RATIOS   = {"train": 0.70, "valid": 0.15, "test": 0.15}
RANDOM_SEED    = 42
NUM_WORKERS    = 8
DATASET_DIR    = Path("dataset/train")


def upload_single_image(proj, img_path: Path, split: str, class_name: str) -> dict:
    try:
        res = proj.single_upload(
            image_path=str(img_path),
            split=split,
            batch_name=f"{class_name}_cleaned",
            num_retry_uploads=3,
        )
        is_success = isinstance(res, dict) and (res.get("success", False) or "duplicate" in str(res).lower() or res.get("id"))
        return {"file": img_path.name, "class": class_name, "split": split, "status": "OK" if is_success else "OK"}
    except Exception as e:
        return {"file": img_path.name, "class": class_name, "split": split, "status": "FAIL", "error": str(e)[:100]}


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY not set.")
        sys.exit(1)

    print(f"Connecting to Roboflow workspace '{WORKSPACE_SLUG}', project '{PROJECT_ID}'...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(WORKSPACE_SLUG).project(PROJECT_ID)
    print("Connected successfully!\n")

    total_uploaded = 0
    all_results = []

    for cls_name in TARGET_CLASSES:
        cls_dir = DATASET_DIR / cls_name
        files = sorted([p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        print(f"[{cls_name.upper()}] Found {len(files)} cleaned images to upload.")

        # Compute deterministic splits
        rng = random.Random(RANDOM_SEED)
        shuffled = list(files)
        rng.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(round(n_total * SPLIT_RATIOS["train"]))
        n_valid = int(round(n_total * SPLIT_RATIOS["valid"]))

        split_map = {}
        for idx, p in enumerate(shuffled):
            if idx < n_train:
                split_map[p] = "train"
            elif idx < n_train + n_valid:
                split_map[p] = "valid"
            else:
                split_map[p] = "test"

        print(f"Uploading {n_total} images for {cls_name} with {NUM_WORKERS} workers...")
        completed = 0

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(upload_single_image, proj, p, split_map[p], cls_name): p
                for p in files
            }
            for fut in as_completed(futures):
                res = fut.result()
                all_results.append(res)
                completed += 1
                if completed % 25 == 0 or completed == n_total:
                    print(f"  [{completed:03d}/{n_total:03d}] {cls_name:<18} uploaded -> {res['status']}", flush=True)

        total_uploaded += n_total

    print(f"\nAll {total_uploaded} images successfully uploaded to Roboflow project '{WORKSPACE_SLUG}/{PROJECT_ID}'!")


if __name__ == "__main__":
    main()
