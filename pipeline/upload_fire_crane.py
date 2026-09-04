# -*- coding: utf-8 -*-
"""
pipeline/upload_fire_crane.py
==============================
Instance-Segmentation Upload Pipeline for 'fire_extinguisher' & 'crane'
Target Project: krish-raj-cgbcn / fire_crane

Features:
- Deterministic 70% Train / 15% Valid / 15% Test split (seed=42).
- Local FastSAM 2D polygon contour mask generation (thread-safe).
- Writes COCO polygon masks and uploads via Roboflow API.
- Multi-threaded processing across 4 worker threads.
- Comprehensive progress tracking and JSON logging.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from roboflow import Roboflow
from ultralytics import FastSAM

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.config import DATASET_TRAIN_DIR, LOGS_DIR

WORKSPACE_SLUG = "krish-raj-cgbcn"
PROJECT_ID     = "fire_crane"
TARGET_CLASSES = ["fire_extinguisher", "crane"]
SPLIT_RATIOS   = {"train": 0.70, "valid": 0.15, "test": 0.15}
RANDOM_SEED    = 42
NUM_WORKERS    = 4
EXTS           = {".jpg", ".jpeg", ".png", ".webp"}

_thread_local = threading.local()


def get_thread_sam_model():
    """Returns a thread-local FastSAM model instance to avoid PyTorch race conditions."""
    if not hasattr(_thread_local, "sam_model"):
        _thread_local.sam_model = FastSAM("FastSAM-s.pt")
    return _thread_local.sam_model


def compute_deterministic_splits(image_paths: list[Path]) -> dict[Path, str]:
    """Assigns deterministic 70/15/15 train/valid/test splits based on seed=42."""
    paths = sorted(image_paths)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(paths)

    total = len(paths)
    n_train = int(round(total * SPLIT_RATIOS["train"]))
    n_valid = int(round(total * SPLIT_RATIOS["valid"]))

    split_map = {}
    for idx, p in enumerate(paths):
        if idx < n_train:
            split_map[p] = "train"
        elif idx < n_train + n_valid:
            split_map[p] = "valid"
        else:
            split_map[p] = "test"
    return split_map


def extract_polygon_mask(img_path: Path, expected_class: str) -> dict | None:
    """Uses FastSAM to extract the dominant machine polygon mask in COCO format."""
    try:
        img = Image.open(img_path)
        w, h = img.size
    except Exception:
        return None

    sam_model = get_thread_sam_model()
    try:
        results = sam_model(str(img_path), device="cpu", retina_masks=True, imgsz=640, conf=0.20, verbose=False)
    except Exception as exc:
        print(f"  [SAM WARN] {img_path.name}: {exc}")
        return None

    best_poly = None
    max_area = 0.0

    for r in results:
        if r.masks is not None:
            for poly in r.masks.xy:
                if len(poly) >= 3:
                    xs = poly[:, 0]
                    ys = poly[:, 1]
                    area = (xs.max() - xs.min()) * (ys.max() - ys.min())
                    # Skip full image background mask (area > 95% of total image)
                    if area > max_area and area < (w * h * 0.95):
                        max_area = area
                        best_poly = poly

    if best_poly is None:
        return None

    flat_points = []
    xs, ys = [], []
    for pt in best_poly:
        x_val = float(pt[0])
        y_val = float(pt[1])
        flat_points.extend([x_val, y_val])
        xs.append(x_val)
        ys.append(y_val)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    bbox_w = max(1.0, xmax - xmin)
    bbox_h = max(1.0, ymax - ymin)

    coco_payload = {
        "images": [{"id": 1, "width": w, "height": h, "file_name": img_path.name}],
        "categories": [{"id": 1, "name": expected_class, "supercategory": "equipment"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [flat_points],
                "area": float(bbox_w * bbox_h),
                "bbox": [xmin, ymin, bbox_w, bbox_h],
                "iscrowd": 0,
            }
        ],
    }

    return {
        "width": w,
        "height": h,
        "points_count": len(best_poly),
        "coco_payload": coco_payload,
    }


def upload_single_image(
    img_path: Path, expected_class: str, target_split: str, proj
) -> dict:
    """Segments image locally and uploads image + polygon mask to Roboflow."""
    filename = img_path.name

    mask_info = extract_polygon_mask(img_path, expected_class)
    if not mask_info:
        return {
            "filename": filename,
            "class": expected_class,
            "split": target_split,
            "status": "FAIL",
            "reason": "FastSAM failed to extract polygon mask",
            "points_count": 0,
        }

    coco_payload = mask_info["coco_payload"]
    points_count = mask_info["points_count"]

    with tempfile.TemporaryDirectory() as tmpdir:
        coco_file = Path(tmpdir) / "_annotations.coco.json"
        coco_file.write_text(json.dumps(coco_payload, indent=2), encoding="utf-8")

        try:
            res = proj.single_upload(
                image_path=str(img_path),
                annotation_path=str(coco_file),
                split=target_split,
                batch_name=expected_class,
                num_retry_uploads=3,
                annotation_overwrite=True,
            )
            anno_res = res.get("annotation", {}) if isinstance(res, dict) else {}
            is_success = anno_res.get("success", False) or (isinstance(res, dict) and "duplicate" in str(res).lower())

            if is_success:
                return {
                    "filename": filename,
                    "class": expected_class,
                    "split": target_split,
                    "status": "OK",
                    "reason": None,
                    "points_count": points_count,
                }
            else:
                return {
                    "filename": filename,
                    "class": expected_class,
                    "split": target_split,
                    "status": "FAIL",
                    "reason": str(res)[:200],
                    "points_count": points_count,
                }
        except Exception as exc:
            return {
                "filename": filename,
                "class": expected_class,
                "split": target_split,
                "status": "FAIL",
                "reason": str(exc)[:200],
                "points_count": points_count,
            }


def process_class_upload(cls_name: str, proj) -> dict:
    """Processes deterministic split, segmentation, and upload for a class."""
    class_dir = DATASET_TRAIN_DIR / cls_name
    if not class_dir.exists():
        print(f"[WARN] Class directory {class_dir} does not exist. Skipping.")
        return {"class": cls_name, "total": 0, "successful": 0, "failed": 0}

    files = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS])
    total = len(files)

    if total == 0:
        print(f"[WARN] No images found in {class_dir}. Skipping.")
        return {"class": cls_name, "total": 0, "successful": 0, "failed": 0}

    split_map = compute_deterministic_splits(files)

    print("\n" + "=" * 75)
    print(f"CLASS: [{cls_name}]  |  Total Images: {total}")
    train_c = sum(1 for s in split_map.values() if s == "train")
    valid_c = sum(1 for s in split_map.values() if s == "valid")
    test_c  = sum(1 for s in split_map.values() if s == "test")
    print(f"  Target Split: Train={train_c} ({train_c/total*100:.1f}%), Valid={valid_c} ({valid_c/total*100:.1f}%), Test={test_c} ({test_c/total*100:.1f}%)")
    print("=" * 75)

    results = []
    completed = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(
                upload_single_image, img_path, cls_name, split_map[img_path], proj
            ): img_path
            for img_path in files
        }

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            completed += 1

            status = res["status"]
            pts = res["points_count"]
            fn = res["filename"]
            sp = res["split"]

            if status == "OK":
                msg = f"pts: {pts:>5} | OK"
            else:
                reason = res.get("reason", "")
                msg = f"pts: {pts:>5} | {status} ({reason})"

            print(f"  [{completed:03d}/{total:03d}] {fn:<40} | {sp:<5} | {msg}", flush=True)

    n_ok   = sum(1 for r in results if r["status"] == "OK")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")

    print(f"\n  [{cls_name} COMPLETE] Successful: {n_ok} | Failed: {n_fail}")

    class_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "class": cls_name,
        "total_images": total,
        "successful": n_ok,
        "failed": n_fail,
        "results": results,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"polygon_upload_{cls_name}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(class_log, f, indent=2)

    return {
        "class": cls_name,
        "total": total,
        "successful": n_ok,
        "failed": n_fail,
    }


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY environment variable not set.")
        sys.exit(1)

    print("=" * 75)
    print("ROBOFLOW POLYGON INSTANCE-SEGMENTATION UPLOAD PIPELINE")
    print(f"  Workspace : {WORKSPACE_SLUG}")
    print(f"  Project   : {PROJECT_ID}")
    print(f"  Split     : 70% Train / 15% Valid / 15% Test (per class)")
    print(f"  Classes   : {TARGET_CLASSES}")
    print("=" * 75)

    print("Loading Roboflow project...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(WORKSPACE_SLUG).project(PROJECT_ID)

    # Warm up FastSAM model on main thread once
    _dummy_model = get_thread_sam_model()

    class_summaries = []
    for cls_name in TARGET_CLASSES:
        summary = process_class_upload(cls_name, proj)
        class_summaries.append(summary)

    print("\n" + "=" * 75)
    print("FINAL UPLOAD SUMMARY FOR FIRE_CRANE")
    print("=" * 75)
    print(f"  {'Class':<25} {'Total':>8} {'Successful':>12} {'Failed':>8}")
    print("  " + "-" * 60)
    for s in class_summaries:
        print(f"  {s['class']:<25} {s['total']:>8} {s['successful']:>12} {s['failed']:>8}")
    print("=" * 75)


if __name__ == "__main__":
    main()
