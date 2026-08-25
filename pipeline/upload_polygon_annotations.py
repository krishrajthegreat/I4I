# -*- coding: utf-8 -*-
"""
pipeline/upload_polygon_annotations.py
=======================================
Full 21-Class SAM Polygon Instance-Segmentation Upload & Gemini Verification.

Workflow:
  1. For each class (or specified classes), reads all cleaned images in dataset/train/<class>.
  2. Applies deterministic 70% Train / 15% Valid / 15% Test per-class split (seed=42).
  3. Evaluates each image with the Roboflow serverless workflow ('gemini-machine-instance-auto-label-2'):
     - Gemini Vision verifies the object class vs expected_class.
     - Class conflict gate catches mislabeled/confusable images.
     - SAM (Segment Anything Model) traces exact 2D contour polygon points.
  4. Writes confirmed polygon annotations (COCO segmentation) back to Roboflow project ('full_try').

Usage:
  python pipeline/upload_polygon_annotations.py --classes lathe
  python pipeline/upload_polygon_annotations.py --classes lathe band_saw forklift
  python pipeline/upload_polygon_annotations.py  # All 21 classes
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

import roboflow
from pipeline.config import ALL_CLASS_NAMES, CLASSES, DATASET_TRAIN_DIR, LOGS_DIR
from pipeline.roboflow_workflow import run_workflow

WORKSPACE_SLUG = "kkr-r543n"
PROJECT_ID     = "full_try-gzggk"
SPLIT_RATIOS   = {"train": 0.70, "valid": 0.15, "test": 0.15}
RANDOM_SEED    = 42
EXTS           = {".jpg", ".jpeg", ".png", ".webp"}


def assign_per_class_splits(image_paths: list[Path], seed: int = RANDOM_SEED) -> dict[str, str]:
    """
    Deterministically assigns 70% Train, 15% Valid, 15% Test for the given image list.
    Returns a mapping: filename -> split name ('train' | 'valid' | 'test').
    """
    sorted_paths = sorted(image_paths, key=lambda p: p.name)
    rng = random.Random(seed)
    shuffled = sorted_paths.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = math.ceil(n * SPLIT_RATIOS["train"])
    n_valid = math.ceil(n * SPLIT_RATIOS["valid"])

    split_map: dict[str, str] = {}
    for i, p in enumerate(shuffled):
        if i < n_train:
            split_map[p.name] = "train"
        elif i < n_train + n_valid:
            split_map[p.name] = "valid"
        else:
            split_map[p.name] = "test"

    return split_map


def process_and_upload_image(
    proj,
    img_path: Path,
    expected_class: str,
    target_split: str,
) -> dict:
    """
    Processes a single image via the Roboflow Workflow:
    - Verifies class with Gemini
    - Extracts SAM polygon points
    - Uploads COCO polygon annotation to Roboflow if verified
    """
    filename = img_path.name

    try:
        wf_resp = run_workflow(str(img_path), expected_class)
        outputs = wf_resp.get("outputs", [])
        if not outputs:
            return {
                "filename": filename,
                "class": expected_class,
                "split": target_split,
                "status": "FAIL",
                "reason": "No workflow output returned",
                "points_count": 0,
            }

        out = outputs[0]
        manual_review = out.get("manual_review_required", False)
        review_reason = out.get("review_reason", "")
        gemini_class  = out.get("gemini_class", "")

        preds = out.get("predictions", {}).get("predictions", [])

        # Accept known exact synonym variations (e.g. drill_press -> drilling)
        synonyms = {
            "drilling": {"drilling", "drill_press", "drill press"},
            "cnc_milling": {"cnc_milling"},
            "sanding_machines": {"sanding_machines", "sander", "belt_sander"},
        }
        valid_syns = synonyms.get(expected_class, {expected_class})
        if gemini_class in valid_syns and preds:
            manual_review = False

        if manual_review or not preds:
            return {
                "filename": filename,
                "class": expected_class,
                "gemini_class": gemini_class,
                "split": target_split,
                "status": "SKIPPED_REVIEW",
                "reason": review_reason or "Empty predictions / flagged by gate",
                "points_count": 0,
            }

        p0 = preds[0]
        raw_points = p0.get("points", [])
        detected_class = p0.get("class", gemini_class or expected_class)

        if not raw_points:
            return {
                "filename": filename,
                "class": expected_class,
                "gemini_class": gemini_class,
                "split": target_split,
                "status": "SKIPPED_NO_POINTS",
                "reason": "No SAM polygon points generated",
                "points_count": 0,
            }

        # Read dimensions
        w, h = 1000, 1000
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            pass

        # Flatten points for COCO format: [x1, y1, x2, y2, ...]
        flat_points = []
        xs, ys = [], []
        for pt in raw_points:
            x_val = float(pt["x"])
            y_val = float(pt["y"])
            flat_points.extend([x_val, y_val])
            xs.append(x_val)
            ys.append(y_val)

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        bbox_w = max(1.0, xmax - xmin)
        bbox_h = max(1.0, ymax - ymin)
        area = bbox_w * bbox_h

        coco_payload = {
            "images": [{"id": 1, "width": w, "height": h, "file_name": filename}],
            "categories": [{"id": 1, "name": detected_class, "supercategory": "machine"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [flat_points],
                    "area": area,
                    "bbox": [xmin, ymin, bbox_w, bbox_h],
                    "iscrowd": 0,
                }
            ],
        }

        # Write temp COCO JSON and upload
        with tempfile.TemporaryDirectory() as tmpdir:
            coco_file = Path(tmpdir) / "_annotations.coco.json"
            coco_file.write_text(json.dumps(coco_payload, indent=2), encoding="utf-8")

            res = proj.single_upload(
                image_path=str(img_path),
                annotation_path=str(coco_file),
                split=target_split,
                batch_name=expected_class,
                num_retry_uploads=3,
                annotation_overwrite=True,
            )

        anno_res = res.get("annotation", {}) if isinstance(res, dict) else {}
        is_success = anno_res.get("success", False) or isinstance(res, dict) and "duplicate" in str(res).lower()

        return {
            "filename": filename,
            "class": detected_class,
            "expected_class": expected_class,
            "split": target_split,
            "status": "OK" if is_success else "FAIL",
            "points_count": len(raw_points),
            "reason": None if is_success else f"Upload response: {res}",
        }

    except Exception as exc:
        return {
            "filename": filename,
            "class": expected_class,
            "split": target_split,
            "status": "FAIL",
            "reason": str(exc),
            "points_count": 0,
        }


def process_class(proj, class_name: str, max_workers: int = 4) -> dict:
    """Processes all images for a single class."""
    class_dir = DATASET_TRAIN_DIR / class_name
    if not class_dir.exists():
        print(f"  [WARN] Directory not found: {class_dir}")
        return {"class": class_name, "total": 0, "ok": 0, "skipped": 0, "failed": 0}

    images = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    total_imgs = len(images)
    if total_imgs == 0:
        print(f"  [WARN] No images found in: {class_dir}")
        return {"class": class_name, "total": 0, "ok": 0, "skipped": 0, "failed": 0}

    split_map = assign_per_class_splits(images, seed=RANDOM_SEED)

    train_c = sum(1 for s in split_map.values() if s == "train")
    valid_c = sum(1 for s in split_map.values() if s == "valid")
    test_c  = sum(1 for s in split_map.values() if s == "test")

    print(f"\n{'=' * 75}", flush=True)
    print(f"CLASS: [{class_name}]  |  Total Images: {total_imgs}", flush=True)
    print(f"  Target Split: Train={train_c} ({train_c/total_imgs*100:.1f}%), Valid={valid_c} ({valid_c/total_imgs*100:.1f}%), Test={test_c} ({test_c/total_imgs*100:.1f}%)", flush=True)
    print(f"{'=' * 75}", flush=True)

    items_to_process = [(img_p, class_name, split_map[img_p.name]) for img_p in images]

    results = []
    completed = 0
    ok_count = 0
    skipped_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {
            executor.submit(process_and_upload_image, proj, img_p, cls_n, spl): img_p.name
            for img_p, cls_n, spl in items_to_process
        }

        for future in as_completed(future_to_item):
            res = future.result()
            results.append(res)
            completed += 1

            status = res["status"]
            if status == "OK":
                ok_count += 1
                st_str = "OK"
            elif "SKIPPED" in status:
                skipped_count += 1
                st_str = f"SKIPPED ({res.get('reason', '')})"
            else:
                fail_count += 1
                st_str = f"FAIL ({res.get('reason', '')})"

            print(
                f"  [{completed:03d}/{total_imgs:03d}] {res['filename']:<36} | {res['split']:<5} | pts: {res.get('points_count', 0):>4} | {st_str}",
                flush=True,
            )

    # Save class log
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    class_log_path = LOGS_DIR / f"polygon_upload_{class_name}.json"
    with open(class_log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "class": class_name,
                "total_images": total_imgs,
                "successful": ok_count,
                "skipped_review": skipped_count,
                "failed": fail_count,
                "split_distribution": {"train": train_c, "valid": valid_c, "test": test_c},
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n  [{class_name} COMPLETE] Successful: {ok_count} | Skipped: {skipped_count} | Failed: {fail_count}", flush=True)
    print(f"  Log saved -> {class_log_path}", flush=True)

    return {
        "class": class_name,
        "total": total_imgs,
        "ok": ok_count,
        "skipped": skipped_count,
        "failed": fail_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Upload SAM polygon annotations verified with Gemini.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="List of classes to process. Defaults to all 21 classes if not specified.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads (default: 4).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY is not set in .env", flush=True)
        sys.exit(1)

    print("=" * 75, flush=True)
    print(f"ROBOFLOW POLYGON INSTANCE-SEGMENTATION UPLOAD PIPELINE", flush=True)
    print(f"  Workspace : {WORKSPACE_SLUG}", flush=True)
    print(f"  Project   : {PROJECT_ID}", flush=True)
    print(f"  Split     : 70% Train / 15% Valid / 15% Test (per class)", flush=True)
    print(f"  Workflow  : gemini-machine-instance-auto-label-2", flush=True)
    print("=" * 75, flush=True)

    rf = roboflow.Roboflow(api_key=api_key)
    proj = rf.workspace(WORKSPACE_SLUG).project(PROJECT_ID)

    target_classes = args.classes or ALL_CLASS_NAMES
    print(f"Target classes ({len(target_classes)}): {', '.join(target_classes)}\n", flush=True)

    summary = []
    for cls_name in target_classes:
        res = process_class(proj, cls_name, max_workers=args.workers)
        summary.append(res)

    print("\n" + "=" * 75, flush=True)
    print("FINAL UPLOAD SUMMARY ACROSS PROCESSED CLASSES", flush=True)
    print("=" * 75, flush=True)
    print(f"  {'Class':<22} {'Total':>8} {'Successful':>12} {'Skipped':>10} {'Failed':>8}", flush=True)
    print("  " + "-" * 65, flush=True)
    tot_all = tot_ok = tot_skip = tot_fail = 0
    for r in summary:
        print(f"  {r['class']:<22} {r['total']:>8} {r['ok']:>12} {r['skipped']:>10} {r['failed']:>8}", flush=True)
        tot_all += r["total"]
        tot_ok += r["ok"]
        tot_skip += r["skipped"]
        tot_fail += r["failed"]
    print("  " + "-" * 65, flush=True)
    print(f"  {'TOTAL':<22} {tot_all:>8} {tot_ok:>12} {tot_skip:>10} {tot_fail:>8}", flush=True)
    print("=" * 75, flush=True)

    summary_log = LOGS_DIR / "polygon_upload_summary.json"
    with open(summary_log, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "workspace": WORKSPACE_SLUG,
                "project": PROJECT_ID,
                "summary": summary,
            },
            f,
            indent=2,
        )
    print(f"\nOverall summary log saved -> {summary_log}", flush=True)


if __name__ == "__main__":
    main()
