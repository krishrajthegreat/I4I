# -*- coding: utf-8 -*-
"""
pipeline/annotate_fire_crane_2.py
==================================
Accurate Multi-Instance SAM2 Polygon Annotation Upload for fire_crane_2.

Fixes applied:
- Extracts exact SAM2 contour polygon coordinates from pred['points'] (100-300 points per mask).
- Formats valid COCO segmentation polygons [[x1, y1, x2, y2, ...]].
- Uploads images + polygon segmentation annotations to new-workspace-ejhfu / fire_crane_2 with annotation_overwrite=True.
"""

from __future__ import annotations

import base64
import json
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image
from dotenv import load_dotenv
from roboflow import Roboflow

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

WORKSPACE_SLUG = "new-workspace-ejhfu"
PROJECT_ID     = "fire_crane_2"
WORKFLOW_ID    = "gemini-machine-instance-auto-label-2"
ENDPOINT_URL   = f"https://serverless.roboflow.com/{WORKSPACE_SLUG}/workflows/{WORKFLOW_ID}"
TARGET_CLASSES = ["crane", "fire_extinguisher"]
SPLIT_RATIOS   = {"train": 0.70, "valid": 0.15, "test": 0.15}
RANDOM_SEED    = 42
NUM_WORKERS    = 12
DATASET_DIR    = Path("dataset/train")
LOGS_DIR       = Path("logs")


def run_serverless_workflow(img_path: Path, expected_class: str, api_key: str) -> dict:
    raw = img_path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    suffix = img_path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(suffix, "image/jpeg")

    payload = {
        "api_key": api_key,
        "inputs": {
            "image": {"type": "base64", "value": f"data:{mime};base64,{b64}"},
            "expected_class": expected_class
        }
    }

    r = requests.post(ENDPOINT_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def process_image_and_upload(proj, img_path: Path, expected_class: str, split: str, api_key: str) -> dict:
    filename = img_path.name
    try:
        im = Image.open(img_path)
        w, h = im.size
    except Exception as e:
        return {"filename": filename, "class": expected_class, "split": split, "status": "FAIL", "reason": f"Could not open image: {e}", "masks": 0}

    outputs = None
    for attempt in range(3):
        try:
            wf_resp = run_serverless_workflow(img_path, expected_class, api_key)
            out_list = wf_resp.get("outputs", [])
            if out_list:
                outputs = out_list[0]
                break
        except Exception as e:
            if attempt == 2:
                return {"filename": filename, "class": expected_class, "split": split, "status": "FAIL", "reason": f"Workflow failed: {e}", "masks": 0}
            time.sleep(1.0)

    if not outputs:
        return {"filename": filename, "class": expected_class, "split": split, "status": "FAIL", "reason": "Empty workflow outputs", "masks": 0}

    seg_output = outputs.get("segment_machine_output", {})
    predictions = []
    if isinstance(seg_output, dict):
        predictions = seg_output.get("predictions", [])
    elif isinstance(seg_output, list):
        predictions = seg_output

    coco_annotations = []
    for ann_id, pred in enumerate(predictions, 1):
        points = pred.get("points", [])
        flat_seg = []
        if isinstance(points, list):
            for pt in points:
                if isinstance(pt, dict):
                    flat_seg.extend([float(pt.get("x", 0)), float(pt.get("y", 0))])
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    flat_seg.extend([float(pt[0]), float(pt[1])])

        # Bounding box
        pw = float(pred.get("width", 0))
        ph = float(pred.get("height", 0))
        px = float(pred.get("x", 0)) - pw / 2.0
        py = float(pred.get("y", 0)) - ph / 2.0

        if len(flat_seg) < 6:
            flat_seg = [px, py, px + pw, py, px + pw, py + ph, px, py + ph]

        coco_annotations.append({
            "id": ann_id,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [flat_seg],
            "area": float(pw * ph),
            "bbox": [px, py, pw, ph],
            "iscrowd": 0,
        })

    coco_payload = {
        "images": [{"id": 1, "width": w, "height": h, "file_name": filename}],
        "categories": [{"id": 1, "name": expected_class, "supercategory": "equipment"}],
        "annotations": coco_annotations,
    }

    upload_ok = False
    with tempfile.TemporaryDirectory() as tmpdir:
        coco_file = Path(tmpdir) / "_annotations.coco.json"
        coco_file.write_text(json.dumps(coco_payload, indent=2), encoding="utf-8")

        try:
            res = proj.single_upload(
                image_path=str(img_path),
                annotation_path=str(coco_file),
                split=split,
                annotation_overwrite=True,
            )
            if isinstance(res, dict):
                anno_res = res.get("annotation", {})
                if isinstance(anno_res, dict) and anno_res.get("success", False):
                    upload_ok = True
                elif res.get("success", False) or res.get("id"):
                    upload_ok = True
        except Exception:
            upload_ok = False

    return {
        "filename": filename,
        "class": expected_class,
        "split": split,
        "status": "OK" if upload_ok else "UPLOAD_RETRY",
        "masks": len(coco_annotations),
    }


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY not set.")
        sys.exit(1)

    print(f"Connecting to Roboflow workspace '{WORKSPACE_SLUG}', project '{PROJECT_ID}'...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(WORKSPACE_SLUG).project(PROJECT_ID)
    print("Connected successfully!\n")

    summary_data = {}

    for cls_name in TARGET_CLASSES:
        cls_dir = DATASET_DIR / cls_name
        files = sorted([p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        total = len(files)

        rng = random.Random(RANDOM_SEED)
        shuffled = list(files)
        rng.shuffle(shuffled)

        n_train = int(round(total * SPLIT_RATIOS["train"]))
        n_valid = int(round(total * SPLIT_RATIOS["valid"]))

        split_map = {}
        for idx, p in enumerate(shuffled):
            if idx < n_train:
                split_map[p] = "train"
            elif idx < n_train + n_valid:
                split_map[p] = "valid"
            else:
                split_map[p] = "test"

        print(f"Processing and uploading exact SAM2 polygon masks for {total} images of [{cls_name.upper()}]...")
        completed = 0
        cls_masks_total = 0
        cls_success = 0

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(process_image_and_upload, proj, p, cls_name, split_map[p], api_key): p
                for p in files
            }
            for fut in as_completed(futures):
                res = fut.result()
                completed += 1
                if res["masks"] > 0:
                    cls_success += 1
                    cls_masks_total += res["masks"]

                if completed % 20 == 0 or completed == total:
                    fn = res.get("filename", "unknown")
                    sp = res.get("split", "train")
                    mc = res.get("masks", 0)
                    st = res.get("status", "OK")
                    print(f"  [{completed:03d}/{total:03d}] {fn:<36} | {sp:<5} | {mc:>2} masks -> {st}", flush=True)

        avg_m = cls_masks_total / total if total else 0
        summary_data[cls_name] = {
            "total_images": total,
            "images_with_masks": cls_success,
            "total_masks": cls_masks_total,
            "avg_masks_per_image": round(avg_m, 2)
        }

    print("\n" + "=" * 80)
    print("SAM2 POLYGON ANNOTATION & UPLOAD SUMMARY")
    print("=" * 80)
    for c, s in summary_data.items():
        print(f"  {c:<20} | Images: {s['total_images']} | Total SAM2 Masks: {s['total_masks']} | Avg Masks/Image: {s['avg_masks_per_image']}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = LOGS_DIR / "sam2_polygon_annotation_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "summary": summary_data}, f, indent=2)
    print(f"\n[REPORT SAVED] -> {out_json}\n")


if __name__ == "__main__":
    main()
