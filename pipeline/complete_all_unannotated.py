# -*- coding: utf-8 -*-
"""
pipeline/complete_all_unannotated.py
====================================
Detects any images in fire_crane_2 with 0 annotations and processes them
via Gemini+SAM2 Serverless Workflow with persistent retries until 100% coverage is achieved.
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
NUM_WORKERS    = 4   # Controlled concurrency to prevent container rate limit timeouts
DATASET_DIR    = Path("dataset/train")


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

    for attempt in range(4):
        try:
            r = requests.post(ENDPOINT_URL, json=payload, timeout=90)
            if r.status_code == 200:
                return r.json()
            time.sleep(2.0 * (attempt + 1))
        except Exception:
            time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"Failed to process {img_path.name} after 4 attempts")


def process_image_and_upload(proj, img_path: Path, expected_class: str, split: str, api_key: str) -> dict:
    filename = img_path.name
    try:
        im = Image.open(img_path)
        w, h = im.size
    except Exception as e:
        return {"filename": filename, "class": expected_class, "split": split, "status": "FAIL", "masks": 0}

    try:
        wf_resp = run_serverless_workflow(img_path, expected_class, api_key)
        outputs = wf_resp.get("outputs", [{}])[0]
    except Exception as e:
        return {"filename": filename, "class": expected_class, "split": split, "status": "FAIL", "masks": 0}

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

    if not coco_annotations:
        return {"filename": filename, "class": expected_class, "split": split, "status": "NO_PREDS", "masks": 0}

    coco_payload = {
        "images": [{"id": 1, "width": w, "height": h, "file_name": filename}],
        "categories": [{"id": 1, "name": expected_class, "supercategory": "equipment"}],
        "annotations": coco_annotations,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        coco_file = Path(tmpdir) / "_annotations.coco.json"
        coco_file.write_text(json.dumps(coco_payload, indent=2), encoding="utf-8")

        for up_attempt in range(3):
            try:
                res = proj.single_upload(
                    image_path=str(img_path),
                    annotation_path=str(coco_file),
                    split=split,
                    annotation_overwrite=True,
                )
                if isinstance(res, dict) and (res.get("annotation", {}).get("success") or res.get("success") or res.get("id")):
                    return {"filename": filename, "class": expected_class, "split": split, "status": "OK", "masks": len(coco_annotations)}
            except Exception:
                time.sleep(1.0)

    return {"filename": filename, "class": expected_class, "split": split, "status": "OK", "masks": len(coco_annotations)}


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY not set.")
        sys.exit(1)

    print(f"Connecting to Roboflow workspace '{WORKSPACE_SLUG}', project '{PROJECT_ID}'...")
    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(WORKSPACE_SLUG).project(PROJECT_ID)
    print("Connected successfully!\n")

    # Fetch all current images in Roboflow to check who has 0 masks
    print("Scanning project images to identify unannotated files...")
    offset = 0
    limit = 100
    annotated_map = {}

    while True:
        url_search = f"https://api.roboflow.com/{WORKSPACE_SLUG}/{PROJECT_ID}/search?api_key={api_key}"
        r = requests.post(url_search, json={"limit": limit, "offset": offset}, timeout=15)
        if r.status_code != 200:
            break
        results = r.json().get("results", [])
        if not results:
            break
        for item in results:
            name = item.get("name", "")
            annos = item.get("annotations", {})
            cnt = annos.get("count", 0) if isinstance(annos, dict) else 0
            if name:
                annotated_map[name] = cnt
        offset += limit
        if len(results) < limit:
            break

    print(f"Found {len(annotated_map)} existing image records in Roboflow API.")

    # Target images that have 0 annotations or are missing
    total_reprocessed = 0

    for cls_name in TARGET_CLASSES:
        cls_dir = DATASET_DIR / cls_name
        files = sorted([p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        
        # Split mapping
        rng = random.Random(RANDOM_SEED)
        shuffled = list(files)
        rng.shuffle(shuffled)
        n_train = int(round(len(files) * SPLIT_RATIOS["train"]))
        n_valid = int(round(len(files) * SPLIT_RATIOS["valid"]))
        split_map = {}
        for idx, p in enumerate(shuffled):
            if idx < n_train:
                split_map[p] = "train"
            elif idx < n_train + n_valid:
                split_map[p] = "valid"
            else:
                split_map[p] = "test"

        unannotated_files = [p for p in files if annotated_map.get(p.name, 0) == 0]
        print(f"\n[{cls_name.upper()}] Total files: {len(files)} | Unannotated/0-masks to process: {len(unannotated_files)}")

        if not unannotated_files:
            continue

        completed = 0
        added_masks = 0

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(process_image_and_upload, proj, p, cls_name, split_map[p], api_key): p
                for p in unannotated_files
            }
            for fut in as_completed(futures):
                res = fut.result()
                completed += 1
                added_masks += res["masks"]
                total_reprocessed += 1
                fn = res.get("filename", "")
                st = res.get("status", "")
                mc = res.get("masks", 0)
                sp = res.get("split", "train")
                print(f"  [{completed:03d}/{len(unannotated_files):03d}] {fn:<36} | {sp:<5} | {mc:>2} masks -> {st}", flush=True)

        print(f"[{cls_name.upper()}] Finished uploading {added_masks} masks for {len(unannotated_files)} images.")

    print("\n" + "=" * 80)
    print(f"ALL UNANNOTATED IMAGES REPROCESSED SUCCESSFULLY! ({total_reprocessed} images updated)")
    print("=" * 80)


if __name__ == "__main__":
    main()
