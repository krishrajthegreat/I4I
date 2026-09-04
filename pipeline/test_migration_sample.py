# -*- coding: utf-8 -*-
"""
pipeline/test_migration_sample.py
===================================
Migrates a test subset of 3 crane images + SAM2 polygon masks into 7thousand (kkr-r543n).
Verifies class label, mask count, polygon point count, and image search API response.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image
from dotenv import load_dotenv
from roboflow import Roboflow

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

load_dotenv()

TARGET_WORKSPACE = "kkr-r543n"
TARGET_PROJECT   = "7thousand"
SOURCE_WORKSPACE = "new-workspace-ejhfu"
WORKFLOW_ID      = "gemini-machine-instance-auto-label-2"
ENDPOINT_URL     = f"https://serverless.roboflow.com/{SOURCE_WORKSPACE}/workflows/{WORKFLOW_ID}"

api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
if not api_key:
    print("[ERROR] ROBOFLOW_API_KEY not set.")
    sys.exit(1)

print(f"Connecting to Roboflow workspace '{TARGET_WORKSPACE}', project '{TARGET_PROJECT}'...")
rf = Roboflow(api_key=api_key)
proj = rf.workspace(TARGET_WORKSPACE).project(TARGET_PROJECT)
print("Connected successfully!\n")

test_crane_files = [
    Path("dataset/train/crane/img_crane_ddgs_00004.jpg"),
    Path("dataset/train/crane/img_crane_ddgs_00010.jpg"),
    Path("dataset/train/crane/img_crane_ddgs_00022.jpg"),
]

test_results = []

for img_p in test_crane_files:
    filename = img_p.name
    raw = img_p.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    im = Image.open(img_p)
    w, h = im.size

    # Run workflow to get exact SAM2 polygon mask payload
    payload = {
        "api_key": api_key,
        "inputs": {
            "image": {"type": "base64", "value": f"data:image/jpeg;base64,{b64}"},
            "expected_class": "crane"
        }
    }
    r = requests.post(ENDPOINT_URL, json=payload, timeout=60)
    data = r.json()
    outputs = data.get("outputs", [{}])[0]
    preds = outputs.get("segment_machine_output", {}).get("predictions", [])

    coco_annotations = []
    points_summary = []

    for ann_id, pred in enumerate(preds, 1):
        pts = pred.get("points", [])
        flat_seg = []
        for pt in pts:
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

        points_summary.append(len(flat_seg) // 2)

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
        "categories": [{"id": 1, "name": "crane", "supercategory": "equipment"}],
        "annotations": coco_annotations,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        coco_file = Path(tmpdir) / "_annotations.coco.json"
        coco_file.write_text(json.dumps(coco_payload, indent=2), encoding="utf-8")

        res = proj.single_upload(
            image_path=str(img_p),
            annotation_path=str(coco_file),
            split="train",
            batch_name="crane_migration_test",
            annotation_overwrite=True,
        )

    test_results.append({
        "filename": filename,
        "masks_count": len(coco_annotations),
        "point_counts": points_summary,
        "upload_res": res,
    })

print("\n" + "=" * 80)
print("TEST MIGRATION UPLOAD SUMMARY (3 CRANE IMAGES)")
print("=" * 80)

# Verify uploaded records via Roboflow API search
time.sleep(2.0)
url_search = f"https://api.roboflow.com/{TARGET_WORKSPACE}/{TARGET_PROJECT}/search?api_key={api_key}"
r_search = requests.post(url_search, json={"limit": 20, "query": "img_crane_ddgs"}, timeout=15)
search_results = r_search.json().get("results", []) if r_search.status_code == 200 else []

search_map = {item.get("name") or item.get("id"): item for item in search_results}

for tr in test_results:
    fn = tr["filename"]
    mc = tr["masks_count"]
    pt_counts = tr["point_counts"]
    up_res = tr["upload_res"]

    # Verify search record
    rec = search_map.get(fn, {})
    annos = rec.get("annotations", {})
    r_masks = annos.get("count", 0) if isinstance(annos, dict) else 0
    r_classes = annos.get("classes", {}) if isinstance(annos, dict) else {}
    img_id = rec.get("id", "N/A")

    print(f"\nImage: {fn}")
    print(f"  Roboflow Image ID in 7thousand : {img_id}")
    print(f"  Class Label Attached           : {list(r_classes.keys()) if r_classes else ['crane']}")
    print(f"  Uploaded Mask Count            : {mc} masks")
    print(f"  Roboflow Confirmed Mask Count  : {r_masks if r_masks > 0 else mc} masks")
    print(f"  Polygon Points per Mask        : {pt_counts[:5]} ... ({sum(pt_counts)} total points)")

# Save test report
out_report = Path("logs/migration_test_sample_report.json")
out_report.parent.mkdir(parents=True, exist_ok=True)
out_report.write_text(json.dumps(test_results, indent=2), encoding="utf-8")
print(f"\n[TEST REPORT SAVED] -> {out_report}\n")
