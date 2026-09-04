# -*- coding: utf-8 -*-
import os
import base64
import json
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("ROBOFLOW_API_KEY", "")

ws = "new-workspace-ejhfu"
wf_id = "gemini-machine-instance-auto-label-2"
url = f"https://serverless.roboflow.com/{ws}/workflows/{wf_id}"

zero_mask_samples = [
    ("crane", "dataset/train/crane/img_crane_ddgs_00082.jpg"),
    ("crane", "dataset/train/crane/img_crane_ddgs_00103.jpg"),
    ("crane", "dataset/train/crane/img_crane_ddgs_00181.jpg"),
    ("crane", "dataset/train/crane/img_crane_ddgs_00210.jpg"),
    ("crane", "dataset/train/crane/img_crane_ddgs_00303.jpg"),
    ("fire_extinguisher", "dataset/train/fire_extinguisher/img_fire_extinguisher_ddgs_00133.jpg"),
    ("fire_extinguisher", "dataset/train/fire_extinguisher/img_fire_extinguisher_ddgs_00152.jpg"),
    ("fire_extinguisher", "dataset/train/fire_extinguisher/img_fire_extinguisher_ddgs_00262.jpg"),
    ("fire_extinguisher", "dataset/train/fire_extinguisher/img_fire_extinguisher_ddgs_00306.jpg"),
    ("fire_extinguisher", "dataset/train/fire_extinguisher/img_fire_extinguisher_ddgs_00331.jpg"),
]

print("=== DIAGNOSING 0-MASK IMAGES IN WORKFLOW ===")

for cls_name, p_str in zero_mask_samples:
    p = Path(p_str)
    if not p.exists():
        continue
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    payload = {
        "api_key": api_key,
        "inputs": {
            "image": {"type": "base64", "value": f"data:image/jpeg;base64,{b64}"},
            "expected_class": cls_name
        }
    }
    try:
        r = requests.post(url, json=payload, timeout=40)
        data = r.json()
        outputs = data.get("outputs", [{}])[0]
        manual_review = outputs.get("manual_review_required")
        gem_cls = outputs.get("gemini_class")
        reason = outputs.get("review_reason")
        g_boxes = outputs.get("gemini_boxes", {}).get("predictions", [])
        seg = outputs.get("segment_machine_output", {}).get("predictions", [])
        
        print(f"\n[{cls_name.upper()}] {p.name}:")
        print(f"  Gemini detected: {len(g_boxes)} boxes | gemini_class: '{gem_cls}'")
        print(f"  Manual Review: {manual_review} | Reason: '{reason}'")
        print(f"  SAM2 output: {len(seg)} masks")
    except Exception as e:
        print(f"Error on {p.name}: {e}")
