# -*- coding: utf-8 -*-
import os
import sys
import dotenv
import requests

dotenv.load_dotenv()
api_key = os.environ.get("ROBOFLOW_API_KEY", "")

ws = "new-workspace-ejhfu"
proj_id = "fire_crane_2"

url_p = f"https://api.roboflow.com/{ws}/{proj_id}?api_key={api_key}"
r_p = requests.get(url_p, timeout=10).json().get("project", {})

print("=== ROBOFLOW PROJECT STATUS ===")
print("Project Name:", r_p.get("name"))
print("Total Images in Split Dataset:", r_p.get("images"))
print("Unannotated in Queue:", r_p.get("unannotated"))
print("Annotated Classes:", r_p.get("classes"))
print("Splits:", r_p.get("splits"))

url_b = f"https://api.roboflow.com/{ws}/{proj_id}/batches?api_key={api_key}"
r_b = requests.get(url_b, timeout=10).json()
print("\n=== BATCHES STATUS ===")
for b in r_b.get("batches", []):
    bname = b.get("name")
    bid = b.get("id")
    jobs = b.get("numJobs")
    imgs = b.get("images")
    print(f"  Batch '{bname}' ({bid}) -> numJobs: {jobs}, total images: {imgs}")

url_s = f"https://api.roboflow.com/{ws}/{proj_id}/search?api_key={api_key}"
r_s = requests.post(url_s, json={"limit": 20, "offset": 0}, timeout=10).json()
print("\n=== SAMPLE PROCESSED IMAGES ===")
for item in r_s.get("results", [])[:15]:
    annos = item.get("annotations", {})
    cnt = annos.get("count", 0) if isinstance(annos, dict) else 0
    classes = annos.get("classes", {}) if isinstance(annos, dict) else {}
    split = item.get("split", "unassigned")
    img_id = item.get("id")
    print(f"  Image ID {img_id:<20} [{split:<6}] -> masks: {cnt:<2} | classes: {classes}")
