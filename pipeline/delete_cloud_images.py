# -*- coding: utf-8 -*-
import os
import sys
import dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

dotenv.load_dotenv()
api_key = os.environ.get("ROBOFLOW_API_KEY", "")

DEL_FNS = {
    "img_crane_ddgs_00479.jpg",
    "img_fire_extinguisher_ddgs_00079.jpg",
    "img_fire_extinguisher_ddgs_00099.jpg",
    "img_fire_extinguisher_ddgs_00153.jpg",
    "img_fire_extinguisher_ddgs_00291.jpg",
}

print("Fetching all image search records from Roboflow...")
offset = 0
limit = 100
all_results = []
while True:
    url = f"https://api.roboflow.com/krish-raj-cgbcn/fire_crane/search?api_key={api_key}"
    r = requests.post(url, json={"limit": limit, "offset": offset}, timeout=10)
    res = r.json().get("results", [])
    if not res:
        break
    all_results.extend(res)
    offset += limit
    if offset >= r.json().get("total", 0):
        break

print(f"Total search records: {len(all_results)}")


def check_and_delete(item):
    img_id = item["id"]
    detail_url = f"https://api.roboflow.com/krish-raj-cgbcn/fire_crane/images/{img_id}?api_key={api_key}"
    try:
        det = requests.get(detail_url, timeout=10).json()
        im_name = det.get("image", {}).get("name", "")
        if im_name in DEL_FNS:
            print(f"MATCH FOUND: {im_name} (ID: {img_id}) -> Deleting...")
            del_url = f"https://api.roboflow.com/krish-raj-cgbcn/fire_crane/images/{img_id}?api_key={api_key}"
            del_r = requests.delete(del_url, timeout=10)
            return {"name": im_name, "id": img_id, "status": del_r.status_code}
    except Exception as e:
        pass
    return None


deleted = []
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(check_and_delete, item) for item in all_results]
    for fut in as_completed(futures):
        res = fut.result()
        if res:
            deleted.append(res)
            print(f"  DELETED FROM ROBOFLOW: {res['name']} | HTTP {res['status']}")

print(f"\nTotal mislabeled images deleted from Roboflow cloud: {len(deleted)}")

# Verify new project count
proj_url = f"https://api.roboflow.com/krish-raj-cgbcn/fire_crane?api_key={api_key}"
proj_data = requests.get(proj_url, timeout=10).json().get("project", {})
print(f"New Cloud Dataset Count: {proj_data.get('images')} images")
