# -*- coding: utf-8 -*-
"""
pipeline/audit_conveyor_cluster3.py
====================================
K-Means clustering (k=4) on conveyor dataset (dataset/train/conveyor).
Generates contact sheet PNG and HTML report for Conveyor Cluster 3.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model.to(device)
model.eval()

conv_dir = Path("dataset/train/conveyor")
files = sorted([p for p in conv_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
print(f"Loading {len(files)} conveyor images for clustering...")

tensors = []
valid_files = []
for p in files:
    try:
        tensors.append(preprocess(Image.open(p).convert("RGB")))
        valid_files.append(p)
    except Exception:
        pass

all_embs = []
batch_size = 32
for i in range(0, len(tensors), batch_size):
    b = torch.stack(tensors[i:i+batch_size]).to(device)
    with torch.no_grad():
        embs = model.encode_image(b)
        embs /= embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs.numpy())

X = np.vstack(all_embs)

# K-Means k=4 with seed=42
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

clusters = {i: [] for i in range(4)}
for p, lbl in zip(valid_files, labels):
    clusters[lbl].append(p)

print("\nConveyor Cluster Sizes (k=4):")
for cid, p_list in clusters.items():
    print(f"  Cluster {cid}: {len(p_list)} images ({len(p_list)/len(valid_files)*100:.1f}%)")

c3_files = clusters[3]
print(f"\nCluster 3 has {len(c3_files)} images.")

# Save image list to json
out_json = Path("logs/conveyor_cluster_3_files.json")
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps([p.name for p in c3_files], indent=2), encoding="utf-8")

# Generate composite PNG grid contact sheet for Cluster 3
cols = 5
thumb_size = (220, 220)
rows = math.ceil(len(c3_files) / cols)
header_h = 50
footer_h = 30
grid_w = cols * thumb_size[0]
grid_h = header_h + (rows * (thumb_size[1] + footer_h))

grid_img = Image.new("RGB", (grid_w, grid_h), (15, 23, 42))
draw = ImageDraw.Draw(grid_img)

draw.rectangle([(0, 0), (grid_w, header_h)], fill=(30, 41, 59))
draw.text((15, 15), f"CONVEYOR -- Cluster #3 ({len(c3_files)} images)", fill=(56, 189, 248))

for idx, p in enumerate(c3_files):
    r = idx // cols
    c = idx % cols
    x = c * thumb_size[0]
    y = header_h + (r * (thumb_size[1] + footer_h))

    try:
        with Image.open(p) as img:
            thumb = img.convert("RGB")
            thumb.thumbnail(thumb_size)
            # Create box
            tile = Image.new("RGB", thumb_size, (30, 41, 59))
            ox = (thumb_size[0] - thumb.width) // 2
            oy = (thumb_size[1] - thumb.height) // 2
            tile.paste(thumb, (ox, oy))
            grid_img.paste(tile, (x, y))
            draw.text((x + 5, y + thumb_size[1] + 5), p.name[:28], fill=(226, 232, 240))
    except Exception:
        pass

png_path = Path("logs/conveyor_cluster_3_grid.png")
grid_img.save(png_path)
print(f"Cluster 3 Contact Sheet saved to: {png_path}")
