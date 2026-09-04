# -*- coding: utf-8 -*-
"""
pipeline/sweep_packaging_machine.py
====================================
Exhaustive Sweep of all images in dataset/train/packaging_machine
against non-machine packaging concepts (cardboard boxes, mockups, retail goods)
to verify if any other contamination exists beyond the 14 in Cluster 3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.config import DATASET_TRAIN_DIR

CONTAMINATION_QUERIES = [
    "cardboard boxes stacked on floor or table without machinery",
    "retail product packaging design mockup with cosmetic bottles or jars",
    "empty cardboard shipping carton postal delivery package",
    "kraft paper gift box with ribbon flat-pack template",
    "consumer food snack bag chips pouch graphic design"
]

GENUINE_MACHINE_QUERIES = [
    "industrial automated packaging machine with conveyor and stainless steel frame",
    "factory automated bagging flow wrapper vertical form fill seal machine",
    "industrial bottling and capping packaging machinery in manufacturing plant"
]

KNOWN_14 = {
    "img_packaging_machine_bing_00180.jpg",
    "img_packaging_machine_bing_00181.jpg",
    "img_packaging_machine_bing_00182.jpg",
    "img_packaging_machine_bing_00184.jpg",
    "img_packaging_machine_bing_00185.jpg",
    "img_packaging_machine_bing_00186.jpg",
    "img_packaging_machine_bing_00187.jpg",
    "img_packaging_machine_bing_00189.jpg",
    "img_packaging_machine_bing_00191.jpg",
    "img_packaging_machine_bing_00192.jpg",
    "img_packaging_machine_bing_00193.jpg",
    "img_packaging_machine_bing_00194.jpg",
    "img_packaging_machine_bing_00195.jpg",
    "img_packaging_machine_bing_00198.jpg",
}


def sweep_all_packaging():
    print("Loading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")
    model.eval()

    tok_bad = open_clip.tokenize(CONTAMINATION_QUERIES)
    tok_good = open_clip.tokenize(GENUINE_MACHINE_QUERIES)

    with torch.no_grad():
        emb_bad = model.encode_text(tok_bad)
        emb_bad /= emb_bad.norm(dim=-1, keepdim=True)

        emb_good = model.encode_text(tok_good)
        emb_good /= emb_good.norm(dim=-1, keepdim=True)

    pkg_dir = DATASET_TRAIN_DIR / "packaging_machine"
    all_files = sorted(list(pkg_dir.glob("*.jpg")))
    print(f"\nSweeping all {len(all_files)} images in packaging_machine...")

    flagged_outside_14 = []
    scores_known_14 = []

    for p in all_files:
        try:
            im = Image.open(p).convert("RGB")
            t = preprocess(im).unsqueeze(0)
        except Exception:
            continue

        with torch.no_grad():
            img_emb = model.encode_image(t)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)

            sim_bad = float((img_emb @ emb_bad.T).max())
            sim_good = float((img_emb @ emb_good.T).max())

        diff = sim_bad - sim_good  # Positive means looks more like box/mockup than machine

        if p.name in KNOWN_14:
            scores_known_14.append((p.name, sim_bad, sim_good, diff))
        else:
            # Check if any image outside the 14 is heavily leaning towards non-machine box/mockup
            if diff > 0.02 or sim_bad > 0.65:
                flagged_outside_14.append((p.name, sim_bad, sim_good, diff))

    print(f"\n--- Known 14 Contamination Scores (Cluster 3) ---")
    for fn, sb, sg, d in scores_known_14:
        print(f"  {fn}: Bad Sim={sb:.3f}, Good Sim={sg:.3f}, Margin={d:+.3f}")

    print(f"\n--- Images Outside the 14 with High Box/Mockup Similarity ---")
    if not flagged_outside_14:
        print("  NONE! All other 404 images in packaging_machine strongly match genuine industrial packaging machines (Good Sim >> Bad Sim).")
    else:
        for fn, sb, sg, d in flagged_outside_14:
            print(f"  {fn}: Bad Sim={sb:.3f}, Good Sim={sg:.3f}, Margin={d:+.3f}")

    return len(flagged_outside_14)


if __name__ == "__main__":
    sweep_all_packaging()
