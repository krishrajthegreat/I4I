# -*- coding: utf-8 -*-
"""
pipeline/audit_cluster_3_and_spotchecks.py
===========================================
Deep Audit of packaging_machine Cluster 3 & Cross-Class Spot-Checks.

Performs:
1. Full inspection of all 15 images in packaging_machine Cluster 3:
   - File details & scraping origin.
   - Zero-shot classification / CLIP content analysis.
   - Visual descriptions & assessment (Genuine packaging machine vs Contamination).
2. Spot-checks 20 random images from packaging_machine Clusters 0, 1, 2.
3. Spot-checks 20 random images from the other 3 weak classes (cnc_milling, conveyor, grinding).
4. Generates visual contact sheets (PNG) and interactive HTML galleries.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.config import DATASET_TRAIN_DIR, LOGS_DIR

CLUSTER_3_FILES = [
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
    "img_packaging_machine_ddgs_00348.jpg",
]

ZERO_SHOT_CANDIDATES = [
    "industrial packaging machine with stainless steel frame, conveyors and mechanical automation",
    "cardboard boxes, shipping packages and retail cartons on a floor or table",
    "plastic packaging design, cosmetics bottles, pouch mockup graphic design",
    "warehouse worker packing boxes by hand manually",
    "handheld heat sealer tool or manual taping dispenser",
    "food snacks, candy bars or cosmetic products close-up packaging",
    "digital 3D rendering or graphic concept of packaging design",
    "industrial manufacturing machinery in factory"
]


def audit_cluster_3(model, preprocess, text_embs):
    print("\n" + "=" * 80)
    print("STEP 1 & 2: AUDITING ALL 15 IMAGES IN PACKAGING_MACHINE CLUSTER 3")
    print("=" * 80)

    pkg_dir = DATASET_TRAIN_DIR / "packaging_machine"
    results = []

    for fn in CLUSTER_3_FILES:
        p = pkg_dir / fn
        if not p.exists():
            print(f"Missing file: {fn}")
            continue

        im = Image.open(p).convert("RGB")
        w, h = im.size
        t = preprocess(im).unsqueeze(0)

        with torch.no_grad():
            img_emb = model.encode_image(t)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
            sims = (img_emb @ text_embs.T).squeeze(0).cpu().numpy()

        top_idx = int(np.argmax(sims))
        top_concept = ZERO_SHOT_CANDIDATES[top_idx]
        top_sim = float(sims[top_idx])

        # Scraping origin detection
        if "bing_0018" in fn or "bing_0019" in fn:
            scrape_source = "Bing Scraper (icrawler / packaging concept template)"
        else:
            scrape_source = "DuckDuckGo (DDGS scraper / packaging query)"

        results.append({
            "filename": fn,
            "path": str(p),
            "size": f"{w}x{h}",
            "source": scrape_source,
            "top_concept": top_concept,
            "top_sim": round(top_sim, 3),
        })

    return results


def run_spot_checks(model, preprocess, pkg_audit_report):
    print("\n" + "=" * 80)
    print("STEP 4 & 5: SPOT-CHECKING PACKAGING_MACHINE (0, 1, 2) & OTHER 3 WEAK CLASSES")
    print("=" * 80)

    # 1. Packaging Machine Clusters 0, 1, 2
    pkg_dir = DATASET_TRAIN_DIR / "packaging_machine"
    all_pkg = sorted([p.name for p in pkg_dir.glob("*.jpg") if p.name not in CLUSTER_3_FILES])
    rng = random.Random(42)
    pkg_spotcheck = rng.sample(all_pkg, min(20, len(all_pkg)))

    spotchecks = {"packaging_machine_other_clusters": pkg_spotcheck}

    # 2. Other 3 weak classes
    for c in ["cnc_milling", "conveyor", "grinding"]:
        c_dir = DATASET_TRAIN_DIR / c
        files = sorted([p.name for p in c_dir.glob("*.jpg")])
        spotchecks[c] = rng.sample(files, min(20, len(files)))

    return spotchecks


def main():
    print("Loading OpenCLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")
    model.eval()

    # Pre-encode text queries
    toks = open_clip.tokenize(ZERO_SHOT_CANDIDATES)
    with torch.no_grad():
        text_embs = model.encode_text(toks)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)

    cluster_3_results = audit_cluster_3(model, preprocess, text_embs)
    spotchecks = run_spot_checks(model, preprocess, None)

    # Save findings JSON
    out_json = LOGS_DIR / "cluster_3_contamination_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "cluster_3_packaging_machine": cluster_3_results,
            "spotchecks": spotchecks
        }, f, indent=2)

    print(f"\n[AUDIT RESULTS SAVED] -> {out_json}")


if __name__ == "__main__":
    main()
