# -*- coding: utf-8 -*-
"""
pipeline/clip_mask_verification.py
===================================
Class-Verification Mask Filtering Pipeline using Diversified OpenCLIP Reference Banks.

Features:
- STEP 1: Per-mask tight crop embedding & scoring (strict mask bbox isolation).
- STEP 2: Diversified reference banks:
    - Crane (28 prototypes): 50/50 balance across overhead bridge, gantry, jib, mobile, crawler, tower, port container.
    - Fire Extinguisher (32 prototypes): wall-mounted (concrete/brick), floor-stand, cabinet, CO2 horn, wheeled trolley, vehicle mount, workshop shadows (photos only, zero infographics).
- STEP 3: Calibration grids showing isolated TIGHT CROPS of each candidate mask patch.
- STEP 4: Distribution analysis & threshold simulation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import FastSAM

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.config import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DATASET_TRAIN_DIR,
    LOGS_DIR,
)

TARGET_CLASSES = ["crane", "fire_extinguisher"]

# Carefully curated balanced reference banks (genuine photos only, zero infographics)
BALANCED_REFERENCE_SEEDS = {
    "crane": [
        # Indoor Overhead Bridge & Traveling (4)
        "img_crane_ddgs_00423.jpg", "img_crane_ddgs_00047.jpg", "img_crane_ddgs_00077.jpg", "img_crane_ddgs_00219.jpg",
        # Indoor Workshop Jib / Wall-Mounted (4)
        "img_crane_ddgs_00239.jpg", "img_crane_ddgs_00252.jpg", "img_crane_ddgs_00242.jpg", "img_crane_ddgs_00237.jpg",
        # Mobile Hydraulic / Rough Terrain Truck (4)
        "img_crane_ddgs_00124.jpg", "img_crane_ddgs_00129.jpg", "img_crane_ddgs_00138.jpg", "img_crane_ddgs_00116.jpg",
        # Heavy Crawler / Lattice Boom (4)
        "img_crane_ddgs_00190.jpg", "img_crane_ddgs_00187.jpg", "img_crane_ddgs_00164.jpg", "img_crane_ddgs_00121.jpg",
        # Tall Tower / Construction Skyline (4)
        "img_crane_ddgs_00390.jpg", "img_crane_ddgs_00384.jpg", "img_crane_ddgs_00167.jpg", "img_crane_ddgs_00161.jpg",
        # Rail-Mounted Gantry Yard (4)
        "img_crane_ddgs_00098.jpg", "img_crane_ddgs_00049.jpg", "img_crane_ddgs_00054.jpg", "img_crane_ddgs_00045.jpg",
        # Shipyard & Port Container Gantry (4)
        "img_crane_ddgs_00055.jpg", "img_crane_ddgs_00078.jpg", "img_crane_ddgs_00063.jpg", "img_crane_ddgs_00043.jpg"
    ],
    "fire_extinguisher": [
        # Wall-Mounted Concrete (4)
        "img_fire_extinguisher_ddgs_00476.jpg", "img_fire_extinguisher_ddgs_00262.jpg", "img_fire_extinguisher_ddgs_00034.jpg", "img_fire_extinguisher_ddgs_00047.jpg",
        # Wall-Mounted Brick (4)
        "img_fire_extinguisher_ddgs_00019.jpg", "img_fire_extinguisher_ddgs_00503.jpg", "img_fire_extinguisher_ddgs_00041.jpg", "img_fire_extinguisher_ddgs_00073.jpg",
        # Floor Stand / Standalone (4)
        "img_fire_extinguisher_ddgs_00044.jpg", "img_fire_extinguisher_ddgs_00045.jpg", "img_fire_extinguisher_ddgs_00117.jpg", "img_fire_extinguisher_ddgs_00136.jpg",
        # Emergency Metal Cabinet Mount (4)
        "img_fire_extinguisher_ddgs_00122.jpg", "img_fire_extinguisher_ddgs_00269.jpg", "img_fire_extinguisher_ddgs_00061.jpg", "img_fire_extinguisher_ddgs_00270.jpg",
        # Carbon Dioxide CO2 Black Horn Type (4)
        "img_fire_extinguisher_ddgs_00190.jpg", "img_fire_extinguisher_ddgs_00167.jpg", "img_fire_extinguisher_ddgs_00495.jpg", "img_fire_extinguisher_ddgs_00507.jpg",
        # Heavy Wheeled Mobile Trolley (4)
        "img_fire_extinguisher_ddgs_00109.jpg", "img_fire_extinguisher_ddgs_00114.jpg", "img_fire_extinguisher_ddgs_00218.jpg", "img_fire_extinguisher_ddgs_00194.jpg",
        # Vehicle & Machine Chassis Bracket (4)
        "img_fire_extinguisher_ddgs_00289.jpg", "img_fire_extinguisher_ddgs_00093.jpg", "img_fire_extinguisher_ddgs_00170.jpg", "img_fire_extinguisher_ddgs_00056.jpg",
        # Workshop Shadow / Industrial Lighting (4)
        "img_fire_extinguisher_ddgs_00132.jpg", "img_fire_extinguisher_ddgs_00327.jpg", "img_fire_extinguisher_ddgs_00181.jpg", "img_fire_extinguisher_ddgs_00053.jpg"
    ]
}


def load_clip_model():
    print(f"Loading OpenCLIP ({CLIP_MODEL_NAME} / {CLIP_PRETRAINED}) on CPU...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device="cpu"
    )
    model.eval()
    return model, preprocess


def build_balanced_reference_banks(model, preprocess) -> dict[str, np.ndarray]:
    """STEP 2: Embeds balanced reference images into per-class reference banks."""
    print("\n" + "=" * 80)
    print("STEP 2: BUILDING DIVERSIFIED & BALANCED REFERENCE EMBEDDING BANKS")
    print("=" * 80)

    ref_banks = {}
    for cls_name in TARGET_CLASSES:
        cls_dir = DATASET_TRAIN_DIR / cls_name
        seed_names = BALANCED_REFERENCE_SEEDS.get(cls_name, [])

        ref_paths = []
        for fn in seed_names:
            p = cls_dir / fn
            if p.exists():
                ref_paths.append(p)

        print(f"  [{cls_name.upper()}] Curated {len(ref_paths)} balanced reference prototype photos.")
        
        tensors = []
        for p in ref_paths:
            img = Image.open(p).convert("RGB")
            t = preprocess(img)
            tensors.append(t)

        with torch.no_grad():
            batch = torch.stack(tensors)
            embs = model.encode_image(batch)
            embs /= embs.norm(dim=-1, keepdim=True)
            ref_matrix = embs.cpu().numpy()

        ref_banks[cls_name] = ref_matrix
        print(f"  [{cls_name.upper()}] Reference Bank Matrix Shape: {ref_matrix.shape}")

    return ref_banks


def score_all_candidate_masks(model, preprocess, ref_banks: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    """STEP 1 & 2: Proposes FastSAM candidate masks, crops each mask bbox tightly,
    embeds crop with CLIP, and computes max-cosine similarity against reference bank."""
    print("\n" + "=" * 80)
    print("STEP 1 & 2: SCORING INDIVIDUAL MASK CROPS WITH CLIP MAX-SIMILARITY")
    print("=" * 80)

    sam = FastSAM("FastSAM-s.pt")
    all_scored_masks = []
    class_stats = {}

    for cls_name in TARGET_CLASSES:
        cls_dir = DATASET_TRAIN_DIR / cls_name
        ref_matrix = ref_banks[cls_name]  # (N_ref, 512)
        
        files = sorted([p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        print(f"\nScoring {len(files)} images for [{cls_name.upper()}]...")

        cls_scores = []
        cls_mask_count = 0

        for img_idx, p in enumerate(files, 1):
            try:
                img = Image.open(p).convert("RGB")
                w, h = img.size
                total_area = w * h
            except Exception:
                continue

            results = sam(str(p), device="cpu", retina_masks=True, imgsz=640, conf=0.15, verbose=False)

            candidate_polys = []
            for r in results:
                if r.masks is not None:
                    for poly in r.masks.xy:
                        if len(poly) >= 3:
                            xs = poly[:, 0]
                            ys = poly[:, 1]
                            bw = max(1.0, xs.max() - xs.min())
                            bh = max(1.0, ys.max() - ys.min())
                            area = bw * bh
                            area_frac = area / total_area
                            if 0.02 <= area_frac <= 0.85:
                                candidate_polys.append((poly, area, (xs.min(), ys.min(), xs.max(), ys.max()), bw, bh))

            if not candidate_polys:
                continue

            # NMS Deduplication
            candidate_polys.sort(key=lambda x: x[1], reverse=True)
            kept_instances = []
            for poly, area, box, bw, bh in candidate_polys:
                overlap = False
                for _, _, kbox, _, _ in kept_instances:
                    ix1 = max(box[0], kbox[0])
                    iy1 = max(box[1], kbox[1])
                    ix2 = min(box[2], kbox[2])
                    iy2 = min(box[3], kbox[3])
                    iw = max(0.0, ix2 - ix1)
                    ih = max(0.0, iy2 - iy1)
                    inter = iw * ih
                    karea = (kbox[2] - kbox[0]) * (kbox[3] - kbox[1])
                    union = area + karea - inter
                    iou = inter / union if union > 0 else 0
                    if iou > 0.60:
                        overlap = True
                        break
                if not overlap:
                    kept_instances.append((poly, area, box, bw, bh))

            kept_instances = kept_instances[:15]

            # Crop each candidate mask bbox and embed crop with CLIP
            crop_tensors = []
            crop_meta = []

            for m_idx, (poly, area, box, bw, bh) in enumerate(kept_instances):
                # Add 5% context padding around tight bbox
                pad_x = int(bw * 0.05)
                pad_y = int(bh * 0.05)
                x1 = max(0, int(box[0]) - pad_x)
                y1 = max(0, int(box[1]) - pad_y)
                x2 = min(w, int(box[2]) + pad_x)
                y2 = min(h, int(box[3]) + pad_y)

                if (x2 - x1) < 8 or (y2 - y1) < 8:
                    continue

                crop = img.crop((x1, y1, x2, y2))
                crop_tensors.append(preprocess(crop))
                crop_meta.append((poly, area, box, bw, bh, (x1, y1, x2, y2)))

            if not crop_tensors:
                continue

            with torch.no_grad():
                batch = torch.stack(crop_tensors)
                crop_embs = model.encode_image(batch)
                crop_embs /= crop_embs.norm(dim=-1, keepdim=True)
                crop_mat = crop_embs.cpu().numpy()  # (M, 512)

            # Compute Cosine Similarities against reference bank (M, N_ref)
            sim_matrix = np.dot(crop_mat, ref_matrix.T)
            max_sims = np.max(sim_matrix, axis=1)  # (M,)

            for i, sim in enumerate(max_sims):
                poly, area, box, bw, bh, crop_box = crop_meta[i]
                flat_pts = []
                for pt in poly:
                    flat_pts.extend([float(pt[0]), float(pt[1])])

                item = {
                    "class": cls_name,
                    "image_name": p.name,
                    "image_path": str(p),
                    "mask_idx": i,
                    "sim_score": round(float(sim), 4),
                    "area_frac": round(float(area / total_area), 4),
                    "bbox": [float(box[0]), float(box[1]), float(bw), float(bh)],
                    "crop_box": list(crop_box),
                    "points_count": len(poly),
                    "segmentation": [flat_pts],
                }
                all_scored_masks.append(item)
                cls_scores.append(float(sim))
                cls_mask_count += 1

            if img_idx % 40 == 0 or img_idx == len(files):
                print(f"  [{img_idx:03d}/{len(files):03d}] {cls_name:<18} -> {cls_mask_count} mask crops scored", flush=True)

        scores_arr = np.array(cls_scores)
        class_stats[cls_name] = {
            "total_masks": cls_mask_count,
            "min_score": round(float(np.min(scores_arr)), 4) if len(scores_arr) else 0,
            "max_score": round(float(np.max(scores_arr)), 4) if len(scores_arr) else 0,
            "median_score": round(float(np.median(scores_arr)), 4) if len(scores_arr) else 0,
            "mean_score": round(float(np.mean(scores_arr)), 4) if len(scores_arr) else 0,
            "p10": round(float(np.percentile(scores_arr, 10)), 4) if len(scores_arr) else 0,
            "p25": round(float(np.percentile(scores_arr, 25)), 4) if len(scores_arr) else 0,
            "p50": round(float(np.percentile(scores_arr, 50)), 4) if len(scores_arr) else 0,
            "p75": round(float(np.percentile(scores_arr, 75)), 4) if len(scores_arr) else 0,
            "p90": round(float(np.percentile(scores_arr, 90)), 4) if len(scores_arr) else 0,
        }

    # Save to JSON and CSV
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LOGS_DIR / "mask_clip_scores.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "stats": class_stats, "masks": all_scored_masks}, f, indent=2)

    csv_path = LOGS_DIR / "mask_clip_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "class", "mask_idx", "sim_score", "area_frac", "points_count", "crop_x1", "crop_y1", "crop_x2", "crop_y2"])
        for m in all_scored_masks:
            c1, c2, c3, c4 = m["crop_box"]
            writer.writerow([m["image_name"], m["class"], m["mask_idx"], m["sim_score"], m["area_frac"], m["points_count"], c1, c2, c3, c4])

    print(f"\n[SCORES SAVED] -> {json_path} and {csv_path}")
    return class_stats, all_scored_masks


def generate_calibration_crop_grids(all_scored_masks: list[dict]) -> None:
    """STEP 3: Samples 30 mask crops per class spanning low, mid, high scores and creates calibration grids showing ISOLATED TIGHT CROPS."""
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING CALIBRATION CONTACT SHEET GRIDS (TIGHT MASK CROPS)")
    print("=" * 80)

    for cls_name in TARGET_CLASSES:
        cls_masks = [m for m in all_scored_masks if m["class"] == cls_name]
        if not cls_masks:
            continue

        # Sort masks by similarity score ascending
        cls_masks.sort(key=lambda x: x["sim_score"])
        n = len(cls_masks)

        # Sample 30 masks evenly across the full distribution: 10 low (< p25), 10 mid (p25-p75), 10 high (> p75)
        indices = np.linspace(0, n - 1, 30, dtype=int)
        sampled_masks = [cls_masks[i] for i in indices]

        # Generate 5x6 grid image (30 tiles) showing ISOLATED TIGHT CROPS
        cols = 6
        rows = 5
        tile_w, tile_h = 240, 240
        grid_w = cols * tile_w
        grid_h = rows * (tile_h + 45) + 65

        grid_img = Image.new("RGB", (grid_w, grid_h), (15, 23, 42))
        draw_grid = ImageDraw.Draw(grid_img)

        # Header banner
        draw_grid.rectangle([(0, 0), (grid_w, 55)], fill=(30, 41, 59))
        header_title = f"CALIBRATION GRID (TIGHT MASK CROPS) -- {cls_name.upper()}  [Score Range: {sampled_masks[0]['sim_score']:.3f} to {sampled_masks[-1]['sim_score']:.3f}]"
        draw_grid.text((20, 16), header_title, fill=(56, 189, 248))

        for idx, m in enumerate(sampled_masks):
            r = idx // cols
            c = idx % cols
            x = c * tile_w
            y = 60 + (r * (tile_h + 45))

            img_p = Path(m["image_path"])
            try:
                full_img = Image.open(img_p).convert("RGB")
                cx1, cy1, cx2, cy2 = m["crop_box"]
                crop_img = full_img.crop((cx1, cy1, cx2, cy2))

                # Draw local polygon contour on the crop
                draw_crop = ImageDraw.Draw(crop_img)
                flat_pts = m["segmentation"][0]
                local_pts = [(flat_pts[i] - cx1, flat_pts[i+1] - cy1) for i in range(0, len(flat_pts), 2)]
                
                sim = m["sim_score"]
                # Color code score: Green (>=0.65), Yellow (0.50-0.65), Red (<0.50)
                if sim >= 0.65:
                    col = (34, 197, 94)    # Green
                elif sim >= 0.50:
                    col = (234, 179, 8)    # Yellow
                else:
                    col = (239, 68, 68)    # Red

                if len(local_pts) >= 3:
                    draw_crop.polygon(local_pts, outline=col, width=3)

                crop_img.thumbnail((tile_w - 10, tile_h - 10))
                tile = Image.new("RGB", (tile_w, tile_h), (30, 41, 59))
                off_x = (tile_w - crop_img.width) // 2
                off_y = (tile_h - crop_img.height) // 2
                tile.paste(crop_img, (off_x, off_y))
                grid_img.paste(tile, (x, y))

                # Score and filename label below tile
                lbl_text = f"SIM: {sim:.3f} | #{m['mask_idx']} {m['image_name'][:12]}.."
                draw_grid.text((x + 6, y + tile_h + 6), lbl_text, fill=col)
                size_text = f"Crop: {cx2-cx1}x{cy2-cy1} px ({m['area_frac']*100:.1f}%)"
                draw_grid.text((x + 6, y + tile_h + 24), size_text, fill=(148, 163, 184))
            except Exception:
                continue

        out_png = LOGS_DIR / f"calibration_grid_{cls_name}.png"
        grid_img.save(out_png, "PNG")
        print(f"  [{cls_name.upper()}] Calibration crop grid saved -> {out_png}")


def print_threshold_simulation_table(all_scored_masks: list[dict], stats: dict) -> None:
    print("\n" + "=" * 85)
    print("EMPIRICAL THRESHOLD SIMULATION TABLE (PER-MASK TIGHT CROP SCORING)")
    print("=" * 85)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75]

    for cls_name in TARGET_CLASSES:
        cls_masks = [m for m in all_scored_masks if m["class"] == cls_name]
        total_imgs = 320 if cls_name == "crane" else 319
        scores = np.array([m["sim_score"] for m in cls_masks])
        total_m = len(scores)

        print(f"\n[{cls_name.upper()}] Total Candidate Masks: {total_m} across {total_imgs} images")
        print(f"  {'Threshold':<12} {'Surviving Masks':>16} {'Filtered Out (%)':>18} {'Avg Masks/Image':>18}")
        print("  " + "-" * 68)

        for th in thresholds:
            survived = int(np.sum(scores >= th))
            pct_discarded = ((total_m - survived) / total_m) * 100.0
            avg_per_img = survived / total_imgs
            rec_tag = "  <-- RECOMMENDED" if th in [0.55, 0.58, 0.60] and th == 0.58 else ""
            print(f"  >= {th:.2f}            {survived:>12}             {pct_discarded:>6.1f}%            {avg_per_img:>6.2f}{rec_tag}")

    print("\n" + "=" * 85)


def main():
    model, preprocess = load_clip_model()
    ref_banks = build_balanced_reference_banks(model, preprocess)
    stats, all_masks = score_all_candidate_masks(model, preprocess, ref_banks)
    generate_calibration_crop_grids(all_masks)
    print_threshold_simulation_table(all_masks, stats)


if __name__ == "__main__":
    main()
