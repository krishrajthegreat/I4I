# -*- coding: utf-8 -*-
"""
pipeline/sweep_mislabeled_images.py
====================================
Coarse Content & Cross-Class Verification Sweep for Source Images.

Features:
- STEP 1: Evaluates whole source images in dataset/train/crane and dataset/train/fire_extinguisher.
- Compares whole-image embeddings against:
  1. Own class reference bank.
  2. Alternative industrial machine reference banks (cnc_milling, lathe, milling, etc.) and diagram anchors.
- Identifies images where:
  a) Max mask crop score in the image < 0.50.
  b) Whole image matches an alternative class significantly higher than its assigned class.
  c) Whole image similarity to class is below coarse threshold.
- Generates visual contact sheet (PNG + HTML) of flagged candidate images for manual review.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw, ImageFont

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
from pipeline.clip_mask_verification import BALANCED_REFERENCE_SEEDS, load_clip_model

TARGET_CLASSES = ["crane", "fire_extinguisher"]


def build_cross_class_anchors(model, preprocess) -> dict[str, torch.Tensor]:
    """Builds reference vectors for other common machines to detect cross-contamination."""
    other_classes = ["cnc_milling", "lathe", "milling", "table_saw", "conveyor", "control_panel"]
    other_banks = {}

    for c in other_classes:
        c_dir = DATASET_TRAIN_DIR / c
        if c_dir.exists():
            files = sorted(list(c_dir.glob("*.jpg")))[:12]
            tensors = [preprocess(Image.open(f).convert("RGB")) for f in files if f.exists()]
            if tensors:
                with torch.no_grad():
                    b = torch.stack(tensors)
                    embs = model.encode_image(b)
                    embs /= embs.norm(dim=-1, keepdim=True)
                    other_banks[c] = embs

    # Text anchors for diagrams/infographics/posters
    diagram_queries = [
        "technical spec sheet diagram illustration chart",
        "safety poster infographic vector graphic with text and icons",
        "instruction manual diagram schematic drawing"
    ]
    with torch.no_grad():
        toks = open_clip.tokenize(diagram_queries)
        d_embs = model.encode_text(toks)
        d_embs /= d_embs.norm(dim=-1, keepdim=True)
        other_banks["infographic_diagram"] = d_embs

    return other_banks


def sweep_source_images() -> tuple[list[dict], dict]:
    model, preprocess = load_clip_model()

    # Load existing mask crop scores
    mask_scores_file = LOGS_DIR / "mask_clip_scores.json"
    image_max_mask_scores = {}
    if mask_scores_file.exists():
        with open(mask_scores_file, "r", encoding="utf-8") as f:
            d = json.load(f)
        for m in d.get("masks", []):
            img_name = m["image_name"]
            score = m["sim_score"]
            if img_name not in image_max_mask_scores or score > image_max_mask_scores[img_name]:
                image_max_mask_scores[img_name] = score

    # Build reference banks
    ref_banks = {}
    for cls_name in TARGET_CLASSES:
        cls_dir = DATASET_TRAIN_DIR / cls_name
        seed_names = BALANCED_REFERENCE_SEEDS.get(cls_name, [])
        ref_paths = [cls_dir / fn for fn in seed_names if (cls_dir / fn).exists()]
        tensors = [preprocess(Image.open(p).convert("RGB")) for p in ref_paths]
        with torch.no_grad():
            b = torch.stack(tensors)
            embs = model.encode_image(b)
            embs /= embs.norm(dim=-1, keepdim=True)
            ref_banks[cls_name] = embs

    other_banks = build_cross_class_anchors(model, preprocess)

    flagged_candidates = []
    class_stats = {}

    print("\n" + "=" * 80)
    print("STEP 1: SWEEPING SOURCE IMAGES FOR MISLABELED CONTENT")
    print("=" * 80)

    for cls_name in TARGET_CLASSES:
        cls_dir = DATASET_TRAIN_DIR / cls_name
        files = sorted([p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        own_ref = ref_banks[cls_name]

        print(f"\nAnalyzing {len(files)} whole source images in [{cls_name.upper()}]...")
        cls_flagged = []

        for p in files:
            try:
                img = Image.open(p).convert("RGB")
                t = preprocess(img).unsqueeze(0)
            except Exception:
                continue

            with torch.no_grad():
                img_emb = model.encode_image(t)
                img_emb /= img_emb.norm(dim=-1, keepdim=True)

                # Whole-image similarity to own class bank
                own_sim = float((img_emb @ own_ref.T).max())

                # Check against other machine types & diagrams
                best_other_class = None
                best_other_sim = 0.0

                for other_c, other_mat in other_banks.items():
                    sim = float((img_emb @ other_mat.T).max())
                    if sim > best_other_sim:
                        best_other_sim = sim
                        best_other_class = other_c

            max_mask_sim = image_max_mask_scores.get(p.name, 0.0)

            # Flagging criteria:
            # 1. No mask in the image even reaches 0.50 similarity
            # 2. Or matches another class (e.g. CNC milling / diagram) higher than own class
            # 3. Or whole image similarity < 0.44
            is_flagged = False
            flag_reason = []

            if max_mask_sim < 0.50:
                is_flagged = True
                flag_reason.append(f"No mask >= 0.50 (best mask sim = {max_mask_sim:.3f})")

            if best_other_sim > own_sim + 0.03:
                is_flagged = True
                flag_reason.append(f"Closer to '{best_other_class}' ({best_other_sim:.3f}) than '{cls_name}' ({own_sim:.3f})")

            if own_sim < 0.44 and max_mask_sim < 0.55:
                is_flagged = True
                flag_reason.append(f"Low whole-image similarity ({own_sim:.3f})")

            if is_flagged:
                item = {
                    "class": cls_name,
                    "filename": p.name,
                    "image_path": str(p),
                    "own_class_sim": round(own_sim, 3),
                    "max_mask_sim": round(max_mask_sim, 3),
                    "best_alt_class": best_other_class,
                    "best_alt_sim": round(best_other_sim, 3),
                    "reasons": flag_reason,
                }
                cls_flagged.append(item)
                flagged_candidates.append(item)

        class_stats[cls_name] = {
            "total_images": len(files),
            "flagged_count": len(cls_flagged),
            "clean_count": len(files) - len(cls_flagged),
        }
        print(f"  [{cls_name.upper()}] Flagged {len(cls_flagged)} candidate mislabeled images out of {len(files)}.")

    return flagged_candidates, class_stats


def generate_flagged_contact_sheet(flagged_candidates: list[dict]) -> None:
    """Generates visual PNG and HTML contact sheets of all flagged candidate images."""
    if not flagged_candidates:
        print("[INFO] No candidate mislabeled images were flagged.")
        return

    out_png = LOGS_DIR / "flagged_mislabeled_candidates.png"
    out_html = LOGS_DIR / "flagged_mislabeled_candidates.html"

    # Composite PNG Grid
    n = len(flagged_candidates)
    cols = 4
    rows = math.ceil(n / cols)
    tile_w, tile_h = 320, 260
    header_h = 60
    footer_h = 60

    grid_w = cols * tile_w
    grid_h = header_h + rows * (tile_h + footer_h)

    grid_img = Image.new("RGB", (grid_w, grid_h), (15, 23, 42))
    draw_grid = ImageDraw.Draw(grid_img)

    # Banner
    draw_grid.rectangle([(0, 0), (grid_w, header_h)], fill=(30, 41, 59))
    draw_grid.text((20, 18), f"FLAGGED MISLABELED CANDIDATES AUDIT ({n} Total Images Flagged for Review)", fill=(239, 68, 68))

    for idx, item in enumerate(flagged_candidates):
        r = idx // cols
        c = idx % cols
        x = c * tile_w
        y = header_h + r * (tile_h + footer_h)

        img_p = Path(item["image_path"])
        try:
            im = Image.open(img_p).convert("RGB")
            im.thumbnail((tile_w - 16, tile_h - 16))
            tile = Image.new("RGB", (tile_w - 16, tile_h - 16), (30, 41, 59))
            off_x = ((tile_w - 16) - im.width) // 2
            off_y = ((tile_h - 16) - im.height) // 2
            tile.paste(im, (off_x, off_y))
            grid_img.paste(tile, (x + 8, y + 8))

            # Labels
            fn_lbl = f"{item['class'].upper()}: {item['filename'][:20]}.."
            draw_grid.text((x + 8, y + tile_h - 4), fn_lbl, fill=(248, 250, 252))
            
            sim_lbl = f"Own: {item['own_class_sim']:.2f} | Mask: {item['max_mask_sim']:.2f} | Alt: {item['best_alt_class']} ({item['best_alt_sim']:.2f})"
            draw_grid.text((x + 8, y + tile_h + 12), sim_lbl, fill=(244, 63, 94))

            r_lbl = f"Reason: {item['reasons'][0][:38]}"
            draw_grid.text((x + 8, y + tile_h + 28), r_lbl, fill=(148, 163, 184))
        except Exception:
            continue

    out_png.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(out_png, "PNG")
    print(f"\n[CONTACT SHEET SAVED] -> {out_png}")

    # Interactive HTML Contact Sheet
    cards_html = []
    for item in flagged_candidates:
        p = Path(item["image_path"])
        uri = p.resolve().as_uri()
        reasons_str = "<br>".join(item["reasons"])
        card = f"""
        <div class="card">
            <a href="{uri}" target="_blank"><img src="{uri}" alt="{item['filename']}" loading="lazy"/></a>
            <div class="info">
                <span class="fname">{item['filename']}</span>
                <span class="badge {item['class']}">{item['class'].upper()}</span>
                <div class="scores">
                    <div><strong>Own Class Sim:</strong> {item['own_class_sim']:.3f}</div>
                    <div><strong>Best Mask Sim:</strong> {item['max_mask_sim']:.3f}</div>
                    <div><strong>Best Match:</strong> <span class="alt">{item['best_alt_class']}</span> ({item['best_alt_sim']:.3f})</div>
                </div>
                <div class="reason"><strong>Flag Reason:</strong><br>{reasons_str}</div>
            </div>
        </div>
        """
        cards_html.append(card)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Flagged Mislabeled Candidate Images</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b1120; color: #f8fafc; margin: 0; padding: 24px; }}
        h1 {{ color: #ef4444; margin: 0 0 8px 0; }}
        p.subtitle {{ color: #94a3b8; margin: 0 0 24px 0; font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px; }}
        .card {{ background: #0f172a; border: 1px solid #334155; border-radius: 10px; overflow: hidden; display: flex; flex-direction: column; }}
        .card img {{ width: 100%; height: 220px; object-fit: cover; display: block; background: #1e293b; }}
        .info {{ padding: 14px; font-size: 12px; display: flex; flex-direction: column; gap: 8px; }}
        .fname {{ font-weight: 700; color: #e2e8f0; word-break: break-all; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; width: fit-content; }}
        .badge.crane {{ background: #0284c7; color: white; }}
        .badge.fire_extinguisher {{ background: #dc2626; color: white; }}
        .scores {{ background: #1e293b; padding: 8px; border-radius: 6px; }}
        .alt {{ color: #f59e0b; font-weight: 600; }}
        .reason {{ color: #fca5a5; font-size: 11px; background: #450a0a; padding: 6px; border-radius: 4px; border: 1px solid #7f1d1d; }}
    </style>
</head>
<body>
    <h1>Flagged Mislabeled Candidate Images Review</h1>
    <p class="subtitle">Total Flagged: {len(flagged_candidates)} images across crane and fire_extinguisher. Review each image before removal.</p>
    <div class="grid">
        {"".join(cards_html)}
    </div>
</body>
</html>
"""
    out_html.write_text(html_doc, encoding="utf-8")
    print(f"[HTML GALLERY SAVED] -> {out_html}")


def main():
    flagged, stats = sweep_source_images()
    generate_flagged_contact_sheet(flagged)

    # Save to JSON
    report_path = LOGS_DIR / "flagged_mislabeled_images.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "stats": stats, "flagged": flagged}, f, indent=2)

    print("\n" + "=" * 80)
    print("MISLABELED SWEEP SUMMARY")
    print("=" * 80)
    for c, s in stats.items():
        print(f"  {c:<20} | Total Images: {s['total_images']} | Flagged Mislabeled: {s['flagged_count']} | Clean: {s['clean_count']}")
    print(f"\n[REPORT SAVED] -> {report_path}\n")


if __name__ == "__main__":
    main()
