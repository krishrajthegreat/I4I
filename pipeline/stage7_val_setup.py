"""
pipeline/stage7_val_setup.py
=============================
Stage 7 -- Real-World Validation Set Support & CLIP Sanity Check (PRD Section 4).

1. Folder Scaffold & CSV Manifest:
   Ensures dataset/real_validation/<class>/ subfolders exist for all 22 classes.
   Creates/maintains dataset/real_validation/real_validation_manifest.csv
   with schema: filename, class, angle, distance, lighting_notes.

2. Pipeline Sanity Check (PRD Section 4 - Stage 1):
   Scores real validation images through the CLIP filtering pipeline to verify
   that real CCTV/phone images of machines rank correctly against their true class
   before model training.

Usage
-----
    python -m pipeline.stage7_val_setup                  # setup scaffold & run sanity check
    python -m pipeline.stage7_val_setup --check-only      # run sanity check on existing images

Output
------
- dataset/real_validation/real_validation_manifest.csv
- CLIP sanity check scores & confusion report saved to logs/stage7_val_sanity_log.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import (
    ALL_CLASS_NAMES,
    CLASSES,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DATASET_VAL_DIR,
    LOGS_DIR,
    REFERENCE_IMAGES_DIR,
    class_by_name,
)

MANIFEST_PATH = DATASET_VAL_DIR / "real_validation_manifest.csv"


def setup_scaffold() -> None:
    DATASET_VAL_DIR.mkdir(parents=True, exist_ok=True)
    for c in ALL_CLASS_NAMES:
        (DATASET_VAL_DIR / c).mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "class", "angle", "distance", "lighting_notes"])
        print(f"Created real-validation manifest scaffold -> {MANIFEST_PATH}")
    else:
        print(f"Real-validation manifest scaffold exists -> {MANIFEST_PATH}")


def run_clip_sanity_check() -> dict:
    try:
        import open_clip
    except ImportError:
        print("[ERROR] open_clip not installed. Run: pip install open-clip-torch")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading OpenCLIP ({CLIP_MODEL_NAME}) for Stage 7 Sanity Check...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model.eval()

    # Pre-compute text embeddings for all 22 classes
    class_text_embs = []
    for cls in CLASSES:
        prompt = f"a photo of a {cls['display']}"
        tokens = tokenizer([prompt]).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb /= emb.norm(dim=-1, keepdim=True)
        class_text_embs.append(emb)

    text_matrix = torch.cat(class_text_embs, dim=0)  # (22, 512)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    results = []
    total_val_imgs = 0
    correct_top1 = 0

    for idx, cls in enumerate(CLASSES):
        cls_name = cls["name"]
        val_dir = DATASET_VAL_DIR / cls_name
        files = sorted([p for p in val_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]) if val_dir.exists() else []

        if not files:
            results.append({
                "class": cls_name,
                "image_count": 0,
                "top1_correct": 0,
                "accuracy": None,
            })
            continue

        cls_correct = 0
        for p in files:
            total_val_imgs += 1
            try:
                img = Image.open(p).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    img_emb = model.encode_image(tensor)
                    img_emb /= img_emb.norm(dim=-1, keepdim=True)

                sims = (img_emb @ text_matrix.T).squeeze(0).cpu().numpy()
                pred_idx = int(np.argmax(sims))
                pred_class = CLASSES[pred_idx]["name"]

                if pred_class == cls_name:
                    cls_correct += 1
                    correct_top1 += 1
            except Exception as exc:
                print(f"  [WARN] Error processing {p.name}: {exc}")

        acc = (cls_correct / len(files)) if files else 0.0
        results.append({
            "class": cls_name,
            "image_count": len(files),
            "top1_correct": cls_correct,
            "accuracy": acc,
        })

    overall_acc = (correct_top1 / total_val_imgs) if total_val_imgs > 0 else None

    log_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_val_images": total_val_imgs,
        "correct_top1": correct_top1,
        "overall_accuracy": overall_acc,
        "class_breakdown": results,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "stage7_val_sanity_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    print("\n" + "=" * 65)
    print("STAGE 7 -- REAL VALIDATION SANITY CHECK REPORT")
    print("=" * 65)
    print(f"  {'Class':<22} {'Images':>6} {'Top1 Correct':>13} {'Accuracy':>10}")
    print("-" * 65)
    for r in results:
        acc_str = f"{r['accuracy']*100:.1f}%" if r['accuracy'] is not None else "N/A (0 imgs)"
        print(f"  {r['class']:<22} {r['image_count']:>6} {r['top1_correct']:>13} {acc_str:>10}")
    print("-" * 65)
    if overall_acc is not None:
        print(f"  OVERALL ACCURACY: {overall_acc*100:.1f}% ({correct_top1}/{total_val_imgs})")
    else:
        print("  NO REAL VALIDATION IMAGES FOUND YET (Populate dataset/real_validation/<class>/)")
    print("=" * 65)
    print(f"\nLog saved -> {log_path}")

    return log_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7: Real-World Validation Setup & CLIP Sanity Check.")
    parser.add_argument("--check-only", action="store_true", help="Run sanity check without recreating scaffold")
    args = parser.parse_args()

    if not args.check_only:
        setup_scaffold()

    run_clip_sanity_check()


if __name__ == "__main__":
    main()
