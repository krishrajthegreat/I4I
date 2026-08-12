"""
pipeline/stage4b_calibrate.py
==============================
CLIP Threshold Calibration Utility (PRD Section 3.2).

Raw CLIP similarity scores (text-to-image or image-to-image) depend heavily on
the specific prompt and model architecture.  This script scores a small sample
of labeled images (good vs. bad examples) to empirically derive working similarity
thresholds rather than guessing arbitrary values like 0.8.

Usage
-----
    python -m pipeline.stage4b_calibrate --class lathe \
        --good dataset/train/lathe \
        --bad path/to/bad_samples

Arguments
---------
--class    Class name (e.g. lathe, table_saw)
--good     Directory of verified GOOD images for this class
--bad      Directory of verified BAD/irrelevant images
--out      Output calibration plot image path (default: logs/calibration_<class>.png)

Output
------
- Prints percentile statistics (min, max, mean, std) for good vs bad sets
- Suggests optimal threshold separating the two distributions
- Saves a visual histogram plot to logs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import (
    CLASSES,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    LOGS_DIR,
    REFERENCE_IMAGES_DIR,
    class_by_name,
)

try:
    import open_clip
except ImportError:
    print("[ERROR] open_clip not installed. Run: pip install open-clip-torch")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_clip_model():
    device = "cpu"
    if torch.cuda.is_available():
        try:
            _dummy = torch.zeros(1, device="cuda") + 1
            device = "cuda"
        except Exception:
            device = "cpu"
    print(f"Loading OpenCLIP ({CLIP_MODEL_NAME} / {CLIP_PRETRAINED}) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model.eval()
    return model, preprocess, tokenizer, device


def get_image_embeddings(image_paths: list[Path], model, preprocess, device) -> torch.Tensor:
    embeddings = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(tensor)
                emb /= emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb)
        except Exception as exc:
            print(f"  [WARN] Failed to process {path.name}: {exc}")
    if not embeddings:
        return torch.empty((0, 512), device=device)
    return torch.cat(embeddings, dim=0)


def calibrate(class_name: str, good_dir: Path, bad_dir: Path | None = None) -> None:
    cls = class_by_name(class_name)
    display_name = cls["display"]
    is_confusable = cls["confusable"]

    model, preprocess, tokenizer, device = load_clip_model()

    # Text embedding for coarse filter
    text_prompt = f"a photo of a {display_name}"
    text_tokens = tokenizer([text_prompt]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)

    # Reference embeddings for fine filter if confusable
    ref_emb = None
    ref_dir = REFERENCE_IMAGES_DIR / class_name
    if is_confusable and ref_dir.exists():
        ref_files = [p for p in ref_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if ref_files:
            print(f"Loading {len(ref_files)} reference image(s) for {class_name}...")
            ref_emb = get_image_embeddings(ref_files, model, preprocess, device)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    good_files = sorted([p for p in good_dir.iterdir() if p.suffix.lower() in exts]) if good_dir.exists() else []
    bad_files = sorted([p for p in bad_dir.iterdir() if p.suffix.lower() in exts]) if (bad_dir and bad_dir.exists()) else []

    print(f"\nCalibrating scores for class: {class_name}")
    print(f"  Good samples: {len(good_files)}")
    print(f"  Bad samples : {len(bad_files)}")

    if not good_files:
        print("[ERROR] No good sample images found.")
        return

    # Compute good scores
    good_img_embs = get_image_embeddings(good_files, model, preprocess, device)
    good_text_scores = (good_img_embs @ text_emb.T).squeeze(-1).cpu().numpy()

    good_fine_scores = None
    if ref_emb is not None and len(ref_emb) > 0:
        sim_matrix = good_img_embs @ ref_emb.T  # (N, M)
        good_fine_scores = sim_matrix.mean(dim=-1).cpu().numpy()

    print("\n--- Coarse (Text-to-Image) Score Statistics ---")
    print(f"  Good Mean: {np.mean(good_text_scores):.4f} | Std: {np.std(good_text_scores):.4f}")
    print(f"  Good Min : {np.min(good_text_scores):.4f} | Max: {np.max(good_text_scores):.4f}")
    print(f"  Good 1st percentile: {np.percentile(good_text_scores, 1):.4f}")
    print(f"  Good 5th percentile: {np.percentile(good_text_scores, 5):.4f}")

    rec_coarse = float(np.percentile(good_text_scores, 1))

    if bad_files:
        bad_img_embs = get_image_embeddings(bad_files, model, preprocess, device)
        bad_text_scores = (bad_img_embs @ text_emb.T).squeeze(-1).cpu().numpy()
        print(f"  Bad  Mean: {np.mean(bad_text_scores):.4f} | Std: {np.std(bad_text_scores):.4f}")
        print(f"  Bad  Max : {np.max(bad_text_scores):.4f}")
        # NOTE: if bad_max >= good_1st_pct, good/bad distributions overlap on the coarse score.
        # In that case, discard the midpoint and use a fixed loose floor (e.g. 0.20) instead,
        # relying on the fine image-to-image filter for real discrimination.
        good_1st = np.percentile(good_text_scores, 1)
        bad_mx = np.max(bad_text_scores)
        if bad_mx >= good_1st:
            print(f"  [OVERLAP] bad_max ({bad_mx:.4f}) >= good_1st_pct ({good_1st:.4f}) — coarse filter CANNOT separate.")
            print(f"  [OVERLAP] Recommended action: set coarse to fixed loose floor (e.g. 0.20) and rely on fine filter.")
        else:
            # Midpoint between bad_max and good 1st percentile (more lenient than old good-5th midpoint)
            midpoint = (bad_mx + good_1st) / 2
            rec_coarse = float(midpoint)

    print(f"\n>>> Recommended Coarse Text Threshold: {rec_coarse:.4f}")

    if good_fine_scores is not None:
        print("\n--- Fine (Image-to-Image) Score Statistics ---")
        print(f"  Good Mean: {np.mean(good_fine_scores):.4f} | Std: {np.std(good_fine_scores):.4f}")
        print(f"  Good Min : {np.min(good_fine_scores):.4f} | Max: {np.max(good_fine_scores):.4f}")
        print(f"  Good 1st percentile: {np.percentile(good_fine_scores, 1):.4f}")
        print(f"  Good 5th percentile: {np.percentile(good_fine_scores, 5):.4f}")
        rec_fine = float(np.percentile(good_fine_scores, 1))
        print(f"\n>>> Recommended Fine Reference Threshold: {rec_fine:.4f}")

    # Plot if matplotlib available
    if HAS_MATPLOTLIB:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = LOGS_DIR / f"calibration_{class_name}.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(good_text_scores, bins=20, alpha=0.6, label="Good examples", color="green")
        if bad_files:
            ax.hist(bad_text_scores, bins=20, alpha=0.6, label="Bad examples", color="red")
        ax.axvline(rec_coarse, color="black", linestyle="--", label=f"Threshold ({rec_coarse:.4f})")
        ax.set_title(f"CLIP Coarse Threshold Calibration -- {class_name}")
        ax.set_xlabel("Text-Image Cosine Similarity Score")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"\nCalibration plot saved -> {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4b: CLIP Threshold Calibration.")
    parser.add_argument("--class", dest="class_name", required=True, help="Class slug (e.g. lathe)")
    parser.add_argument("--good", type=Path, required=True, help="Directory of GOOD sample images")
    parser.add_argument("--bad", type=Path, default=None, help="Directory of BAD sample images (optional)")
    args = parser.parse_args()
    calibrate(args.class_name, args.good, args.bad)


if __name__ == "__main__":
    main()
