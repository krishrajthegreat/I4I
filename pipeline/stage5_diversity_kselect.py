"""
pipeline/stage5_diversity_kselect.py
=====================================
Stage 5 -- Diversity Deduplication & K-Selection (PRD Section 3.3 & 3.4).

1. Embedding Diversity Filtering:
   Greedy max-coverage filter. Iteratively selects candidate images whose CLIP
   embedding distance to all already-selected images is above the threshold,
   preventing multiple crops/re-uploads of stock photos.

2. K-Selection & Stopping Rule (PRD Section 3.4):
   - Minimum required: 80 images (standard classes) / 150 images (confusable classes)
   - Selects top-K highest quality / diverse images up to cap.
   - Flags (does not drop) any class that falls below the required minimum.

Usage
-----
    python -m pipeline.stage5_diversity_kselect
    python -m pipeline.stage5_diversity_kselect --classes lathe table_saw

Output
------
- Retains top-K diverse images in dataset/train/<class>/ and removes excess
- Logs target vs. actual counts and flags under-represented classes in logs/stage5_kselect_log.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import (
    ALL_CLASS_NAMES,
    CLASSES,
    CLIP_BATCH_SIZE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DATASET_TRAIN_DIR,
    DIVERSITY_SIM_THRESHOLD,
    K_MIN_CONFUSABLE,
    K_MIN_FLOOR,
    K_MIN_STANDARD,
    LOGS_DIR,
    class_by_name,
)

try:
    import open_clip
except ImportError:
    print("[ERROR] open_clip not installed. Run: pip install open-clip-torch")
    sys.exit(1)


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


def get_image_embeddings(
    image_paths: list[Path], model, preprocess, device, batch_size: int = CLIP_BATCH_SIZE
) -> tuple[list[Path], torch.Tensor]:
    valid_paths = []
    tensors = []
    all_embs = []

    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            t = preprocess(img)
            tensors.append(t)
            valid_paths.append(p)
        except Exception:
            p.unlink(missing_ok=True)

    if not tensors:
        return [], torch.empty((0, 512), device=device)

    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        with torch.no_grad():
            embs = model.encode_image(batch)
            embs /= embs.norm(dim=-1, keepdim=True)
        all_embs.append(embs)

    return valid_paths, torch.cat(all_embs, dim=0)


def process_class(
    cls_name: str, model, preprocess, tokenizer, device, sim_thresh: float = DIVERSITY_SIM_THRESHOLD
) -> dict:
    cls = class_by_name(cls_name)
    is_confusable = cls["confusable"]
    # Hard floor: 150 for confusable-cluster classes, 80 for standard classes.
    hard_floor = K_MIN_CONFUSABLE if is_confusable else K_MIN_STANDARD
    target_k = hard_floor  # target and hard floor are the same
    class_dir = DATASET_TRAIN_DIR / cls_name

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]) if class_dir.exists() else []

    if not files:
        return {
            "class": cls_name,
            "confusable": is_confusable,
            "hard_floor": hard_floor,
            "target_k": target_k,
            "input_count": 0,
            "selected_count": 0,
            "removed_diversity": 0,
            "meets_floor": False,
            "meets_min": False,
            "meets_target": False,
            "below_target": not is_confusable,
            "flagged": True,
        }

    paths, embs = get_image_embeddings(files, model, preprocess, device)
    if len(paths) == 0:
        return {
            "class": cls_name,
            "confusable": is_confusable,
            "hard_floor": hard_floor,
            "target_k": target_k,
            "input_count": len(files),
            "selected_count": 0,
            "removed_diversity": len(files),
            "meets_floor": False,
            "meets_min": False,
            "meets_target": False,
            "below_target": not is_confusable,
            "flagged": True,
        }

    # Text prompt ranking for initial quality ordering
    prompt = f"a photo of a {cls['display']}"
    tokens = tokenizer([prompt]).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(tokens)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)

    text_scores = (embs @ text_emb.T).squeeze(-1).cpu().numpy()
    rank_order = np.argsort(-text_scores)  # sort descending by quality score

    selected_indices: list[int] = []
    selected_embs: list[torch.Tensor] = []

    for idx in rank_order:
        cand_emb = embs[idx : idx + 1]
        is_too_similar = False

        if selected_embs:
            sel_matrix = torch.cat(selected_embs, dim=0)
            sims = (cand_emb @ sel_matrix.T).squeeze(0).cpu().numpy()
            if np.max(sims) > sim_thresh:
                is_too_similar = True

        if not is_too_similar:
            selected_indices.append(idx)
            selected_embs.append(cand_emb)

    selected_paths = {paths[i] for i in selected_indices}
    all_set = set(paths)
    removed_paths = all_set - selected_paths

    for p in removed_paths:
        p.unlink(missing_ok=True)

    selected_count = len(selected_paths)
    meets_floor = selected_count >= hard_floor
    meets_target = meets_floor  # floor == target now (no split)
    flagged = not meets_floor  # FLAGGED if under the class-specific hard floor

    return {
        "class": cls_name,
        "confusable": is_confusable,
        "hard_floor": hard_floor,
        "target_k": target_k,
        "input_count": len(files),
        "selected_count": selected_count,
        "removed_diversity": len(removed_paths),
        "meets_floor": meets_floor,
        "meets_min": meets_floor,  # backwards-compatibility alias
        "meets_target": meets_target,
        "below_target": is_confusable and not meets_target,
        "flagged": flagged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: Diversity Dedup & K-Selection.")
    parser.add_argument("--classes", nargs="+", default=None, help="Process specific class slugs")
    parser.add_argument("--sim-thresh", type=float, default=DIVERSITY_SIM_THRESHOLD, help="Embedding cosine similarity threshold for dedup")
    args = parser.parse_args()

    if args.classes:
        target_names = [c for c in args.classes if c in ALL_CLASS_NAMES]
    else:
        target_names = ALL_CLASS_NAMES

    model, preprocess, tokenizer, device = load_clip_model()

    print("=" * 75)
    print("STAGE 5 -- Diversity Dedup & K-Selection")
    print(f"Classes                  : {len(target_names)}")
    print(f"Hard Floor (standard)    : {K_MIN_STANDARD}")
    print(f"Hard Floor (confusable)  : {K_MIN_CONFUSABLE}")
    print(f"Diversity Sim Thresh     : {args.sim_thresh}")
    print("=" * 75)

    results = []
    flagged_classes = []

    for cls_name in tqdm(target_names, desc="K-Selection", unit="class"):
        res = process_class(cls_name, model, preprocess, tokenizer, device, args.sim_thresh)
        results.append(res)
        if res["flagged"]:
            flagged_classes.append(res)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "stage5_kselect_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sim_thresh": args.sim_thresh,
                "hard_floor": K_MIN_FLOOR,
                "results": results,
                "flagged_classes": [r["class"] for r in flagged_classes],
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 75)
    print("K-SELECTION SUMMARY")
    print("=" * 75)
    print(f"  {'Class':<22} {'Input':>6} {'Floor':>6} {'TargetK':>8} {'Selected':>9} {'Status':<25}")
    print("-" * 75)
    tot_in = tot_sel = 0
    for r in results:
        if not r["meets_floor"]:
            floor_val = K_MIN_CONFUSABLE if r["confusable"] else K_MIN_STANDARD
            status_str = f"FLAGGED (<{floor_val})"
        else:
            status_str = "PASS"
        print(
            f"  {r['class']:<22} {r['input_count']:>6} {r['hard_floor']:>6} {r['target_k']:>8} "
            f"{r['selected_count']:>9} {status_str:<20}"
        )
        tot_in += r["input_count"]
        tot_sel += r["selected_count"]
    print("-" * 75)
    print(f"  {'TOTAL':<22} {tot_in:>6} {'':>6} {'':>8} {tot_sel:>9}")
    print("=" * 75)

    if flagged_classes:
        print("\nWARNING: The following classes failed to meet the hard minimum floor:")
        for fc in flagged_classes:
            floor_val = K_MIN_CONFUSABLE if fc["confusable"] else K_MIN_STANDARD
            print(f"  * {fc['class']} -> {fc['selected_count']}/{floor_val} (confusable: {fc['confusable']})")
        print("  Per PRD Section 3.4: flagged for alternate sourcing or scrape top-up.")
    else:
        print("\nSUCCESS: All classes cleared their hard minimum floors!")

    print(f"\nLog saved -> {log_path}")


if __name__ == "__main__":
    main()
