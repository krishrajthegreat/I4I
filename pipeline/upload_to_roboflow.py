# -*- coding: utf-8 -*-
"""
pipeline/upload_to_roboflow.py
===============================
Stage 6 (Cloud) -- Upload cleaned dataset to Roboflow with train/valid/test splits.

Each class folder is uploaded with a 70/20/10 train/valid/test split applied per image.
Reads ROBOFLOW_API_KEY from the environment or a local .env file (never hardcode it).

Usage
-----
1. Create a .env file in the project root:
       ROBOFLOW_API_KEY=your-private-api-key-here

2. Edit the CONFIG block below, then run:
       .venv\\Scripts\\python.exe pipeline/upload_to_roboflow.py
"""

from __future__ import annotations

import os
import sys
import math
import random
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------
PROJECT_ID: str = "conveyer-kniz0"           # Roboflow project slug
DATASET_DIR: Path = Path("dataset/train")    # root with per-class subfolders

# Set to None to upload ALL classes, or specify a subset:
CLASSES_TO_UPLOAD: list[str] | None = None

# Train / Valid / Test split ratios (must sum to 1.0)
SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "valid": 0.20, "test": 0.10}

RANDOM_SEED: int = 42
PROJECT_LICENSE: str = "MIT"
# ---------------------------------------------------------------------------


def assign_splits(images: list[Path], ratios: dict[str, float], seed: int = 42) -> dict[str, list[Path]]:
    """Randomly shuffle images and assign to train/valid/test."""
    imgs = images.copy()
    random.seed(seed)
    random.shuffle(imgs)
    n = len(imgs)
    n_train = math.ceil(n * ratios["train"])
    n_valid = math.ceil(n * ratios["valid"])
    return {
        "train": imgs[:n_train],
        "valid": imgs[n_train:n_train + n_valid],
        "test":  imgs[n_train + n_valid:],
    }


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY not set. Add it to .env or set in PowerShell.")
        sys.exit(1)

    try:
        import roboflow
        from tqdm import tqdm
    except ImportError as e:
        print(f"[ERROR] Missing package: {e}")
        sys.exit(1)

    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    available = sorted([
        d.name for d in DATASET_DIR.iterdir()
        if d.is_dir() and any(f.suffix.lower() in exts for f in d.iterdir())
    ])

    if CLASSES_TO_UPLOAD is not None:
        missing = [c for c in CLASSES_TO_UPLOAD if c not in available]
        if missing:
            print(f"[ERROR] Classes not found in {DATASET_DIR}: {missing}")
            sys.exit(1)
        target_classes = CLASSES_TO_UPLOAD
    else:
        target_classes = available

    # --- Summary table ---
    print("=" * 68)
    print("ROBOFLOW UPLOAD — Dataset Summary")
    print("=" * 68)
    print(f"  Project ID   : {PROJECT_ID}")
    print(f"  Dataset Dir  : {DATASET_DIR.resolve()}")
    print(f"  Split ratios : train={SPLIT_RATIOS['train']:.0%}  "
          f"valid={SPLIT_RATIOS['valid']:.0%}  test={SPLIT_RATIOS['test']:.0%}")
    print(f"  {'Class':<25} {'Total':>7} {'Train':>7} {'Valid':>7} {'Test':>6}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    class_splits: dict[str, dict[str, list[Path]]] = {}
    for cls in target_classes:
        imgs = sorted([f for f in (DATASET_DIR / cls).iterdir()
                       if f.suffix.lower() in exts])
        splits = assign_splits(imgs, SPLIT_RATIOS, RANDOM_SEED)
        class_splits[cls] = splits
        print(f"  {cls:<25} {len(imgs):>7} {len(splits['train']):>7} "
              f"{len(splits['valid']):>7} {len(splits['test']):>6}")
    print("=" * 68)

    confirm = input("\nProceed with upload? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Upload cancelled.")
        sys.exit(0)

    # --- Connect ---
    rf = roboflow.Roboflow(api_key=api_key)
    proj = rf.workspace().project(PROJECT_ID)

    # --- Upload per class with splits ---
    print("\nStarting upload...\n")
    total_ok = total_fail = 0

    for cls in target_classes:
        splits = class_splits[cls]
        ok = fail = 0
        print(f"  [{cls}]")
        for split_name, split_imgs in splits.items():
            if not split_imgs:
                continue
            for img_path in tqdm(split_imgs, desc=f"    {split_name:5}", leave=True):
                try:
                    proj.single_upload(
                        image_path=str(img_path),
                        split=split_name,
                        batch_name=cls,
                        num_retry_uploads=2,
                    )
                    ok += 1
                except Exception as e:
                    fail += 1
                    tqdm.write(f"      [FAIL] {img_path.name}: {e}")
        total_ok += ok
        total_fail += fail
        print(f"    => {ok} uploaded, {fail} failed\n")

    print("=" * 68)
    print(f"Upload complete!  {total_ok} uploaded, {total_fail} failed.")
    print(f"  View: https://app.roboflow.com/{PROJECT_ID}")
    print("=" * 68)


if __name__ == "__main__":
    main()
