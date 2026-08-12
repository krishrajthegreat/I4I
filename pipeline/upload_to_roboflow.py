"""
pipeline/upload_to_roboflow.py
===============================
Stage 6 (Cloud) -- Upload cleaned dataset to Roboflow for labeling and training.

Reads the ROBOFLOW_API_KEY from the environment (or from a local .env file).
NEVER hardcode your API key here — .env is excluded from git via .gitignore.

Usage
-----
1. Create a .env file in the project root (one-time setup):
       ROBOFLOW_API_KEY=your-private-api-key-here
   Or set it directly in PowerShell for the current session:
       $env:ROBOFLOW_API_KEY = "your-private-api-key-here"

2. Run the uploader:
       .venv\\Scripts\\python.exe pipeline/upload_to_roboflow.py

Arguments (edit the CONFIG block below before running)
-------------------------------------------------------
    PROJECT_ID       Your Roboflow project slug (from the project URL on roboflow.com)
    DATASET_DIR      Root folder containing per-class subfolders to upload
    CLASSES_TO_UPLOAD  List of class folder names to upload. Set to None to upload all.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Load .env file if present (python-dotenv is listed in requirements.txt)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed; rely on env vars set in shell

# ---------------------------------------------------------------------------
# ⚙️  CONFIG — edit these before running
# ---------------------------------------------------------------------------
PROJECT_ID: str = "your-project-id"         # e.g. "machine-type-classifier"
DATASET_DIR: Path = Path("dataset/train")   # root folder with per-class subfolders

# Set to None to upload ALL class folders, or list specific ones:
#   CLASSES_TO_UPLOAD = ["lathe", "cnc_milling", "table_saw"]
CLASSES_TO_UPLOAD: list[str] | None = None

PROJECT_LICENSE: str = "MIT"
PROJECT_TYPE: str = "single-label-classification"
# ---------------------------------------------------------------------------


def main() -> None:
    # --- Validate API key ---
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY environment variable is not set.")
        print("  Option 1 — create a .env file in the project root:")
        print("             ROBOFLOW_API_KEY=your-private-api-key-here")
        print("  Option 2 — set it in PowerShell:")
        print('             $env:ROBOFLOW_API_KEY = "your-private-api-key-here"')
        sys.exit(1)

    try:
        import roboflow
    except ImportError:
        print("[ERROR] roboflow package not installed.")
        print("  Run: .venv\\Scripts\\python.exe -m pip install roboflow")
        sys.exit(1)

    # --- Resolve class folders to upload ---
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
            print(f"[ERROR] These classes were listed but not found in {DATASET_DIR}: {missing}")
            sys.exit(1)
        target_classes = CLASSES_TO_UPLOAD
    else:
        target_classes = available

    # --- Print summary before uploading ---
    print("=" * 65)
    print("ROBOFLOW UPLOAD — Dataset Summary")
    print("=" * 65)
    print(f"  Project ID      : {PROJECT_ID}")
    print(f"  Dataset Dir     : {DATASET_DIR.resolve()}")
    print(f"  Classes         : {len(target_classes)}")
    for cls in target_classes:
        imgs = [f for f in (DATASET_DIR / cls).iterdir() if f.suffix.lower() in exts]
        print(f"    {cls:<25} {len(imgs):>5} images")
    print("=" * 65)

    confirm = input("\nProceed with upload? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Upload cancelled.")
        sys.exit(0)

    # --- Connect to Roboflow ---
    rf = roboflow.Roboflow(api_key=api_key)
    workspace = rf.workspace()

    # --- Upload each class folder ---
    print("\nStarting upload...")
    for cls in target_classes:
        cls_dir = DATASET_DIR / cls
        print(f"\n  Uploading class: {cls} ...")
        workspace.upload_dataset(
            str(cls_dir),
            PROJECT_ID,
            project_license=PROJECT_LICENSE,
            project_type=PROJECT_TYPE,
        )
        print(f"  ✓ {cls} done.")

    print("\n" + "=" * 65)
    print("Upload complete!")
    print(f"View your dataset: https://app.roboflow.com/{PROJECT_ID}")
    print("=" * 65)


if __name__ == "__main__":
    main()
