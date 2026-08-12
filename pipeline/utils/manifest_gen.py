"""
pipeline/utils/manifest_gen.py
===============================
Helper utility — Option C reference-image ingestion.

Usage
-----
Point this at your flat folder of 81 reference images (arbitrary filenames)
and it will generate a blank manifest CSV you fill in:

    python -m pipeline.utils.manifest_gen --src /path/to/your/ref/images

Output: reference_manifest.csv in the repo root with columns:
    filename, class

Rules:
  - Fill in 'class' using the exact slugs printed below the table.
  - Leave 'class' blank for any image you want to exclude.
  - Valid class slugs are printed to the console when you run this script.
  - Re-run stage1_ingest_refs.py once the CSV is filled in.
"""

import argparse
import csv
import shutil
from pathlib import Path

# Import config relative to repo root so this works both as a module
# and via `python -m pipeline.utils.manifest_gen`
from pipeline.config import ROOT, ALL_CLASS_NAMES, CLASSES


MANIFEST_PATH = ROOT / "reference_manifest.csv"


def generate_manifest(src_dir: Path) -> None:
    src_dir = src_dir.resolve()
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    # Collect all image files (common extensions)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    images = sorted(
        p for p in src_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )

    if not images:
        print(f"[WARN] No image files found in {src_dir}")
        return

    # Write blank manifest
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "class"])
        for img in images:
            writer.writerow([img.name, ""])

    print(f"\n✅  Manifest written → {MANIFEST_PATH}")
    print(f"   {len(images)} image(s) listed.\n")
    print("Valid class slugs (copy-paste exactly into the 'class' column):")
    print("-" * 55)
    for c in CLASSES:
        cluster_info = f"  [confusable: {c['cluster']}]" if c["confusable"] else ""
        print(f"  {c['name']:<22} {c['display']}{cluster_info}")
    print("-" * 55)
    print("\nNext steps:")
    print("  1. Open reference_manifest.csv and fill in the 'class' column.")
    print("  2. Leave 'class' blank for any image you want to exclude.")
    print("  3. Run:  python -m pipeline.stage1_ingest_refs")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a blank reference manifest CSV from a flat image folder."
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Path to the flat directory containing your 81 reference images.",
    )
    args = parser.parse_args()
    generate_manifest(args.src)


if __name__ == "__main__":
    main()
