"""
pipeline/stage1_ingest_refs.py
================================
Stage 1b — Ingest hand-picked reference images from a manifest CSV into
reference_images/<class>/ and print a coverage report.

Prerequisites
-------------
1. Run manifest_gen.py to produce reference_manifest.csv.
2. Fill in the 'class' column with valid slugs from pipeline/config.py.
3. Run this script.

Usage
-----
    python -m pipeline.stage1_ingest_refs [--manifest PATH] [--src PATH] [--dry-run]

Arguments
---------
--manifest  Path to the filled-in CSV.  Default: <repo root>/reference_manifest.csv
--src       Directory where the original images live.
            If omitted the script tries to infer it from the manifest path.
--dry-run   Print what would happen but do NOT copy any files.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import (
    CLASSES,
    LOGS_DIR,
    MIN_REFS_PER_CONFUSABLE_CLASS,
    REFERENCE_IMAGES_DIR,
    ROOT,
    ALL_CLASS_NAMES,
    confusable_classes,
)

MANIFEST_DEFAULT = ROOT / "reference_manifest.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_manifest(manifest_path: Path) -> list[dict]:
    """Read CSV and return list of dicts with 'filename' and 'class'."""
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        print("        Run:  python -m pipeline.utils.manifest_gen --src <your_ref_folder>")
        sys.exit(1)

    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=2):   # start=2 → skip header row 1
            fn = row.get("filename", "").strip()
            cls = row.get("class", "").strip()
            if not fn:
                continue
            if not cls:
                continue  # user intentionally excluded this image
            rows.append({"filename": fn, "class": cls, "line": i})
    return rows


def _validate_class_slugs(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split rows into valid and invalid class slugs."""
    valid, invalid = [], []
    for row in rows:
        if row["class"] in ALL_CLASS_NAMES:
            valid.append(row)
        else:
            invalid.append(row)
    return valid, invalid


def _infer_src_dir(manifest_path: Path) -> Path:
    """Guess source dir = same directory as the manifest."""
    return manifest_path.parent


# ──────────────────────────────────────────────────────────────────────────────
# Main ingestion logic
# ──────────────────────────────────────────────────────────────────────────────

def ingest(
    manifest_path: Path,
    src_dir: Path | None,
    dry_run: bool = False,
) -> None:
    rows = _load_manifest(manifest_path)
    if src_dir is None:
        src_dir = _infer_src_dir(manifest_path)

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Ingesting reference images")
    print(f"  Manifest : {manifest_path}")
    print(f"  Source   : {src_dir}")
    print(f"  Target   : {REFERENCE_IMAGES_DIR}\n")

    # Validate slugs
    valid_rows, bad_rows = _validate_class_slugs(rows)
    if bad_rows:
        print("[WARN] The following rows have unrecognised class slugs and will be SKIPPED:")
        for r in bad_rows:
            print(f"       Line {r['line']:>3}: {r['filename']!r:40s} → {r['class']!r}")
        print()

    # Group by class
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in valid_rows:
        by_class[row["class"]].append(row)

    # Copy files
    copied: dict[str, list[str]] = defaultdict(list)
    missing_files: list[str] = []

    for row in valid_rows:
        src_file = src_dir / row["filename"]
        if not src_file.exists():
            missing_files.append(str(src_file))
            continue

        dst_dir = REFERENCE_IMAGES_DIR / row["class"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir / row["filename"]

        if not dry_run:
            shutil.copy2(src_file, dst_file)
        copied[row["class"]].append(row["filename"])

    if missing_files:
        print("[WARN] Files listed in manifest but NOT found in source directory:")
        for f in missing_files:
            print(f"       {f}")
        print()

    # ── Coverage report ─────────────────────────────────────────────────────
    confusable_names = {c["name"] for c in confusable_classes()}

    print("=" * 65)
    print("REFERENCE IMAGE COVERAGE REPORT")
    print("=" * 65)
    print(f"{'Class':<25} {'Confusable':<12} {'Refs copied':>11}  Status")
    print("-" * 65)

    log_data: list[dict] = []
    gap_classes: list[str] = []  # confusable classes with < MIN_REFS

    for cls in CLASSES:
        name = cls["name"]
        is_confusable = cls["confusable"]
        count = len(copied.get(name, []))

        if is_confusable:
            if count == 0:
                status = "MISSING — fill before Stage 4"
                gap_classes.append(name)
            elif count < MIN_REFS_PER_CONFUSABLE_CLASS:
                status = f"LOW ({count}/{MIN_REFS_PER_CONFUSABLE_CLASS} min recommended)"
                gap_classes.append(name)
            else:
                status = "OK"
        else:
            # Non-confusable classes don't need reference images
            status = "-  (fine-filter not applied)"

        print(f"  {name:<23} {'yes' if is_confusable else 'no':<12} {count:>11}  {status}")
        log_data.append({
            "class": name,
            "confusable": is_confusable,
            "refs_copied": count,
            "status": status,
        })

    print("=" * 65)
    total_copied = sum(len(v) for v in copied.values())
    print(f"\nTotal images copied: {total_copied}")
    print(f"Skipped (bad slug) : {len(bad_rows)}")
    print(f"Skipped (missing)  : {len(missing_files)}")

    if gap_classes:
        print(f"\nWARNING: ACTION REQUIRED before Stage 4 can run:")
        print(f"   The following confusable-cluster classes need more reference images:")
        for gc in gap_classes:
            cls_info = next(c for c in CLASSES if c["name"] == gc)
            print(f"   * {gc}  (display: {cls_info['display']}, cluster: {cls_info['cluster']})")
        print(f"   Minimum recommended: {MIN_REFS_PER_CONFUSABLE_CLASS} per confusable class.")
        print(f"   Add images to reference_images/<class>/ directly, or")
        print(f"   update reference_manifest.csv and re-run this script.")
    else:
        print("\nSUCCESS: All confusable-cluster classes have sufficient reference images.")

    print()

    # ── Save log ────────────────────────────────────────────────────────────
    if not dry_run:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / "stage1_ingest_log.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "manifest": str(manifest_path),
                    "src_dir": str(src_dir),
                    "total_copied": total_copied,
                    "skipped_bad_slug": len(bad_rows),
                    "skipped_missing_file": len(missing_files),
                    "gap_classes": gap_classes,
                    "coverage": log_data,
                },
                f,
                indent=2,
            )
        print(f"Log saved -> {log_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1b: Ingest hand-picked reference images into reference_images/<class>/ "
            "using a filled-in manifest CSV."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_DEFAULT,
        help=f"Path to the filled manifest CSV. Default: {MANIFEST_DEFAULT}",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help=(
            "Directory where the original images live. "
            "Defaults to the directory containing the manifest CSV."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen but do NOT copy any files.",
    )
    args = parser.parse_args()
    ingest(args.manifest, args.src, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
