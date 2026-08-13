# -*- coding: utf-8 -*-
"""
pipeline/upload_to_roboflow.py
===============================
Stage 6 (Cloud) -- Upload cleaned dataset to Roboflow with an interactive CLI.

Walks you through:
  1. Class selection (validated against dataset/train/ folders)
  2. Roboflow project ID
  3. Train/valid/test split (validated to sum to 100)
  4. Per-class label naming (rename any folder slug to a custom Roboflow label)
  5. Upload with progress milestones every 25 images, per-class error isolation,
     and a final summary table.

Usage
-----
  .venv\\Scripts\\python.exe pipeline/upload_to_roboflow.py

Environment
-----------
  Set ROBOFLOW_API_KEY in a .env file at the project root (never hardcode it).
"""

from __future__ import annotations

import os
import sys
import math
import random
import tempfile
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("PYTHONUTF8", "1")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_DIR: Path = Path("dataset/train")
RANDOM_SEED: int = 42
EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
PROGRESS_MILESTONE: int = 25   # print a milestone line every N images


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
class ClassConfig(NamedTuple):
    folder: str          # folder name under dataset/train/
    label: str           # Roboflow annotation label (may differ from folder)
    images: list[Path]   # all image paths for this class


class UploadResult(NamedTuple):
    folder: str
    label: str
    total: int
    uploaded: int
    failed: int
    error: str | None    # set if the whole class crashed mid-run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hr(char: str = "=", width: int = 68) -> str:
    return char * width


def _discover_classes() -> list[str]:
    """Return sorted list of class folder names that actually contain images."""
    if not DATASET_DIR.exists():
        return []
    return sorted(
        d.name for d in DATASET_DIR.iterdir()
        if d.is_dir() and any(f.suffix.lower() in EXTS for f in d.iterdir())
    )


def _count_images(folder: str) -> int:
    d = DATASET_DIR / folder
    return sum(1 for f in d.iterdir() if f.suffix.lower() in EXTS)


def _list_images(folder: str) -> list[Path]:
    d = DATASET_DIR / folder
    return sorted(f for f in d.iterdir() if f.suffix.lower() in EXTS)


def _prompt(msg: str) -> str:
    """Prompt and strip; loop until non-empty."""
    while True:
        val = input(msg).strip()
        if val:
            return val
        print("  [!] Input cannot be empty, please try again.")


def _prompt_optional(msg: str, default: str = "") -> str:
    """Prompt; returns default if user presses Enter."""
    val = input(msg).strip()
    return val if val else default


# ---------------------------------------------------------------------------
# VOC XML annotation
# ---------------------------------------------------------------------------

def make_voc_xml(filename: str, width: int, height: int, label: str) -> str:
    """Pascal VOC XML with one full-image bounding box."""
    return (
        "<annotation>\n"
        f"  <folder>{label}</folder>\n"
        f"  <filename>{filename}</filename>\n"
        "  <size>\n"
        f"    <width>{width}</width>\n"
        f"    <height>{height}</height>\n"
        "    <depth>3</depth>\n"
        "  </size>\n"
        "  <object>\n"
        f"    <name>{label}</name>\n"
        "    <pose>Unspecified</pose>\n"
        "    <truncated>0</truncated>\n"
        "    <difficult>0</difficult>\n"
        "    <bndbox>\n"
        "      <xmin>1</xmin>\n"
        "      <ymin>1</ymin>\n"
        f"      <xmax>{max(width - 1, 2)}</xmax>\n"
        f"      <ymax>{max(height - 1, 2)}</ymax>\n"
        "    </bndbox>\n"
        "  </object>\n"
        "</annotation>"
    )


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_splits(
    images: list[Path],
    ratios: dict[str, float],
    seed: int = RANDOM_SEED,
) -> dict[str, list[Path]]:
    """Randomly shuffle images and assign to train/valid/test splits."""
    imgs = images.copy()
    random.seed(seed)
    random.shuffle(imgs)
    n = len(imgs)
    n_train = math.ceil(n * ratios["train"])
    n_valid = math.ceil(n * ratios["valid"])
    return {
        "train": imgs[:n_train],
        "valid": imgs[n_train : n_train + n_valid],
        "test":  imgs[n_train + n_valid :],
    }


# ---------------------------------------------------------------------------
# Interactive Q&A
# ---------------------------------------------------------------------------

def ask_classes(available: list[str]) -> list[str]:
    """
    Step 1 -- Class selection.
    Returns the validated list of folder names to upload.
    """
    print("\n" + _hr())
    print("STEP 1 -- CLASS SELECTION")
    print(_hr())
    print("  Available classes:")
    for name in available:
        print(f"    * {name}  ({_count_images(name)} images)")
    print()

    while True:
        raw = _prompt(
            "  Which class(es) to upload?\n"
            "  Enter a class name, comma-separated names, or 'all': "
        )

        if raw.lower() == "all":
            print(f"\n  All {len(available)} classes selected:")
            for name in available:
                print(f"      {name}")
            confirm = _prompt_optional("\n  Confirm? [Y/n]: ", default="y").lower()
            if confirm in ("y", "yes", ""):
                return list(available)
            print("  Restarting class selection...\n")
            continue

        selected = [s.strip() for s in raw.split(",") if s.strip()]
        invalid = [s for s in selected if s not in available]
        if invalid:
            print(f"\n  [!] Unknown class(es): {invalid}")
            print(f"      Valid names: {available}\n")
            continue

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped = [s for s in selected if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]
        return deduped


def ask_project_id() -> str:
    """Step 2 -- Roboflow project ID."""
    print("\n" + _hr())
    print("STEP 2 -- ROBOFLOW PROJECT ID")
    print(_hr())

    while True:
        pid = _prompt("  Enter your Roboflow project ID: ")
        confirm = _prompt_optional(f"  Project ID is '{pid}'. Confirm? [Y/n]: ", "y").lower()
        if confirm in ("y", "yes", ""):
            return pid
        print("  Re-entering project ID...\n")


def ask_split() -> dict[str, float]:
    """Step 3 -- Train/valid/test split percentages."""
    print("\n" + _hr())
    print("STEP 3 -- TRAIN/VALID/TEST SPLIT")
    print(_hr())

    while True:
        raw = _prompt_optional(
            "  Enter split as TRAIN/VALID/TEST percentages (e.g. 80/10/10)\n"
            "  or press Enter for default 70/20/10: ",
            default="70/20/10",
        )

        parts = [p.strip() for p in raw.split("/")]
        if len(parts) != 3:
            print("  [!] Expected exactly 3 numbers separated by '/'.  \n")
            continue

        try:
            pcts = [float(p) for p in parts]
        except ValueError:
            print("  [!] All three values must be numbers.\n")
            continue

        if abs(sum(pcts) - 100) > 0.01:
            print(f"  [!] Percentages must sum to 100, got {sum(pcts):.1f}.\n")
            continue

        train_r, valid_r, test_r = (p / 100 for p in pcts)
        ratios = {"train": train_r, "valid": valid_r, "test": test_r}
        print(
            f"\n  Split confirmed: "
            f"train={pcts[0]:.0f}%  valid={pcts[1]:.0f}%  test={pcts[2]:.0f}%"
        )
        return ratios


def ask_labels(selected_classes: list[str]) -> list[ClassConfig]:
    """
    Step 4 -- Per-class label naming + summary confirmation.
    Returns list of ClassConfig (folder, label, images).
    """
    print("\n" + _hr())
    print("STEP 4 -- LABEL NAMING (per class)")
    print(_hr())
    print("  Press Enter to keep the folder name, or type a custom Roboflow label.\n")

    configs: list[ClassConfig] = []
    for folder in selected_classes:
        imgs = _list_images(folder)
        label = _prompt_optional(
            f"  Label for '{folder}' ({len(imgs)} images) [default: {folder}]: ",
            default=folder,
        )
        configs.append(ClassConfig(folder=folder, label=label, images=imgs))

    # Summary table
    print("\n" + _hr("-"))
    print("  UPLOAD SUMMARY")
    print(_hr("-"))
    col_f = max(len(c.folder) for c in configs) + 2
    col_l = max(len(c.label)  for c in configs) + 2
    print(f"  {'Folder':<{col_f}} {'Label':<{col_l}} {'Images':>7}")
    print(f"  {'-'*col_f} {'-'*col_l} {'-'*7}")
    for c in configs:
        rename_tag = " <- renamed" if c.label != c.folder else ""
        print(f"  {c.folder:<{col_f}} {c.label:<{col_l}} {len(c.images):>7}{rename_tag}")
    print(_hr("-"))
    total_imgs = sum(len(c.images) for c in configs)
    print(f"  Total images to upload: {total_imgs}")
    print(_hr("-"))

    while True:
        go = _prompt_optional("\n  Start upload? [y/N]: ", default="n").lower()
        if go in ("y", "yes"):
            return configs
        if go in ("n", "no", ""):
            print("\n  Upload cancelled. Exiting.")
            sys.exit(0)
        print("  [!] Please enter y or n.")


# ---------------------------------------------------------------------------
# Core upload logic (one class at a time)
# ---------------------------------------------------------------------------

def upload_class(
    proj,
    cfg: ClassConfig,
    ratios: dict[str, float],
    tmpdir: str,
) -> UploadResult:
    """
    Upload all images for a single class to Roboflow.
    Returns an UploadResult; never raises (catches and records any exception).
    """
    from PIL import Image  # type: ignore

    splits = assign_splits(cfg.images, ratios)
    total = len(cfg.images)
    ok = fail = 0
    class_error: str | None = None

    print(f"\n  [{cfg.folder}] Starting -- {total} images -> label '{cfg.label}'")

    try:
        for split_name, split_imgs in splits.items():
            if not split_imgs:
                continue
            print(f"    > {split_name} ({len(split_imgs)} images)")

            for img_path in split_imgs:
                try:
                    # Measure image dimensions
                    w, h = 640, 480
                    try:
                        with Image.open(img_path) as im:
                            w, h = im.size
                    except Exception:
                        pass

                    # Generate and write VOC annotation
                    xml_str = make_voc_xml(img_path.name, w, h, cfg.label)
                    xml_path = Path(tmpdir) / f"{img_path.stem}.xml"
                    xml_path.write_text(xml_str, encoding="utf-8")

                    # Upload
                    proj.single_upload(
                        image_path=str(img_path),
                        annotation_path=str(xml_path),
                        split=split_name,
                        batch_name=cfg.folder,
                        num_retry_uploads=3,
                    )
                    ok += 1

                    # Milestone progress
                    cumulative = ok + fail
                    if cumulative % PROGRESS_MILESTONE == 0:
                        print(
                            f"    [{cfg.folder}] {cumulative}/{total} images uploaded...",
                            flush=True,
                        )

                except Exception as e:
                    fail += 1
                    print(f"      [FAIL] {img_path.name}: {e}", flush=True)

    except Exception as fatal:
        class_error = str(fatal)
        print(f"\n  [FATAL] [{cfg.folder}] Aborted mid-run: {fatal}", flush=True)

    # Per-class done line
    print(
        f"  [{cfg.folder}] Done -- {ok}/{total} uploaded, "
        f"labeled as '{cfg.label}', {fail} failed."
    )
    return UploadResult(
        folder=cfg.folder,
        label=cfg.label,
        total=total,
        uploaded=ok,
        failed=fail,
        error=class_error,
    )


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def print_summary(results: list[UploadResult], project_id: str) -> None:
    print("\n" + _hr())
    print("UPLOAD COMPLETE -- Final Summary")
    print(_hr())

    col_f = max(len(r.folder) for r in results) + 2
    col_l = max(len(r.label)  for r in results) + 2
    print(
        f"  {'Class':<{col_f}} {'Label':<{col_l}} "
        f"{'Total':>7} {'Uploaded':>9} {'Failed':>7}  Status"
    )
    print(
        f"  {'-'*col_f} {'-'*col_l} "
        f"{'-'*7} {'-'*9} {'-'*7}  {'-'*10}"
    )
    for r in results:
        if r.error:
            status = f"ABORTED ({r.error[:40]})"
        elif r.failed:
            status = f"WARN: {r.failed} failed"
        else:
            status = "OK"
        print(
            f"  {r.folder:<{col_f}} {r.label:<{col_l}} "
            f"{r.total:>7} {r.uploaded:>9} {r.failed:>7}  {status}"
        )

    grand_total = sum(r.total    for r in results)
    grand_ok    = sum(r.uploaded for r in results)
    grand_fail  = sum(r.failed   for r in results)
    print(_hr("-"))
    print(f"  {'TOTAL':<{col_f}} {'':<{col_l}} {grand_total:>7} {grand_ok:>9} {grand_fail:>7}")
    print(_hr())
    print(f"  View project: https://app.roboflow.com/{project_id}")
    print(_hr())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # -- API key --
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY not set. Add it to .env or set it in PowerShell.")
        sys.exit(1)

    # -- Imports --
    try:
        import roboflow  # type: ignore  # noqa: F401
    except ImportError:
        print("[ERROR] roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("[ERROR] Pillow not installed. Run: pip install Pillow")
        sys.exit(1)

    # -- Discover available classes --
    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR.resolve()}")
        sys.exit(1)

    available = _discover_classes()
    if not available:
        print(f"[ERROR] No class folders with images found in {DATASET_DIR.resolve()}")
        sys.exit(1)

    print("\n" + _hr())
    print(" Roboflow Interactive Uploader")
    print(_hr())

    # -- Q&A flow --
    selected   = ask_classes(available)   # Step 1
    project_id = ask_project_id()         # Step 2
    ratios     = ask_split()              # Step 3
    configs    = ask_labels(selected)     # Step 4 (includes confirmation table)

    # -- Connect to Roboflow --
    print(f"\n  Connecting to project '{project_id}'...", flush=True)
    try:
        import roboflow  # type: ignore
        rf   = roboflow.Roboflow(api_key=api_key)
        proj = rf.workspace().project(project_id)
    except Exception as e:
        print(f"[ERROR] Could not connect to Roboflow project: {e}")
        sys.exit(1)

    print("  Connected. Starting upload...\n")
    print(_hr())

    # -- Upload --
    results: list[UploadResult] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for cfg in configs:
            result = upload_class(proj, cfg, ratios, tmpdir)
            results.append(result)

    # -- Final summary --
    print_summary(results, project_id)


if __name__ == "__main__":
    main()
