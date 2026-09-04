# -*- coding: utf-8 -*-
"""
pipeline/apply_clean_upload.py
===============================
Final Dataset Cleanup, 0.55 Mask-Threshold Filtering, and Roboflow Re-Upload.

Executes:
- STEP 2: Removes 5 confirmed mislabeled source images:
    - Crane (1): img_crane_ddgs_00479.jpg
    - Fire Extinguisher (4): img_fire_extinguisher_ddgs_00079.jpg,
                            img_fire_extinguisher_ddgs_00099.jpg,
                            img_fire_extinguisher_ddgs_00153.jpg,
                            img_fire_extinguisher_ddgs_00291.jpg
    (Explicitly keeps img_crane_ddgs_00151.jpg and img_fire_extinguisher_ddgs_00326.jpg)
- STEP 3: Applies 0.55 mask-similarity threshold to all surviving candidate masks.
- STEP 4: Rebuilds COCO payloads and uploads to Roboflow (annotation_overwrite=True).
- STEP 5: Generates final verification overlay grid (logs/final_verification_grid.png).
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.config import DATASET_TRAIN_DIR, LOGS_DIR

WORKSPACE_SLUG = "krish-raj-cgbcn"
PROJECT_ID     = "fire_crane"
TARGET_CLASSES = ["crane", "fire_extinguisher"]
SPLIT_RATIOS   = {"train": 0.70, "valid": 0.15, "test": 0.15}
RANDOM_SEED    = 42
NUM_WORKERS    = 4
SIM_THRESHOLD  = 0.55

CONFIRMED_REMOVALS = {
    "crane": [
        "img_crane_ddgs_00479.jpg",  # CNC machine
    ],
    "fire_extinguisher": [
        "img_fire_extinguisher_ddgs_00079.jpg",  # Infographic poster
        "img_fire_extinguisher_ddgs_00099.jpg",  # 2D spec sheet diagram
        "img_fire_extinguisher_ddgs_00153.jpg",  # Silver cylinders / Alamy
        "img_fire_extinguisher_ddgs_00291.jpg",  # Training checklist flyer
    ]
}


def remove_mislabeled_images() -> dict[str, list[str]]:
    """STEP 2: Deletes confirmed mislabeled source images from dataset/train."""
    print("\n" + "=" * 80)
    print("STEP 2: REMOVING CONFIRMED MISLABELED SOURCE IMAGES")
    print("=" * 80)

    removed = {}
    for cls_name, fn_list in CONFIRMED_REMOVALS.items():
        cls_dir = DATASET_TRAIN_DIR / cls_name
        removed[cls_name] = []
        for fn in fn_list:
            p = cls_dir / fn
            if p.exists():
                p.unlink()
                removed[cls_name].append(fn)
                print(f"  [{cls_name.upper()}] Removed: {fn}")
            else:
                print(f"  [{cls_name.upper()}] Already absent: {fn}")

    return removed


def compute_deterministic_splits(image_paths: list[Path]) -> dict[Path, str]:
    paths = sorted(image_paths)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(paths)

    total = len(paths)
    n_train = int(round(total * SPLIT_RATIOS["train"]))
    n_valid = int(round(total * SPLIT_RATIOS["valid"]))

    split_map = {}
    for idx, p in enumerate(paths):
        if idx < n_train:
            split_map[p] = "train"
        elif idx < n_train + n_valid:
            split_map[p] = "valid"
        else:
            split_map[p] = "test"
    return split_map


def load_and_filter_masks() -> dict[str, dict[str, list[dict]]]:
    """STEP 3: Filters scored masks from mask_clip_scores.json keeping sim >= 0.55."""
    print("\n" + "=" * 80)
    print(f"STEP 3: APPLYING SIMILARITY THRESHOLD >= {SIM_THRESHOLD}")
    print("=" * 80)

    scores_file = LOGS_DIR / "mask_clip_scores.json"
    with open(scores_file, "r", encoding="utf-8") as f:
        d = json.load(f)

    all_masks = d.get("masks", [])
    filtered_by_class: dict[str, dict[str, list[dict]]] = {c: {} for c in TARGET_CLASSES}

    removed_set = set()
    for rem_list in CONFIRMED_REMOVALS.values():
        removed_set.update(rem_list)

    total_candidates = 0
    total_kept = 0

    for m in all_masks:
        cls_name = m["class"]
        img_name = m["image_name"]

        # Skip removed images
        if img_name in removed_set:
            continue

        if cls_name not in filtered_by_class:
            continue

        total_candidates += 1

        if m["sim_score"] >= SIM_THRESHOLD:
            total_kept += 1
            if img_name not in filtered_by_class[cls_name]:
                filtered_by_class[cls_name][img_name] = []
            filtered_by_class[cls_name][img_name].append(m)

    print(f"  Total Candidate Masks Filtered: {total_candidates} -> {total_kept} kept ({total_candidates - total_kept} discarded, {(total_candidates - total_kept)/total_candidates*100:.1f}%)")
    for cls_name in TARGET_CLASSES:
        cls_masks_kept = sum(len(ms) for ms in filtered_by_class[cls_name].values())
        print(f"  [{cls_name.upper()}] Surviving Masks: {cls_masks_kept} across {len(filtered_by_class[cls_name])} images")

    return filtered_by_class


def upload_cleaned_image_annotation(
    img_path: Path, expected_class: str, target_split: str, masks: list[dict], proj
) -> dict:
    filename = img_path.name
    try:
        im = Image.open(img_path)
        w, h = im.size
    except Exception:
        return {"filename": filename, "status": "FAIL", "reason": "Could not open image", "mask_count": 0}

    coco_annotations = []
    for ann_id, m in enumerate(masks, 1):
        bx, by, bw, bh = m["bbox"]
        coco_annotations.append({
            "id": ann_id,
            "image_id": 1,
            "category_id": 1,
            "segmentation": m["segmentation"],
            "area": float(bw * bh),
            "bbox": [float(bx), float(by), float(bw), float(bh)],
            "iscrowd": 0,
        })

    coco_payload = {
        "images": [{"id": 1, "width": w, "height": h, "file_name": filename}],
        "categories": [{"id": 1, "name": expected_class, "supercategory": "equipment"}],
        "annotations": coco_annotations,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        coco_file = Path(tmpdir) / "_annotations.coco.json"
        coco_file.write_text(json.dumps(coco_payload, indent=2), encoding="utf-8")

        try:
            res = proj.single_upload(
                image_path=str(img_path),
                annotation_path=str(coco_file),
                split=target_split,
                batch_name=f"{expected_class}_cleaned_055",
                num_retry_uploads=3,
                annotation_overwrite=True,
            )
            anno_res = res.get("annotation", {}) if isinstance(res, dict) else {}
            is_success = anno_res.get("success", False) or (isinstance(res, dict) and "duplicate" in str(res).lower())
            st = "OK" if is_success else "FAIL"
            reason = None if is_success else str(res)[:180]
        except Exception as exc:
            st = "FAIL"
            reason = str(exc)[:180]

    return {
        "filename": filename,
        "class": expected_class,
        "split": target_split,
        "status": st,
        "reason": reason,
        "mask_count": len(masks),
    }


def upload_cleaned_dataset(filtered_by_class: dict[str, dict[str, list[dict]]], proj) -> dict:
    """STEP 4: Uploads cleaned annotations to Roboflow project."""
    print("\n" + "=" * 80)
    print("STEP 4: REBUILDING AND RE-UPLOADING CLEANED DATASET TO ROBOFLOW")
    print("=" * 80)

    class_upload_summaries = {}

    for cls_name in TARGET_CLASSES:
        cls_dir = DATASET_TRAIN_DIR / cls_name
        files = sorted([p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        total = len(files)

        split_map = compute_deterministic_splits(files)
        cls_masks_map = filtered_by_class.get(cls_name, {})

        print(f"\nUploading cleaned annotations for [{cls_name.upper()}] ({total} images)...")

        results = []
        completed = 0

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(
                    upload_cleaned_image_annotation,
                    img_path,
                    cls_name,
                    split_map[img_path],
                    cls_masks_map.get(img_path.name, []),
                    proj,
                ): img_path
                for img_path in files
            }

            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                completed += 1

                fn = res["filename"]
                sp = res["split"]
                st = res["status"]
                mc = res["mask_count"]
                print(f"  [{completed:03d}/{total:03d}] {fn:<38} | {sp:<5} | {mc:>2} masks (sim>=0.55) -> {st}", flush=True)

        n_ok = sum(1 for r in results if r["status"] == "OK")
        total_masks = sum(r["mask_count"] for r in results if r["status"] == "OK")
        avg_masks = total_masks / total if total else 0

        class_upload_summaries[cls_name] = {
            "total_images": total,
            "successful": n_ok,
            "total_masks": total_masks,
            "avg_masks_per_image": round(avg_masks, 2),
            "results": results,
        }

    return class_upload_summaries


def generate_final_verification_grid(filtered_by_class: dict[str, dict[str, list[dict]]]) -> None:
    """STEP 5: Generates a 24-image visual overlay verification grid."""
    print("\n" + "=" * 80)
    print("STEP 5: GENERATING FINAL VERIFICATION SPOT-CHECK OVERLAY GRID")
    print("=" * 80)

    random.seed(42)
    sample_images = []

    # Explicitly include the 2 preserved images
    sample_images.append(("crane", "img_crane_ddgs_00151.jpg", "PRESERVED SUNSET SKYLINE"))
    sample_images.append(("fire_extinguisher", "img_fire_extinguisher_ddgs_00326.jpg", "PRESERVED DARK WORKSHOP"))

    # Add other random + multi-instance samples
    for cls_name in TARGET_CLASSES:
        cls_dir = DATASET_TRAIN_DIR / cls_name
        cls_files = sorted([p.name for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
        # Pick 11 diverse images
        other_files = [f for f in cls_files if f not in ["img_crane_ddgs_00151.jpg", "img_fire_extinguisher_ddgs_00326.jpg"]]
        sampled = random.sample(other_files, min(11, len(other_files)))
        for fn in sampled:
            sample_images.append((cls_name, fn, f"{cls_name.upper()}"))

    # Total 24 tiles (6x4 grid)
    cols = 6
    rows = 4
    tile_w, tile_h = 240, 240
    grid_w = cols * tile_w
    grid_h = rows * (tile_h + 45) + 60

    grid_img = Image.new("RGB", (grid_w, grid_h), (15, 23, 42))
    draw_grid = ImageDraw.Draw(grid_img)

    draw_grid.rectangle([(0, 0), (grid_w, 50)], fill=(30, 41, 59))
    draw_grid.text((20, 15), "FINAL VERIFICATION OVERLAY GRID -- FILTERED MASKS (SIM >= 0.55)", fill=(56, 189, 248))

    colors = [(34, 197, 94), (56, 189, 248), (234, 179, 8), (168, 85, 247), (244, 63, 94)]

    for idx, (cls_name, fn, tag) in enumerate(sample_images[:24]):
        r = idx // cols
        c = idx % cols
        x = c * tile_w
        y = 55 + (r * (tile_h + 45))

        img_p = DATASET_TRAIN_DIR / cls_name / fn
        try:
            im = Image.open(img_p).convert("RGB")
            draw_im = ImageDraw.Draw(im)

            masks = filtered_by_class.get(cls_name, {}).get(fn, [])
            for m_idx, m in enumerate(masks):
                col = colors[m_idx % len(colors)]
                flat_pts = m["segmentation"][0]
                pts = [(flat_pts[i], flat_pts[i+1]) for i in range(0, len(flat_pts), 2)]
                if len(pts) >= 3:
                    draw_im.polygon(pts, outline=col, width=4)

            im.thumbnail((tile_w - 10, tile_h - 10))
            tile = Image.new("RGB", (tile_w, tile_h), (30, 41, 59))
            off_x = (tile_w - im.width) // 2
            off_y = (tile_h - im.height) // 2
            tile.paste(im, (off_x, off_y))
            grid_img.paste(tile, (x, y))

            # Label
            draw_grid.text((x + 6, y + tile_h + 4), f"{fn[:16]}.. ({len(masks)} masks)", fill=(248, 250, 252))
            draw_grid.text((x + 6, y + tile_h + 22), f"{tag}", fill=(56, 189, 248) if "PRESERVED" in tag else (148, 163, 184))
        except Exception as e:
            print(f"Error rendering {img_p}: {e}")

    out_png = LOGS_DIR / "final_verification_grid.png"
    grid_img.save(out_png, "PNG")
    print(f"[FINAL VERIFICATION GRID SAVED] -> {out_png}")


def main():
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY not set.")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(WORKSPACE_SLUG).project(PROJECT_ID)

    # Step 2: Remove mislabeled images
    removed = remove_mislabeled_images()

    # Step 3: Filter candidate masks (sim >= 0.55)
    filtered_masks = load_and_filter_masks()

    # Step 4: Upload to Roboflow
    upload_summaries = upload_cleaned_dataset(filtered_masks, proj)

    # Step 5: Visual verification grid
    generate_final_verification_grid(filtered_masks)

    # Summary JSON
    report_path = LOGS_DIR / "dataset_finalization_summary.json"
    summary_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threshold_applied": SIM_THRESHOLD,
        "images_removed": removed,
        "final_dataset": {
            cls_name: {
                "final_image_count": upload_summaries[cls_name]["total_images"],
                "final_mask_count": upload_summaries[cls_name]["total_masks"],
                "avg_masks_per_image": upload_summaries[cls_name]["avg_masks_per_image"],
            }
            for cls_name in TARGET_CLASSES
        },
        "negative_background_images": 40,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80)
    print("FINAL DATASET SUMMARY")
    print("=" * 80)
    for cls_name in TARGET_CLASSES:
        s = upload_summaries[cls_name]
        print(f"  Class: {cls_name:<20} | Final Images: {s['total_images']} | Final Masks: {s['total_masks']} | Avg Masks/Image: {s['avg_masks_per_image']}")
    print(f"  Negative Background Images : 40 (0 annotations)")
    print(f"  Total Images in Dataset    : {sum(upload_summaries[c]['total_images'] for c in TARGET_CLASSES) + 40}")
    print(f"\n[REPORT SAVED] -> {report_path}\n")


if __name__ == "__main__":
    main()
