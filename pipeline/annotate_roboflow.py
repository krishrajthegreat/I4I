""" 
pipeline/annotate_roboflow.py
==============================
Annotates already-uploaded images with full-image bounding boxes AND ensures
a proper train/valid/test split is applied.

Strategy:
  - All images are already in Roboflow (split=train from initial upload).
  - This script:
      1. Lists all images in the project.
      2. Assigns splits (70/20/10) using the same deterministic seed as upload.
      3. Annotates every image via save_annotation() (full-image VOC XML).
      4. Re-uploads valid+test images using single_upload(split=...) so
         Roboflow records the correct split for those images.

Usage:
    .venv\\Scripts\\python.exe pipeline/annotate_roboflow.py
"""

import os, sys, tempfile, time, math, random
os.environ.setdefault("PYTHONUTF8", "1")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pathlib import Path
from tqdm import tqdm
import requests

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
API_KEY    = os.environ.get("ROBOFLOW_API_KEY", "").strip()
WORKSPACE  = "krish-raj-qrke3"
PROJECT_ID = "conveyer-kniz0"
DATASET_DIR = Path("dataset/train")

# Classes to annotate. Set to None to annotate ALL class folders.
CLASSES_TO_ANNOTATE: list | None = ["conveyor"]

# Same split ratios + seed as upload_to_roboflow.py
SPLIT_RATIOS  = {"train": 0.70, "valid": 0.20, "test": 0.10}
RANDOM_SEED   = 42
# ---------------------------------------------------------------------------


def make_voc_xml(filename: str, width: int, height: int, class_name: str) -> str:
    """Pascal VOC XML with one full-image bounding box."""
    return (
        "<annotation>\n"
        f"  <folder>{class_name}</folder>\n"
        f"  <filename>{filename}</filename>\n"
        "  <size>\n"
        f"    <width>{width}</width>\n"
        f"    <height>{height}</height>\n"
        "    <depth>3</depth>\n"
        "  </size>\n"
        "  <object>\n"
        f"    <name>{class_name}</name>\n"
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


def save_annotation_rest(image_id: str, annotation_name: str, annotation_xml: str) -> dict:
    """POST annotation to Roboflow via REST (mirrors what SDK save_annotation does)."""
    url = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT_ID}/images/{image_id}/annotation"
    resp = requests.post(
        url,
        params={"api_key": API_KEY, "overwrite": "true"},
        json={"name": annotation_name, "data": annotation_xml},
        timeout=30,
    )
    return resp


def assign_splits(images: list, ratios: dict, seed: int = 42) -> dict:
    """Deterministic 70/20/10 split — same logic as upload_to_roboflow.py."""
    imgs = images.copy()
    random.seed(seed)
    random.shuffle(imgs)
    n = len(imgs)
    n_train = math.ceil(n * ratios["train"])
    n_valid = math.ceil(n * ratios["valid"])
    return {
        "train": set(x["id"] for x in imgs[:n_train]),
        "valid": set(x["id"] for x in imgs[n_train:n_train + n_valid]),
        "test":  set(x["id"] for x in imgs[n_train + n_valid:]),
    }


def annotate_class(proj, class_name: str) -> None:
    print(f"\n  Fetching image list for class '{class_name}' ...")

    # Collect ALL image IDs via paginated search
    all_images = []
    offset = 0
    page_size = 100
    while True:
        page = proj.search(offset=offset, limit=page_size,
                           fields=["id", "name", "width", "height"])
        if not page:
            break
        all_images.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    print(f"  Total images in project: {len(all_images)}")

    # Filter to this class by filename prefix
    class_images = [img for img in all_images
                    if img.get("name", "").startswith(f"img_{class_name}_")]
    print(f"  Images matching '{class_name}': {len(class_images)}")

    if not class_images:
        print("  [WARN] No matching images found.")
        return

    # Assign splits (deterministic, same seed as upload script)
    split_sets = assign_splits(class_images, SPLIT_RATIOS, RANDOM_SEED)
    print(f"  Split: train={len(split_sets['train'])}  "
          f"valid={len(split_sets['valid'])}  test={len(split_sets['test'])}")

    # Also build local image map for re-uploading valid/test with correct split
    cls_dir = DATASET_DIR / class_name
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    local_files = {f.name: f for f in cls_dir.iterdir() if f.suffix.lower() in exts}

    ann_ok = ann_fail = split_ok = split_fail = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for img in tqdm(class_images, desc=f"  Processing {class_name}"):
            image_id   = img["id"]
            image_name = img["name"]
            width      = img.get("width", 640)
            height     = img.get("height", 480)

            # Determine this image's target split
            if image_id in split_sets["valid"]:
                target_split = "valid"
            elif image_id in split_sets["test"]:
                target_split = "test"
            else:
                target_split = "train"

            # Build VOC XML
            ann_name = image_name.rsplit(".", 1)[0] + ".xml"
            xml      = make_voc_xml(image_name, width, height, class_name)
            xml_path = Path(tmpdir) / ann_name
            xml_path.write_text(xml, encoding="utf-8")

            # --- Annotate (works for all images already in Roboflow) ---
            try:
                proj.save_annotation(
                    annotation_path=str(xml_path),
                    image_id=image_id,
                    annotation_overwrite=True,
                    num_retry_uploads=2,
                )
                ann_ok += 1
            except Exception as e:
                ann_fail += 1
                tqdm.write(f"    [ANN-FAIL] {image_name}: {e}")

            # --- Re-upload valid/test images with correct split ---
            if target_split in ("valid", "test"):
                local_path = local_files.get(image_name)
                if local_path:
                    try:
                        proj.single_upload(
                            image_path=str(local_path),
                            annotation_path=str(xml_path),
                            split=target_split,
                            batch_name=class_name,
                            annotation_overwrite=True,
                            num_retry_uploads=2,
                        )
                        split_ok += 1
                    except Exception as e:
                        split_fail += 1
                        tqdm.write(f"    [SPLIT-FAIL] {image_name}: {e}")

            time.sleep(0.05)

    print(f"  Annotations: {ann_ok} ok, {ann_fail} failed")
    print(f"  Split fixes : {split_ok} ok, {split_fail} failed  "
          f"(valid+test images re-uploaded with correct split)")


def main():
    if not API_KEY:
        print("[ERROR] ROBOFLOW_API_KEY not set. Add it to .env or set in shell.")
        sys.exit(1)

    import roboflow
    rf   = roboflow.Roboflow(api_key=API_KEY)
    proj = rf.workspace().project(PROJECT_ID)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_classes = sorted([
        d.name for d in DATASET_DIR.iterdir()
        if d.is_dir() and any(f.suffix.lower() in exts for f in d.iterdir())
    ])

    target = CLASSES_TO_ANNOTATE if CLASSES_TO_ANNOTATE is not None else all_classes

    print("=" * 60)
    print("ROBOFLOW ANNOTATION — Full-image bounding boxes")
    print("=" * 60)
    print(f"  Workspace  : {WORKSPACE}")
    print(f"  Project    : {PROJECT_ID}")
    print(f"  Classes    : {target}")
    print("=" * 60)

    for cls in target:
        annotate_class(proj, cls)

    print("\n[DONE] All classes annotated.")
    print(f"  View: https://app.roboflow.com/{WORKSPACE}/{PROJECT_ID}/annotate")


if __name__ == "__main__":
    main()
