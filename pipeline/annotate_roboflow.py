"""
pipeline/annotate_roboflow.py
==============================
Annotates already-uploaded images with full-image bounding boxes.

Use this when images were uploaded without annotations to a Roboflow
object-detection project. Each image gets one bounding box that covers
the full image, labelled with the class name derived from the folder name
(e.g., all images in dataset/train/conveyor/ get class = "conveyor").

Usage:
    .venv\\Scripts\\python.exe pipeline/annotate_roboflow.py

Config:
    Edit the CONFIG block below, or pass environment variables.
"""

import os, sys, tempfile, time
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
    print(f"  Images matching class '{class_name}': {len(class_images)}")

    if not class_images:
        print("  [WARN] No matching images found — check class name.")
        return

    success, failed = 0, 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for img in tqdm(class_images, desc=f"  Annotating {class_name}"):
            image_id   = img["id"]
            image_name = img["name"]
            width      = img.get("width", 640)
            height     = img.get("height", 480)

            # Build VOC XML
            ann_name = image_name.rsplit(".", 1)[0] + ".xml"
            xml      = make_voc_xml(image_name, width, height, class_name)
            xml_path = Path(tmpdir) / ann_name
            xml_path.write_text(xml, encoding="utf-8")

            # Upload via SDK's save_annotation (handles auth/retries)
            try:
                proj.save_annotation(
                    annotation_path=str(xml_path),
                    image_id=image_id,
                    annotation_overwrite=True,
                    num_retry_uploads=2,
                )
                success += 1
            except Exception as e:
                # Fallback: try REST directly
                try:
                    resp = save_annotation_rest(image_id, ann_name, xml)
                    if resp.status_code == 200:
                        success += 1
                    else:
                        failed += 1
                        tqdm.write(f"    [FAIL] {image_name}: {resp.status_code} {resp.text[:120]}")
                except Exception as e2:
                    failed += 1
                    tqdm.write(f"    [ERR]  {image_name}: {e2}")

            time.sleep(0.05)  # gentle rate-limit

    print(f"  Result: {success} annotated, {failed} failed")


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
