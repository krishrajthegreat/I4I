"""
pipeline/stage6_contact_sheet.py
=================================
Stage 6 -- Manual Review Support / Contact Sheet Generator (PRD Section 3.5).

Generates two contact sheet formats per class for rapid manual inspection:
1. HTML Contact Sheet (logs/contact_sheets/<class>.html):
   Interactive HTML gallery showing thumbnails, filenames, and image dimensions.
2. Composite PNG Grid Image (logs/contact_sheets/<class>_grid.png):
   Single summary grid image compiled via PIL.

Usage
-----
    python -m pipeline.stage6_contact_sheet
    python -m pipeline.stage6_contact_sheet --classes lathe table_saw

Output
------
- Saved to logs/contact_sheets/<class>.html and logs/contact_sheets/<class>_grid.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import (
    ALL_CLASS_NAMES,
    CLASSES,
    DATASET_TRAIN_DIR,
    LOGS_DIR,
    class_by_name,
)

OUTPUT_DIR = LOGS_DIR / "contact_sheets"


def make_html_contact_sheet(cls_name: str, image_paths: list[Path], out_html: Path) -> None:
    cls = class_by_name(cls_name)
    display_name = cls["display"]

    cards_html = []
    for p in image_paths:
        try:
            rel_path = p.resolve().as_uri()
            with Image.open(p) as img:
                w, h = img.size
            card = f"""
            <div class="card">
                <img src="{rel_path}" alt="{p.name}" loading="lazy"/>
                <div class="info">
                    <span class="fname">{p.name}</span>
                    <span class="dim">{w}x{h} px</span>
                </div>
            </div>
            """
            cards_html.append(card)
        except Exception:
            continue

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Contact Sheet -- {display_name} ({cls_name})</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; margin: 20px; }}
        h1 {{ color: #38bdf8; margin-bottom: 5px; }}
        p.subtitle {{ color: #94a3b8; margin-top: 0; margin-bottom: 25px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
        .card {{ background: #1e293b; border-radius: 8px; overflow: hidden; border: 1px solid #334155; transition: transform 0.2s; }}
        .card:hover {{ transform: scale(1.03); border-color: #38bdf8; }}
        .card img {{ width: 100%; height: 180px; object-fit: cover; display: block; }}
        .info {{ padding: 8px; font-size: 11px; }}
        .fname {{ display: block; color: #e2e8f0; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .dim {{ color: #64748b; margin-top: 2px; display: block; }}
    </style>
</head>
<body>
    <h1>{display_name} ({cls_name})</h1>
    <p class="subtitle">Total Selected Images: {len(cards_html)} | Manual Review Pass (PRD Section 3.5)</p>
    <div class="grid">
        {"".join(cards_html)}
    </div>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_content, encoding="utf-8")


def make_png_grid(cls_name: str, image_paths: list[Path], out_png: Path, max_imgs: int = 64) -> None:
    if not image_paths:
        return

    sampled_paths = image_paths[:max_imgs]
    n = len(sampled_paths)
    cols = 8
    rows = math.ceil(n / cols)

    thumb_size = (150, 150)
    grid_w = cols * thumb_size[0]
    grid_h = rows * thumb_size[1]

    grid_img = Image.new("RGB", (grid_w, grid_h), (15, 23, 42))

    for idx, p in enumerate(sampled_paths):
        r = idx // cols
        c = idx % cols
        x = c * thumb_size[0]
        y = r * thumb_size[1]

        try:
            with Image.open(p) as img:
                img_conv = img.convert("RGB")
                img_conv.thumbnail(thumb_size)
                # Center on tile
                tile = Image.new("RGB", thumb_size, (30, 41, 59))
                off_x = (thumb_size[0] - img_conv.width) // 2
                off_y = (thumb_size[1] - img_conv.height) // 2
                tile.paste(img_conv, (off_x, off_y))
                grid_img.paste(tile, (x, y))
        except Exception:
            continue

    out_png.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(out_png, "PNG")


def process_class(cls_name: str) -> dict:
    class_dir = DATASET_TRAIN_DIR / cls_name
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]) if class_dir.exists() else []

    out_html = OUTPUT_DIR / f"{cls_name}.html"
    out_png = OUTPUT_DIR / f"{cls_name}_grid.png"

    if files:
        make_html_contact_sheet(cls_name, files, out_html)
        make_png_grid(cls_name, files, out_png)

    return {
        "class": cls_name,
        "image_count": len(files),
        "html_path": str(out_html) if files else None,
        "png_path": str(out_png) if files else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6: Manual Review Support / Contact Sheet Generator.")
    parser.add_argument("--classes", nargs="+", default=None, help="Generate contact sheets for specific class slugs")
    args = parser.parse_args()

    if args.classes:
        target_names = [c for c in args.classes if c in ALL_CLASS_NAMES]
    else:
        target_names = ALL_CLASS_NAMES

    print("=" * 65)
    print("STAGE 6 -- Manual Review Contact Sheet Generator")
    print(f"Classes: {len(target_names)}")
    print(f"Output : {OUTPUT_DIR}")
    print("=" * 65)

    results = []
    for cls_name in tqdm(target_names, desc="Contact Sheets", unit="class"):
        res = process_class(cls_name)
        results.append(res)

    print("\n" + "=" * 65)
    print("CONTACT SHEET SUMMARY")
    print("=" * 65)
    for r in results:
        print(f"  {r['class']:<22} {r['image_count']:>5} images -> {r['html_path']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
