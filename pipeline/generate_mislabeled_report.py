# -*- coding: utf-8 -*-
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

flagged_list = [
    {
        "class": "crane",
        "filename": "img_crane_ddgs_00479.jpg",
        "content_type": "CNC Milling / Machining Center",
        "issue": "Industrial milling/lathe machine misfiled as crane.",
        "own_sim": 0.523,
        "alt_class": "cnc_milling",
        "alt_sim": 0.699,
        "recommendation": "REMOVE (Mislabeled machine)"
    },
    {
        "class": "crane",
        "filename": "img_crane_ddgs_00151.jpg",
        "content_type": "Electrical Switchgear / Control Panel",
        "issue": "Close-up industrial electrical cabinet, no crane visible.",
        "own_sim": 0.442,
        "alt_class": "control_panel",
        "alt_sim": 0.522,
        "recommendation": "REMOVE (Mislabeled machine)"
    },
    {
        "class": "fire_extinguisher",
        "filename": "img_fire_extinguisher_ddgs_00079.jpg",
        "content_type": "Safety Infographic / Poster",
        "issue": "Graphic poster with text, icons, and diagrams (not a physical photo).",
        "own_sim": 0.481,
        "alt_class": "infographic_diagram",
        "alt_sim": 0.590,
        "recommendation": "REMOVE (Non-photo graphic)"
    },
    {
        "class": "fire_extinguisher",
        "filename": "img_fire_extinguisher_ddgs_00099.jpg",
        "content_type": "Product Spec Sheet / Vector Graphic",
        "issue": "Digital 2D vector schematic with technical tables.",
        "own_sim": 0.509,
        "alt_class": "infographic_diagram",
        "alt_sim": 0.584,
        "recommendation": "REMOVE (Non-photo graphic)"
    },
    {
        "class": "fire_extinguisher",
        "filename": "img_fire_extinguisher_ddgs_00153.jpg",
        "content_type": "Fire Alarm Control Panel / Call Point",
        "issue": "Alarm bell & wall control panel, no extinguisher present.",
        "own_sim": 0.477,
        "alt_class": "control_panel",
        "alt_sim": 0.580,
        "recommendation": "REMOVE (Mislabeled object)"
    },
    {
        "class": "fire_extinguisher",
        "filename": "img_fire_extinguisher_ddgs_00291.jpg",
        "content_type": "Safety Training Flyer / Text Chart",
        "issue": "Printed checklist flyer with clipart extinguisher icons.",
        "own_sim": 0.446,
        "alt_class": "infographic_diagram",
        "alt_sim": 0.572,
        "recommendation": "REMOVE (Non-photo graphic)"
    },
    {
        "class": "fire_extinguisher",
        "filename": "img_fire_extinguisher_ddgs_00326.jpg",
        "content_type": "Electrical Junction Box",
        "issue": "Wall electrical box misfiled during web scraping.",
        "own_sim": 0.432,
        "alt_class": "control_panel",
        "alt_sim": 0.684,
        "recommendation": "REMOVE (Mislabeled object)"
    }
]

# Generate Composite PNG Grid
n = len(flagged_list)
cols = 4
rows = math.ceil(n / cols)
tile_w, tile_h = 320, 260
header_h = 60
footer_h = 75

grid_w = cols * tile_w
grid_h = header_h + rows * (tile_h + footer_h)

grid_img = Image.new("RGB", (grid_w, grid_h), (15, 23, 42))
draw_grid = ImageDraw.Draw(grid_img)

draw_grid.rectangle([(0, 0), (grid_w, header_h)], fill=(30, 41, 59))
draw_grid.text((20, 18), f"FLAGGED MISLABELED SOURCE IMAGES AUDIT ({n} Total Images Flagged)", fill=(239, 68, 68))

for idx, item in enumerate(flagged_list):
    r = idx // cols
    c = idx % cols
    x = c * tile_w
    y = header_h + r * (tile_h + footer_h)
    
    img_p = Path(f"dataset/train/{item['class']}/{item['filename']}")
    try:
        im = Image.open(img_p).convert("RGB")
        im.thumbnail((tile_w - 16, tile_h - 16))
        tile = Image.new("RGB", (tile_w - 16, tile_h - 16), (30, 41, 59))
        off_x = ((tile_w - 16) - im.width) // 2
        off_y = ((tile_h - 16) - im.height) // 2
        tile.paste(im, (off_x, off_y))
        grid_img.paste(tile, (x + 8, y + 8))
        
        draw_grid.text((x + 8, y + tile_h - 4), f"{item['class'].upper()}: {item['filename'][:18]}..", fill=(248, 250, 252))
        draw_grid.text((x + 8, y + tile_h + 12), f"Actual: {item['content_type']}", fill=(244, 63, 94))
        draw_grid.text((x + 8, y + tile_h + 28), f"Closer to: {item['alt_class']} ({item['alt_sim']:.2f})", fill=(234, 179, 8))
        draw_grid.text((x + 8, y + tile_h + 44), f"Action: {item['recommendation']}", fill=(239, 68, 68))
    except Exception as e:
        print(f"Error rendering {img_p}: {e}")

out_png = Path("logs/flagged_mislabeled_candidates.png")
grid_img.save(out_png, "PNG")
print(f"Generated visual grid: {out_png}")

# Interactive HTML Gallery
cards_html = []
for item in flagged_list:
    p = Path(f"dataset/train/{item['class']}/{item['filename']}")
    uri = p.resolve().as_uri()
    card = f"""
    <div class="card">
        <a href="{uri}" target="_blank"><img src="{uri}" alt="{item['filename']}" loading="lazy"/></a>
        <div class="info">
            <span class="fname">{item['filename']}</span>
            <span class="badge {item['class']}">{item['class'].upper()}</span>
            <div class="detected"><strong>Identified Content:</strong> <span class="highlight">{item['content_type']}</span></div>
            <div class="issue"><strong>Issue:</strong> {item['issue']}</div>
            <div class="scores">
                <div><strong>Own Class Similarity:</strong> {item['own_sim']:.3f}</div>
                <div><strong>Best Match ({item['alt_class']}):</strong> {item['alt_sim']:.3f}</div>
            </div>
            <div class="action"><strong>Recommended Action:</strong> {item['recommendation']}</div>
        </div>
    </div>
    """
    cards_html.append(card)

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Flagged Mislabeled Candidate Images Review</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b1120; color: #f8fafc; margin: 0; padding: 24px; }}
        h1 {{ color: #ef4444; margin: 0 0 8px 0; }}
        p.subtitle {{ color: #94a3b8; margin: 0 0 24px 0; font-size: 14px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: #0f172a; border: 1px solid #334155; border-radius: 10px; overflow: hidden; display: flex; flex-direction: column; }}
        .card img {{ width: 100%; height: 220px; object-fit: cover; display: block; background: #1e293b; }}
        .info {{ padding: 14px; font-size: 12px; display: flex; flex-direction: column; gap: 8px; }}
        .fname {{ font-weight: 700; color: #e2e8f0; word-break: break-all; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; width: fit-content; }}
        .badge.crane {{ background: #0284c7; color: white; }}
        .badge.fire_extinguisher {{ background: #dc2626; color: white; }}
        .detected {{ font-size: 13px; color: #f1f5f9; }}
        .highlight {{ color: #f87171; font-weight: 700; }}
        .issue {{ color: #cbd5e1; font-size: 11px; line-height: 1.4; }}
        .scores {{ background: #1e293b; padding: 8px; border-radius: 6px; }}
        .action {{ background: #450a0a; border: 1px solid #7f1d1d; color: #fca5a5; padding: 6px 10px; border-radius: 4px; font-weight: 600; }}
    </style>
</head>
<body>
    <h1>Flagged Mislabeled Candidate Images Review</h1>
    <p class="subtitle">Total Flagged: {len(flagged_list)} source images across crane and fire_extinguisher. Review and confirm before removal.</p>
    <div class="grid">
        {"".join(cards_html)}
    </div>
</body>
</html>
"""
out_html = Path("logs/flagged_mislabeled_candidates.html")
out_html.write_text(html_doc, encoding="utf-8")
print(f"Generated HTML gallery: {out_html}")
