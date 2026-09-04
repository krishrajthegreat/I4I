# -*- coding: utf-8 -*-
"""
pipeline/sub_type_audit.py
===========================
Diagnostic Visual Sub-Type Audit for the 4 Weak-Performing Classes:
  - cnc_milling
  - conveyor
  - packaging_machine
  - grinding

Features:
- Extracts L2-normalized OpenCLIP (ViT-B-32 / laion2b_s34b_b79k) image embeddings.
- Runs K-Means clustering with k=4 per class.
- Selects 16 most representative images per cluster (closest to cluster centroid).
- Generates composite PNG grid contact sheets and interactive HTML contact sheets.
- Saves full audit log to logs/sub_type_audit_report.json.
- STRICTLY READ-ONLY: No files or directories in dataset/train/ are deleted, merged, or modified.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pipeline.config import (
    CLIP_BATCH_SIZE,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DATASET_TRAIN_DIR,
    LOGS_DIR,
    class_by_name,
)

try:
    import open_clip
except ImportError:
    print("[ERROR] open_clip not installed. Run: pip install open-clip-torch")
    sys.exit(1)

AUDIT_DIR = LOGS_DIR / "sub_type_audit"
TARGET_CLASSES = ["cnc_milling", "conveyor", "packaging_machine", "grinding"]
DEFAULT_K = 4
SAMPLES_PER_CLUSTER = 16
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_clip_model():
    """Loads OpenCLIP model in evaluation mode."""
    device = "cpu"
    if torch.cuda.is_available():
        try:
            _dummy = torch.zeros(1, device="cuda") + 1
            device = "cuda"
        except Exception:
            device = "cpu"

    print(f"Loading OpenCLIP ({CLIP_MODEL_NAME} / {CLIP_PRETRAINED}) on {device}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=device
    )
    model.eval()
    return model, preprocess, device


def get_class_image_embeddings(
    image_paths: list[Path], model, preprocess, device, batch_size: int = CLIP_BATCH_SIZE
) -> tuple[list[Path], np.ndarray]:
    """Generates L2-normalized CLIP embeddings for a list of image paths (Read-Only)."""
    valid_paths = []
    tensors = []

    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            t = preprocess(img)
            tensors.append(t)
            valid_paths.append(p)
        except Exception:
            pass

    if not tensors:
        return [], np.empty((0, 512), dtype=np.float32)

    all_embs = []
    for i in range(0, len(tensors), batch_size):
        batch = torch.stack(tensors[i : i + batch_size]).to(device)
        with torch.no_grad():
            embs = model.encode_image(batch)
            embs /= embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu().numpy())

    return valid_paths, np.vstack(all_embs)


def generate_cluster_png_grid(
    cls_name: str,
    cluster_id: int,
    image_paths: list[Path],
    out_png: Path,
    cols: int = 4,
    thumb_size: tuple[int, int] = (220, 220),
) -> None:
    """Generates a composite PNG contact sheet for a specific cluster."""
    if not image_paths:
        return

    n = len(image_paths)
    rows = math.ceil(n / cols)

    header_h = 50
    footer_h = 30
    grid_w = cols * thumb_size[0]
    grid_h = header_h + (rows * (thumb_size[1] + footer_h))

    grid_img = Image.new("RGB", (grid_w, grid_h), (15, 23, 42))
    draw = ImageDraw.Draw(grid_img)

    # Header banner
    draw.rectangle([(0, 0), (grid_w, header_h)], fill=(30, 41, 59))
    header_text = f"{cls_name.upper()} -- Cluster #{cluster_id} (Sample of {n} images)"
    draw.text((15, 15), header_text, fill=(56, 189, 248))

    for idx, p in enumerate(image_paths):
        r = idx // cols
        c = idx % cols
        x = c * thumb_size[0]
        y = header_h + (r * (thumb_size[1] + footer_h))

        try:
            with Image.open(p) as img:
                img_conv = img.convert("RGB")
                img_conv.thumbnail(thumb_size)

                # Centered thumbnail on tile
                tile = Image.new("RGB", thumb_size, (15, 23, 42))
                off_x = (thumb_size[0] - img_conv.width) // 2
                off_y = (thumb_size[1] - img_conv.height) // 2
                tile.paste(img_conv, (off_x, off_y))
                grid_img.paste(tile, (x, y))

                # Image label below tile
                label_y = y + thumb_size[1] + 4
                fname = p.name[:24] + "..." if len(p.name) > 27 else p.name
                draw.text((x + 6, label_y), fname, fill=(226, 232, 240))
        except Exception:
            continue

    out_png.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(out_png, "PNG")


def generate_class_composite_grid(
    cls_name: str,
    clusters_data: list[dict],
    out_png: Path,
    thumb_size: tuple[int, int] = (160, 160),
) -> None:
    """Generates a master 4-row composite PNG showing all 4 clusters side-by-side."""
    k = len(clusters_data)
    cols = 8  # 8 images per row
    thumb_w, thumb_h = thumb_size

    row_header_h = 35
    row_h = row_header_h + thumb_h + 20
    master_w = cols * thumb_w + 20
    master_h = (k * row_h) + 60

    master_img = Image.new("RGB", (master_w, master_h), (15, 23, 42))
    draw = ImageDraw.Draw(master_img)

    # Master Header
    draw.rectangle([(0, 0), (master_w, 45)], fill=(30, 41, 59))
    draw.text((15, 12), f"SUB-TYPE AUDIT -- {cls_name.upper()} (All {k} Clusters)", fill=(56, 189, 248))

    for c_idx, c_info in enumerate(clusters_data):
        c_id = c_info["cluster_id"]
        c_size = c_info["size"]
        c_pct = c_info["percentage"]
        c_paths = c_info["representative_paths"][:cols]

        y_base = 50 + (c_idx * row_h)

        # Cluster Header bar
        draw.rectangle([(10, y_base), (master_w - 10, y_base + row_header_h - 5)], fill=(51, 65, 85))
        c_title = f"Cluster #{c_id}  |  Count: {c_size} images ({c_pct:.1f}% of class)"
        draw.text((20, y_base + 8), c_title, fill=(248, 250, 252))

        img_y = y_base + row_header_h
        for i_idx, p in enumerate(c_paths):
            img_x = 10 + (i_idx * thumb_w)
            try:
                with Image.open(p) as img:
                    img_conv = img.convert("RGB")
                    img_conv.thumbnail(thumb_size)
                    tile = Image.new("RGB", thumb_size, (15, 23, 42))
                    off_x = (thumb_w - img_conv.width) // 2
                    off_y = (thumb_h - img_conv.height) // 2
                    tile.paste(img_conv, (off_x, off_y))
                    master_img.paste(tile, (img_x, img_y))
            except Exception:
                continue

    out_png.parent.mkdir(parents=True, exist_ok=True)
    master_img.save(out_png, "PNG")


def generate_html_audit_sheet(
    cls_name: str,
    clusters_data: list[dict],
    out_html: Path,
) -> None:
    """Generates an interactive HTML contact sheet with separate sections per cluster."""
    cls_info = class_by_name(cls_name)
    display_name = cls_info.get("display", cls_name.replace("_", " "))

    sections_html = []
    for c in clusters_data:
        c_id = c["cluster_id"]
        c_size = c["size"]
        c_pct = c["percentage"]
        paths = c["representative_paths"]

        cards = []
        for p in paths:
            try:
                uri = p.resolve().as_uri()
                with Image.open(p) as img:
                    w, h = img.size
                card = f"""
                <div class="card">
                    <a href="{uri}" target="_blank">
                        <img src="{uri}" alt="{p.name}" loading="lazy"/>
                    </a>
                    <div class="info">
                        <span class="fname" title="{p.name}">{p.name}</span>
                        <span class="dim">{w}x{h} px</span>
                    </div>
                </div>
                """
                cards.append(card)
            except Exception:
                continue

        sec = f"""
        <div class="cluster-section">
            <div class="cluster-header">
                <h2>Cluster #{c_id}</h2>
                <span class="badge">{c_size} images ({c_pct:.1f}%)</span>
                <span class="subtext">Showing top {len(cards)} representative samples closest to cluster center</span>
            </div>
            <div class="grid">
                {"".join(cards)}
            </div>
        </div>
        """
        sections_html.append(sec)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sub-Type Audit -- {display_name} ({cls_name})</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0b1120; color: #f8fafc; margin: 0; padding: 24px; }}
        header {{ margin-bottom: 32px; border-bottom: 1px solid #1e293b; padding-bottom: 16px; }}
        h1 {{ color: #38bdf8; margin: 0 0 8px 0; font-size: 28px; }}
        p.subtitle {{ color: #94a3b8; margin: 0; font-size: 14px; }}
        .nav-links {{ margin-top: 16px; display: flex; gap: 12px; }}
        .nav-links a {{ color: #38bdf8; text-decoration: none; font-size: 13px; background: #1e293b; padding: 6px 12px; border-radius: 6px; border: 1px solid #334155; }}
        .nav-links a:hover {{ background: #334155; border-color: #38bdf8; }}
        .cluster-section {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-bottom: 32px; }}
        .cluster-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 20px; border-bottom: 1px solid #1e293b; padding-bottom: 12px; flex-wrap: wrap; }}
        .cluster-header h2 {{ margin: 0; color: #f1f5f9; font-size: 20px; }}
        .badge {{ background: #0284c7; color: white; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }}
        .subtext {{ color: #64748b; font-size: 13px; margin-left: auto; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }}
        .card {{ background: #1e293b; border-radius: 8px; overflow: hidden; border: 1px solid #334155; transition: transform 0.15s ease, border-color 0.15s ease; }}
        .card:hover {{ transform: translateY(-2px); border-color: #38bdf8; }}
        .card img {{ width: 100%; height: 170px; object-fit: cover; display: block; background: #0f172a; }}
        .info {{ padding: 10px; font-size: 11px; }}
        .fname {{ display: block; color: #e2e8f0; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .dim {{ color: #64748b; margin-top: 4px; display: block; }}
    </style>
</head>
<body>
    <header>
        <h1>Sub-Type Audit: {display_name}</h1>
        <p class="subtitle">Class Slug: <code>{cls_name}</code> | K-Means Clustering (k={len(clusters_data)}) | OpenCLIP ViT-B-32 Embeddings</p>
        <div class="nav-links">
            <a href="index.html">&larr; Back to Master Audit Index</a>
            <a href="{cls_name}_subtypes_grid.png" target="_blank">&#128444; View Full PNG Grid</a>
        </div>
    </header>

    {"".join(sections_html)}
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_content, encoding="utf-8")


def generate_master_index_html(audit_results: list[dict], out_html: Path) -> None:
    """Generates a master index HTML page linking all 4 audited classes."""
    cards_html = []
    for res in audit_results:
        cls_name = res["class"]
        display_name = res["display_name"]
        total = res["total_images"]
        k = res["k"]
        clusters = res["clusters"]
        sil = res.get("silhouette_score", 0.0)

        cluster_badges = "".join(
            [f"<div class='c-item'><strong>Cluster #{c['cluster_id']}:</strong> {c['size']} images ({c['percentage']:.1f}%)</div>" for c in clusters]
        )

        card = f"""
        <div class="class-card">
            <h2>{display_name} (<code>{cls_name}</code>)</h2>
            <div class="stats">
                <span><strong>Total Images:</strong> {total}</span>
                <span><strong>Clusters (k):</strong> {k}</span>
                <span><strong>Silhouette Score:</strong> {sil:.3f}</span>
            </div>
            <div class="cluster-list">
                {cluster_badges}
            </div>
            <div class="card-links">
                <a class="btn primary" href="{cls_name}.html">Open Interactive HTML Sheet</a>
                <a class="btn secondary" href="{cls_name}_subtypes_grid.png" target="_blank">View Composite PNG Grid</a>
            </div>
        </div>
        """
        cards_html.append(card)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sub-Type Audit Dashboard -- 4 Weak Classes</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0b1120; color: #f8fafc; margin: 0; padding: 32px; }}
        header {{ margin-bottom: 32px; border-bottom: 1px solid #1e293b; padding-bottom: 20px; }}
        h1 {{ color: #38bdf8; margin: 0 0 8px 0; font-size: 32px; }}
        p.desc {{ color: #94a3b8; font-size: 15px; margin: 0; max-width: 800px; line-height: 1.5; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 24px; }}
        .class-card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 24px; display: flex; flex-direction: column; }}
        .class-card h2 {{ margin: 0 0 16px 0; color: #f1f5f9; font-size: 20px; }}
        .stats {{ display: flex; gap: 16px; font-size: 13px; color: #94a3b8; margin-bottom: 16px; background: #1e293b; padding: 8px 12px; border-radius: 6px; }}
        .stats span strong {{ color: #e2e8f0; }}
        .cluster-list {{ background: #1e293b; border-radius: 6px; padding: 12px; margin-bottom: 20px; font-size: 13px; display: flex; flex-direction: column; gap: 6px; }}
        .c-item {{ color: #cbd5e1; }}
        .c-item strong {{ color: #38bdf8; }}
        .card-links {{ margin-top: auto; display: flex; gap: 12px; }}
        .btn {{ text-decoration: none; font-size: 13px; font-weight: 600; padding: 8px 14px; border-radius: 6px; text-align: center; flex: 1; }}
        .btn.primary {{ background: #0284c7; color: white; }}
        .btn.primary:hover {{ background: #0369a1; }}
        .btn.secondary {{ background: #1e293b; color: #cbd5e1; border: 1px solid #334155; }}
        .btn.secondary:hover {{ background: #334155; color: white; }}
    </style>
</head>
<body>
    <header>
        <h1>Visual Sub-Type Audit Dashboard</h1>
        <p class="desc">Diagnostic K-Means clustering (k=4) on OpenCLIP ViT-B-32 embeddings for the 4 weak-performing classes (<code>cnc_milling</code>, <code>conveyor</code>, <code>packaging_machine</code>, <code>grinding</code>) to evaluate intra-class sub-type variance and visual distribution balance.</p>
    </header>

    <div class="grid">
        {"".join(cards_html)}
    </div>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_content, encoding="utf-8")


def audit_class(
    cls_name: str,
    model,
    preprocess,
    device,
    k: int = DEFAULT_K,
    samples_per_cluster: int = SAMPLES_PER_CLUSTER,
) -> dict:
    """Runs k-means clustering and generates contact sheets for a single class."""
    class_dir = DATASET_TRAIN_DIR / cls_name
    cls_info = class_by_name(cls_name)
    display_name = cls_info.get("display", cls_name.replace("_", " "))

    if not class_dir.exists():
        print(f"[WARN] Class directory {class_dir} does not exist. Skipping.")
        return {"class": cls_name, "display_name": display_name, "total_images": 0, "clusters": []}

    image_paths = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS])
    total_imgs = len(image_paths)

    if total_imgs < k:
        print(f"[WARN] Not enough images in {cls_name} ({total_imgs}) for k={k}. Skipping.")
        return {"class": cls_name, "display_name": display_name, "total_images": total_imgs, "clusters": []}

    print(f"\n[{cls_name.upper()}] Extracting OpenCLIP embeddings for {total_imgs} images...")
    valid_paths, embs = get_class_image_embeddings(image_paths, model, preprocess, device)

    print(f"[{cls_name.upper()}] Running K-Means (k={k})...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embs)
    centroids = kmeans.cluster_centers_

    # Calculate silhouette score
    sil_score = float(silhouette_score(embs, labels))

    clusters_data = []
    for c_id in range(k):
        c_indices = np.where(labels == c_id)[0]
        c_size = len(c_indices)
        c_pct = (c_size / total_imgs) * 100.0

        if c_size == 0:
            clusters_data.append({
                "cluster_id": c_id,
                "size": 0,
                "percentage": 0.0,
                "representative_paths": [],
                "png_grid_path": "",
            })
            continue

        c_embs = embs[c_indices]
        centroid = centroids[c_id]

        # Compute Euclidean distance from each image in cluster to centroid
        dists = np.linalg.norm(c_embs - centroid, axis=1)
        sorted_order = np.argsort(dists)

        # Take top representative images closest to centroid
        rep_indices = [c_indices[i] for i in sorted_order[:samples_per_cluster]]
        rep_paths = [valid_paths[i] for i in rep_indices]

        # Generate individual cluster PNG grid
        out_cluster_png = AUDIT_DIR / f"{cls_name}_cluster_{c_id}_grid.png"
        generate_cluster_png_grid(cls_name, c_id, rep_paths, out_cluster_png)

        clusters_data.append({
            "cluster_id": c_id,
            "size": c_size,
            "percentage": round(c_pct, 2),
            "representative_paths": rep_paths,
            "representative_filenames": [p.name for p in rep_paths],
            "png_grid_path": str(out_cluster_png),
        })

    # Generate master composite PNG for class (all 4 clusters)
    out_composite_png = AUDIT_DIR / f"{cls_name}_subtypes_grid.png"
    generate_class_composite_grid(cls_name, clusters_data, out_composite_png)

    # Generate interactive HTML sheet for class
    out_html = AUDIT_DIR / f"{cls_name}.html"
    generate_html_audit_sheet(cls_name, clusters_data, out_html)

    return {
        "class": cls_name,
        "display_name": display_name,
        "total_images": total_imgs,
        "k": k,
        "silhouette_score": round(sil_score, 4),
        "html_sheet_path": str(out_html),
        "composite_png_path": str(out_composite_png),
        "clusters": clusters_data,
    }


def main():
    print("=" * 80)
    print("DIAGNOSTIC VISUAL SUB-TYPE AUDIT (K-MEANS k=4)")
    print(f"Target Classes (4): {TARGET_CLASSES}")
    print("=" * 80)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    model, preprocess, device = load_clip_model()

    audit_results = []
    for cls_name in TARGET_CLASSES:
        res = audit_class(cls_name, model, preprocess, device, k=DEFAULT_K, samples_per_cluster=SAMPLES_PER_CLUSTER)
        audit_results.append(res)

    # Generate Master Index HTML
    master_index_html = AUDIT_DIR / "index.html"
    generate_master_index_html(audit_results, master_index_html)

    # Save full JSON report
    report_json_path = LOGS_DIR / "sub_type_audit_report.json"
    serializable_results = []
    for r in audit_results:
        c_copy = []
        for c in r["clusters"]:
            c_copy.append({
                "cluster_id": c["cluster_id"],
                "size": c["size"],
                "percentage": c["percentage"],
                "representative_filenames": c.get("representative_filenames", []),
                "png_grid_path": c["png_grid_path"],
            })
        serializable_results.append({
            "class": r["class"],
            "display_name": r["display_name"],
            "total_images": r["total_images"],
            "k": r["k"],
            "silhouette_score": r["silhouette_score"],
            "html_sheet_path": r["html_sheet_path"],
            "composite_png_path": r["composite_png_path"],
            "clusters": c_copy,
        })

    report_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "clip_model": CLIP_MODEL_NAME,
        "clip_pretrained": CLIP_PRETRAINED,
        "k_clusters": DEFAULT_K,
        "samples_per_cluster": SAMPLES_PER_CLUSTER,
        "master_dashboard_html": str(master_index_html),
        "results": serializable_results,
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    print("\n" + "=" * 85)
    print("SUB-TYPE AUDIT SUMMARY: CLUSTER SIZE BREAKDOWN (k=4)")
    print("=" * 85)
    print(f"  {'Class':<20} {'Total':>7}  {'Cluster 0':>12}  {'Cluster 1':>12}  {'Cluster 2':>12}  {'Cluster 3':>12}  {'Imbalance':<10}")
    print("  " + "-" * 80)

    for r in audit_results:
        cls_name = r["class"]
        total = r["total_images"]
        c_sizes = [c["size"] for c in r["clusters"]]
        c_pcts  = [c["percentage"] for c in r["clusters"]]

        # Calculate imbalance ratio: max cluster / min cluster
        non_zero = [s for s in c_sizes if s > 0]
        imbalance_ratio = (max(non_zero) / min(non_zero)) if non_zero else 1.0
        imbal_label = "⚠️ HIGH" if imbalance_ratio > 3.0 else "BALANCED"

        c0_str = f"{c_sizes[0]} ({c_pcts[0]:.0f}%)"
        c1_str = f"{c_sizes[1]} ({c_pcts[1]:.0f}%)"
        c2_str = f"{c_sizes[2]} ({c_pcts[2]:.0f}%)"
        c3_str = f"{c_sizes[3]} ({c_pcts[3]:.0f}%)"

        print(f"  {cls_name:<20} {total:>7}  {c0_str:>12}  {c1_str:>12}  {c2_str:>12}  {c3_str:>12}  {imbal_label:<10}")

    print("=" * 85)
    print(f"\n[COMPLETE] Report saved to: {report_json_path}")
    print(f"[COMPLETE] Interactive Dashboard: {master_index_html}\n")


if __name__ == "__main__":
    main()
