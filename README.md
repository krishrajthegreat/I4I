# Machine Type Classifier — Dataset & Instance Segmentation Pipeline

Automated dataset generation, verification, and auto-annotation pipeline for a 23-class industrial/woodworking Machine Type instance segmentation model.

## Quick Start

```powershell
# 1. Create and activate venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch with CUDA support (check your GPU / CUDA driver first: nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Pipeline Architecture & Workflow

The pipeline follows a 6-stage progression: **Scrape → Clean → Filter → Diversify → Auto-Label → Train**.

| Stage | Script | Description |
|---|---|---|
| 1a | `pipeline/utils/manifest_gen.py` | Scan flat ref-image folder → blank manifest CSV |
| 1b | `pipeline/stage1_ingest_refs.py` | Ingest filled manifest → `reference_images/<class>/` |
| 2  | `pipeline/stage2_scrape.py` | Multi-engine web scraper (5 query templates per class) |
| 3  | `pipeline/stage3_clean.py` | Dedup, corrupt-file removal, resolution filter |
| 4  | `pipeline/stage4_clip_filter.py` | Hybrid OpenCLIP filter (coarse text + fine image similarity) |
| 4b | `pipeline/stage4b_calibrate.py` | Calibrate OpenCLIP thresholds from labeled sample |
| 5  | `pipeline/stage5_diversity_kselect.py` | Diversity dedup + K-selection with stopping rule |
| 6  | `pipeline/upload_polygon_annotations.py` | **Auto-Label & Ingest** — Gemini Vision class verification + SAM2 polygon contour extraction to Roboflow |

---

## Reference Image Ingestion Workflow

```powershell
# Step 1: Generate blank manifest (point at flat folder of reference images)
python -m pipeline.utils.manifest_gen --src C:\path\to\your\ref_images

# Step 2: Open reference_manifest.csv and fill in the 'class' column
#         Use exact slugs from pipeline/config.py (e.g. band_saw, table_saw)

# Step 3: Ingest
python -m pipeline.stage1_ingest_refs
```

---

## Output Layout

```
dataset/
  train/
    <class_name>/        ← Final cleaned images per class
reference_images/
  <class_name>/          ← Hand-picked reference images (confusable clusters, used in Stage 4)
logs/                    ← Per-stage JSON logs and annotation audit outputs
```

---

## Supported Machine Classes (23 Classes)

| Class Slug | Display Name | Confusable Cluster |
|---|---|---|
| `band_saw` | Band Saw | cutting |
| `cnc_milling` | CNC Milling | cnc_group |
| `cnc_router` | CNC Router | cnc_group |
| `control_panel` | Control Panel | — |
| `conveyor` | Conveyor | — |
| `crane` | Industrial Crane | — |
| `drilling` | Drilling | cnc_group |
| `fire_extinguisher` | Fire Extinguisher | — |
| `forklift` | Forklift | — |
| `grinding` | Grinding | — |
| `hydraulic_press` | Hydraulic Press | — |
| `injection_molding` | Injection Molding | — |
| `jointer` | Jointer | surfacing |
| `lathe` | Lathe | — |
| `milling` | Milling | cnc_group |
| `miter_saw` | Miter Saw | cutting |
| `packaging_machine` | Packaging Machine | — |
| `panel_saw` | Panel Saw | cutting |
| `planer` | Planer (Thicknesser) | surfacing |
| `sanding_machines` | Sanding Machines | — |
| `spindle_moulder` | Wood Router / Spindle Moulder | surfacing |
| `table_saw` | Table Saw | cutting |
| `wood_lathe` | Wood Lathe | surfacing |

---

## Model Training & Validation Results

Initial validation metrics from the YOLO instance-segmentation training run (`train-2/`, 97 epochs on GPU):

| Metric Group | Metric | Score |
| :--- | :--- | :--- |
| **Bounding Box** | Precision (B) | 87.6% |
| | Recall (B) | 77.1% |
| | mAP@50 (B) | 83.1% |
| | mAP@50-95 (B) | 74.8% |
| **Polygon Mask** | Precision (M) | 86.9% |
| | Recall (M) | 75.9% |
| | mAP@50 (M) | 81.1% |
| | mAP@50-95 (M) | 70.5% |
