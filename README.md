# Machine Type Classifier — Dataset Pipeline

Automated dataset generation pipeline for a 21-class industrial/woodworking
Machine Type image classifier. **6-stage pipeline** — Stage 6 contact sheet is
the final deliverable per class, feeding directly into Roboflow.

## Quick start

```powershell
# 1. Create and activate venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PyTorch with CUDA (check your CUDA version first: nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Pipeline stages

| Stage | Script | Description |
|---|---|---|
| 1a | `pipeline/utils/manifest_gen.py` | Scan flat ref-image folder → blank manifest CSV |
| 1b | `pipeline/stage1_ingest_refs.py` | Ingest filled manifest → `reference_images/<class>/` |
| 2  | `pipeline/stage2_scrape.py` | DuckDuckGo scraper (5 query templates × 21 classes) |
| 3  | `pipeline/stage3_clean.py` | Dedup, corrupt-file removal, resolution filter |
| 4  | `pipeline/stage4_clip_filter.py` | Hybrid CLIP filter (coarse text + fine image) |
| 4b | `pipeline/stage4b_calibrate.py` | Calibrate CLIP thresholds from labeled sample |
| 5  | `pipeline/stage5_diversity_kselect.py` | Diversity dedup + K-selection with stopping rule |
| 6  | `pipeline/stage6_contact_sheet.py` | **FINAL** — Contact sheets (HTML + PNG) for manual review → Roboflow |

---

## Reference images — Option C workflow

Your reference images have arbitrary filenames. Workflow:

```powershell
# Step 1: generate blank manifest (point at your flat folder of 81 images)
python -m pipeline.utils.manifest_gen --src C:\path\to\your\ref_images

# Step 2: open reference_manifest.csv and fill in the 'class' column
#         Use exact slugs from pipeline/config.py (e.g. band_saw, table_saw)

# Step 3: ingest
python -m pipeline.stage1_ingest_refs

# Optional dry-run first:
python -m pipeline.stage1_ingest_refs --dry-run
```

---

## Output layout

```
dataset/
  train/
    <class_name>/        ← final Stage 5 images (feeds into Roboflow via Stage 6 review)
reference_images/
  <class_name>/          ← hand-picked reference images (confusable clusters, used in Stage 4)
logs/                    ← per-stage JSON logs
```

---

## Class list (21 classes)

| Slug | Display | Confusable cluster |
|---|---|---|
| cnc_milling | CNC Machine | cnc_group |
| lathe | Lathe | — |
| milling | Milling | cnc_group |
| drilling | Drilling | cnc_group |
| grinding | Grinding | — |
| hydraulic_press | Hydraulic Press | — |
| injection_molding | Injection Molding | — |
| conveyor | Conveyor | — |
| packaging_machine | Packaging Machine | — |
| forklift | Forklift | — |
| control_panel | Control Panel | — |
| table_saw | Table Saw | cutting |
| band_saw | Band Saw | cutting |
| miter_saw | Miter Saw | cutting |
| panel_saw | Panel Saw | cutting |
| planer | Planer (Thicknesser) | surfacing |
| jointer | Jointer | surfacing |
| spindle_moulder | Wood Router / Spindle Moulder | surfacing |
| wood_lathe | Wood Lathe | surfacing |
| sanding_machines | Sanding Machines | — |
| cnc_router | CNC Router | cnc_group |
