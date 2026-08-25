# Machine Type Classifier — Dataset & Segmentation Pipeline

An automated computer vision pipeline that scrapes, cleans, filters, and annotates industrial and woodworking machine images to train an instance-segmentation model capable of identifying and outlining machine types in factory and workshop environments.

---

## What It's Currently Doing

- **Pipeline Stages 1–6 Operational**: Automated ingestion, DuckDuckGo web scraping, resolution & pHash deduplication, hybrid OpenCLIP semantic filtering (text prompt + image-to-image reference comparisons), and greedy diversity K-selection.
- **21 Machine Classes Covered**: Spans core industrial equipment (`cnc_milling`, `lathe`, `drilling`, `hydraulic_press`, `injection_molding`, `forklift`, etc.) and woodworking machinery (`table_saw`, `band_saw`, `panel_saw`, `planer`, `jointer`, `cnc_router`, etc.).
- **Automated Cloud Annotation**: Integrated Roboflow serverless workflow (`gemini-machine-instance-auto-label-2`) using Google Gemini Vision to verify class semantics and Segment Anything Model (SAM) to generate pixel-precise COCO polygon segmentations.
- **Model Training Status**: YOLO instance-segmentation model trained and evaluated on cloud GPU (results recorded under `train-2/`); full 21-class scaling and real-world domain benchmarking in progress.

---

## Requirements

### Software & Key Dependencies
- **Python**: 3.10+ (tested on Python 3.13)
- **OpenCLIP** (`open-clip-torch`): Semantic image-text and image-image filtering
- **DuckDuckGo Search** (`duckduckgo-search`): Multi-query image scraping
- **Pillow & ImageHash** (`Pillow`, `ImageHash`): Image validation, format normalization, and perceptual hash deduplication
- **PyTorch** (`torch`, `torchvision`): Tensor operations and local embedding inference
- **Roboflow SDK** (`roboflow`, `python-dotenv`): Cloud dataset synchronization and auto-label workflow execution

### Authentication & Keys
- `ROBOFLOW_API_KEY`: Required in `.env` for dataset upload and workflow annotation calls.
- Web scraper requires no API keys.

### Hardware Expectations
- **Local Machine (CPU or CUDA GPU)**: Handles scraping, cleaning, CLIP filtering, and dispatching cloud annotation workflows.
- **Cloud GPU (Google Colab T4 / A100)**: Used for training the YOLO instance segmentation model.

---

## How to Run It

### 1. Setup Environment
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install pipeline dependencies
pip install -r requirements.txt

# (Optional) Install PyTorch with CUDA for local GPU acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Interactive End-to-End Runner
Run the interactive CLI that walks through class selection, scraping, cleaning, and filtering:
```powershell
python pipeline/run.py
```

### 3. Running Individual Pipeline Stages
```powershell
# Stage 1: Ingest reference images from manifest
python -m pipeline.stage1_ingest_refs

# Stage 2: Scrape images via DuckDuckGo (requires --classes)
python -m pipeline.stage2_scrape --classes lathe band_saw

# Stage 3: Clean low-res (<150px) and near-duplicate images
python -m pipeline.stage3_clean --classes lathe

# Stage 4: CLIP semantic filter (coarse text prompt + fine reference filter)
python -m pipeline.stage4_clip_filter --classes lathe

# Stage 4b: Calibrate CLIP thresholds using labeled ground truth
python -m pipeline.stage4b_calibrate --class lathe --good good_bad/good/lathe --bad good_bad/bad/lathe

# Stage 5: Diversity deduplication and K-selection quota enforcement
python -m pipeline.stage5_diversity_kselect --classes lathe

# Stage 6: Generate HTML & composite grid contact sheets for visual inspection
python -m pipeline.stage6_contact_sheet --classes lathe
```

### 4. Fast Parallel Top-Up & Annotation
```powershell
# Bring class image counts to 300+ with parallel multi-threaded DDGS scraping
python pipeline/batch_topup_300.py

# Run Gemini verification + SAM polygon auto-annotation upload to Roboflow
python pipeline/upload_polygon_annotations.py --classes lathe band_saw
```

### 5. Model Training (Google Colab)
1. Export the annotated dataset version from Roboflow in **YOLOv8/YOLO26 Segmentation** format.
2. Train in Colab using the YOLO CLI:
   ```bash
   yolo task=segment mode=train model=yolo26m-seg.pt data={dataset.location}/data.yaml epochs=100 patience=20 imgsz=640
   ```

---

## Architecture & Design Choices

The pipeline follows a staged progression: **Scrape → Clean → Filter → Annotate → Train**.

- **Hybrid CLIP Filtering**: Scraped images often contain noisy stock photos or incorrect models. Combining zero-shot text classification with image-to-image reference cosine similarity disambiguates visually similar machines (e.g. CNC mills vs manual mills) without manual sorting. *(See [OpenCLIP](https://github.com/mlfoundations/open_clip)).*
- **Two-Step Verification & Segmentation (Gemini + SAM)**: SAM alone segments objects without semantic validation. Running Gemini Vision as an upfront classification gate ensures the machine type is correct before SAM extracts pixel-accurate 2D polygon boundaries. *(See [Segment Anything](https://segment-anything.com/)).*

---

## Objective Tracking / TODO

- [x] 21-class web scraping, cleaning, and pHash deduplication pipeline
- [x] OpenCLIP coarse & fine semantic filtering with threshold calibration
- [x] Roboflow serverless workflow with Gemini Vision gating and SAM polygon generation
- [x] Initial YOLO instance-segmentation training & validation (`train-2/`)
- [ ] Scaled 21-class model training and hyperparameter tuning on Colab GPU
- [ ] Collection of real-world factory CCTV & mobile validation benchmark set
- [ ] *Deferred*: Machine State classification (Running / Idle / Fault) and Zone/Environment models

---

## Performance Notes

Validation metrics from the completed YOLO instance-segmentation training run (`train-2/`, 97 epochs on Colab GPU):

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
