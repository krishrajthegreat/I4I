# PRD: Machine Type Classification — Dataset Engine & Model

## 1. Overview

This project builds an automated dataset generation pipeline for a **Machine Type image classifier**, plus the classifier itself. The system takes a list of industrial and woodworking machine classes, scrapes candidate images from the web, filters them for semantic and visual relevance using CLIP, and outputs a clean, labeled, PyTorch-ready dataset. That dataset is labeled/refined in Roboflow and used to train a classification model intended for eventual deployment against real factory CCTV-style footage.

This is a real build project, not a prototype or hackathon deliverable.

### 1.1 Scope

**In scope (this phase):**
- Machine Type classification only.

**Explicitly deferred (not in this phase):**
- Machine State classification (Running / Idle / Maintenance / Fault / Emergency Stop / Powered Off) — deferred because state is often not visually determinable from a single static image; needs a separate design decision (single-frame vs. video/sequence vs. multi-modal input) before it's workable.
- Environment/zone classification — class list not yet defined.
- Image-licensing review for scraped data — deferred for now; to be revisited before any external release or deployment.

---

## 2. Machine Type — Class List

**Core industrial machines (11):**
1. CNC Milling
2. Lathe
3. Milling
4. Drilling
5. Grinding
6. Hydraulic Press
7. Injection Molding
8. Conveyor
9. Packaging Machine
10. Forklift
11. Control Panel

**Cutting machines (4):**
12. Table saw
13. Band saw
14. Miter saw
15. Panel saw

**Surfacing and shaping (4):**
16. Planer (Thicknesser)
17. Jointer
18. Wood router / Spindle moulder
19. Wood lathe

**Boring and finishing (3):**
20. Drill press
21. Sanding machines
22. CNC router

**Auxiliary machines:**
- Not yet defined. To be finalized before dataset generation begins for this group.

**Total defined so far: 21 classes** (plus any auxiliary machines added later).

### 2.1 Known confusable clusters

These groups are visually similar enough that text-based filtering alone is expected to be unreliable between them, and are the priority candidates for image-to-image (fine) filtering:
- Cutting machines: table saw / band saw / miter saw / panel saw
- Surfacing and shaping: planer / jointer / spindle moulder / wood lathe
- CNC milling / milling / drilling / CNC router (functional and visual overlap)

---

## 3. Dataset Generation Pipeline

```
Class list
   ↓
Query expansion (per class, per variation axis)
   ↓
Web scraper (Bing / Google Images)
   ↓
Cleaning (dedup via hashing, resolution/format checks, corrupt file removal)
   ↓
Hybrid CLIP filtering:
   1. Text-based CLIP filtering (coarse) — discards obviously irrelevant content
   2. Image-based CLIP filtering (fine) — for confusable clusters only, scored
      against hand-picked reference images per class
   ↓
Diversity dedup (embedding-based near-duplicate removal)
   ↓
K-selection (score-based, per class stopping rule)
   ↓
PyTorch-ready dataset folder
   ↓
Roboflow (labeling refinement / training)
```

### 3.1 Query expansion by variation axis

Rather than paraphrasing a single query per class, each class is expanded across explicit variation-axis templates to intentionally source the visual diversity the model needs, rather than relying on filtering to produce it after the fact:

```
templates = [
    "{machine} factory floor",
    "{machine} close-up",
    "{machine} CCTV surveillance distance",
    "{machine} with worker operating",
    "{machine} partially obscured industrial"
]
```

This covers: viewing angle, distance (close-up vs. far/CCTV-style), lighting variation, worker presence/absence, and occlusion — without a fixed per-axis image quota. This step affects the scraping/data-collection stage only; it does not add complexity to the model itself.

### 3.2 Hybrid CLIP filtering

- **Text-based CLIP filtering (coarse):** scores each image against its class text prompt. High recall, low precision — designed to remove clearly irrelevant results (wrong object, no machine in frame, unrelated content), not to distinguish between similar machine types.
- **Image-based CLIP filtering (fine):** applied only to classes in a known confusable cluster (Section 2.1). Each candidate image is scored by cosine similarity against a small set of hand-picked, verified reference images per class (image-to-image, not text-to-image), which is significantly more discriminating for visually similar categories.
- Classes with no visual confusion risk skip the fine-filtering step to save compute.

**Threshold calibration:** no fixed absolute similarity threshold is assumed in advance (raw CLIP similarity scores are typically much lower than an intuitive 0.7–0.8 range). Thresholds are calibrated empirically per filtering stage using a small manually-labeled sample before being applied at scale.

### 3.3 Diversity / near-duplicate removal

Before final K-selection, candidate images are clustered by embedding similarity (or filtered greedily above a similarity threshold) to remove near-duplicates — e.g., multiple crops or re-uploads of the same stock photo — protecting effective dataset diversity, which matters more at small-to-moderate per-class counts.

### 3.4 K-selection & stopping rule

No fixed per-class image cap. Per class:

- **Minimum:** 80–150 images
- **Target:** higher for classes in a confusable cluster
- **Stop condition:** similarity score falls below the calibrated threshold **AND** acceptance rate (proportion of new candidates passing filtering) drops significantly

Classes that cannot reach the minimum from scraping alone (expected for narrow search terms like "spindle moulder" or "panel saw") are flagged for a lower accepted floor or an alternate sourcing plan, rather than silently under-filled.

### 3.5 Manual review

A fast manual pass (contact-sheet style review of the top-K per class) is performed after K-selection and before finalizing the dataset, to catch filtering mistakes that automated scoring misses — cheap relative to the cost of training on a contaminated class.

---

## 4. Real-World Validation Set (Domain Gap Check)

**Problem:** training data is scraped web/stock photography — generally clean, well-lit, catalog-style. Deployment target is CCTV-style factory footage — elevated angle, wider shot, variable lighting, possible occlusion. A model that performs well on held-out scraped data may still fail on real footage; this gap needs to be measured, not assumed.

**Approach:**
- Collect a small set (~5–10 images per class, more where feasible) of **real, non-scraped** images per class — e.g. phone photos of accessible machines (college workshop), existing real-world datasets (Roboflow Universe), or frames extracted from real factory/workshop-tour video footage.
- Store these completely separately from the scraped training data:

```
dataset/
  train/                  ← scraped, CLIP-filtered images
    lathe/
    cnc_milling/
    ...
  real_validation/        ← real, non-scraped images
    lathe/
    cnc_milling/
    ...
```

- Label by placement into class subfolders at capture time; optionally track an accompanying CSV (`filename, class, angle, distance, lighting_notes`) to support later failure diagnosis by condition.
- **Validation happens in two stages:**
  1. **Pipeline sanity check** — score these real images through the CLIP filtering pipeline itself (pre-training) to confirm they rank correctly against their true class. A failure here indicates a problem with reference images or prompts, catchable before training.
  2. **Post-training test** — after training on scraped data only, run inference on `real_validation/` and compare per-class accuracy and confusion matrix against a held-out scraped-data test set. The gap between the two is the measured domain gap.
- **Caveat:** at 5–10 images per class this set is a canary, not a statistically rigorous benchmark. Treat a clearly broken class (e.g. 0/8 correct) as a signal to source more real images for that class, not as a final verdict.

---

## 5. Output Format

```
dataset/
  train/
    <class_name>/
      img_001.jpg
      img_002.jpg
      ...
  real_validation/
    <class_name>/
      ...
```

PyTorch-compatible dataset structure (loadable via a standard `torch.utils.data.Dataset` / `ImageFolder`-style loader), and directly importable into Roboflow for labeling refinement and downstream training.

---

## 6. Tech Stack

**Must use:**
- Python
- CLIP (OpenAI CLIP or OpenCLIP)
- requests / aiohttp (scraping)
- PIL / OpenCV (image handling)
- Roboflow (labeling refinement, training)

**Optional / as needed:**
- Perceptual hashing library (dedup)
- Embedding clustering (diversity filtering)
- Simple blur/quality detection

---

## 7. Risks & Open Items

| Risk | Notes |
|---|---|
| Domain gap between scraped training data and real CCTV deployment footage | Mitigated via real-world validation set (Section 4); not fully solved, only measured |
| Class imbalance across the 22 defined classes | Common/easy-to-source classes (Forklift, Conveyor, Control Panel) vs. narrow/hard-to-source classes (Spindle moulder, Panel saw, Sanding machines) — needs a per-class floor or alternate sourcing decision |
| Auxiliary Machines subcategory undefined | Needs to be finalized before dataset generation covers it |
| Image licensing | Explicitly deferred for this phase; must be revisited before any external release or deployment |
| CLIP threshold miscalibration | Mitigated by empirical calibration rather than assumed fixed thresholds (Section 3.2) |

---

## 8. Out of Scope (This Phase)

- Machine State model
- Environment/zone model
- Multi-head vs. multi-model architecture decision (relevant only once State/Environment are back in scope)
- Image licensing resolution
