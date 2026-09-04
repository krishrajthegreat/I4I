"""
pipeline/config.py
==================
Central configuration for the Machine Type Classifier dataset pipeline.
All pipeline scripts import from here — edit this file to change behaviour
across the whole pipeline without hunting through individual scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root (one level above this file)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Directory layout (matches PRD Section 5)
# ---------------------------------------------------------------------------
DATASET_TRAIN_DIR       = ROOT / "dataset" / "train"
REFERENCE_IMAGES_DIR    = ROOT / "reference_images"
LOGS_DIR                = ROOT / "logs"

# ---------------------------------------------------------------------------
# Scraper settings
# ---------------------------------------------------------------------------
# Maximum raw images to scrape per class across ALL query templates combined.
SCRAPE_CAP_PER_CLASS: int = 400

# Minimum seconds to sleep between consecutive DuckDuckGo requests (per query).
# Increase if you hit rate-limit errors frequently.
DDGS_SLEEP_BETWEEN_QUERIES: float = 1.5

# Maximum retries on a rate-limit hit before giving up on a single query.
DDGS_MAX_RETRIES: int = 5

# Download timeout (seconds) per image URL.
DOWNLOAD_TIMEOUT: int = 10

# ---------------------------------------------------------------------------
# Cleaning settings (Stage 3)
# ---------------------------------------------------------------------------
# Images below this pixel dimension (either axis) are discarded.
MIN_RESOLUTION: int = 150  # px

# pHash Hamming-distance threshold for near-duplicate detection.
PHASH_DUPLICATE_THRESHOLD: int = 8

# ---------------------------------------------------------------------------
# CLIP model settings (Stage 4)
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME: str = "ViT-B-32"
CLIP_PRETRAINED: str = "laion2b_s34b_b79k"   # OpenCLIP weights

# Batch size for CLIP inference (tune down if you hit VRAM limits).
CLIP_BATCH_SIZE: int = 64

# ---------------------------------------------------------------------------
# K-selection settings (Stage 5 / PRD Section 3.4)
# ---------------------------------------------------------------------------
# Hard minimum floor — differs by class type:
#   Standard classes    : 80  (hard pass/fail floor)
#   Confusable-cluster  : 150 (hard pass/fail floor — tighter due to inter-class confusion risk)
# Both values are enforced as FLAGGED if not met. K_TARGET_CONFUSABLE kept as alias.
K_MIN_FLOOR: int = 80
K_MIN_STANDARD: int = 80
K_MIN_CONFUSABLE: int = 150     # hard floor for confusable-cluster classes
K_TARGET_CONFUSABLE: int = 150  # alias (same value; retained for backwards compatibility)

# Greedy diversity filter: images with cosine similarity > this to the
# already-selected set are considered near-duplicates and skipped.
DIVERSITY_SIM_THRESHOLD: float = 0.95

# ---------------------------------------------------------------------------
# Reference images: minimum per confusable class to enable fine-filtering.
# Classes below this count are flagged (not failed) at Stage 1 report time.
# ---------------------------------------------------------------------------
MIN_REFS_PER_CONFUSABLE_CLASS: int = 3

# ---------------------------------------------------------------------------
# Class list (21 classes, PRD Section 2 + Section 2.1 confusable clusters)
# ---------------------------------------------------------------------------
# Each entry:
#   name        : filesystem-safe slug used for folder names everywhere
#   display     : human-readable label used in CLIP text prompts
#   confusable  : True → participates in fine (image-to-image) CLIP filtering
#   cluster     : cluster group name (for reporting and grouping only)
#
# Confusable clusters per PRD Section 2.1:
#   "cnc_group"  : cnc_milling / milling / drilling / cnc_router
#   "cutting"    : table_saw / band_saw / miter_saw / panel_saw
#   "surfacing"  : planer / jointer / spindle_moulder / wood_lathe

CLASSES: list[dict] = [
    # ── Core industrial ──────────────────────────────────────────────────────
    {
        "name": "cnc_milling",
        "display": "CNC milling machine",
        "confusable": True,
        "cluster": "cnc_group",
    },
    {
        "name": "lathe",
        "display": "lathe machine",
        "confusable": False,
        "cluster": None,
    },
    {
        "name": "milling",
        "display": "milling machine",
        "confusable": True,
        "cluster": "cnc_group",
    },
    {
        "name": "drilling",
        "display": "industrial drilling machine",
        "confusable": True,
        "cluster": "cnc_group",
    },
    {
        "name": "grinding",
        "display": "grinding machine",
        "confusable": False,
        "cluster": None,
    },
    {
        "name": "hydraulic_press",
        "display": "hydraulic press",
        "confusable": False,
        "cluster": None,
    },
    {
        "name": "injection_molding",
        "display": "injection molding machine",
        "confusable": False,
        "cluster": None,
    },
    {
        "name": "conveyor",
        "display": "conveyor belt",
        "confusable": False,
        "cluster": None,
    },
    {
        "name": "packaging_machine",
        "display": "packaging machine",
        "confusable": False,
        "cluster": None,
    },
    {
        "name": "forklift",
        "display": "forklift truck",
        "confusable": False,
        "cluster": None,
    },
    {
        "name": "control_panel",
        "display": "industrial control panel",
        "confusable": False,
        "cluster": None,
    },
    # ── Cutting machines ─────────────────────────────────────────────────────
    {
        "name": "table_saw",
        "display": "table saw",
        "confusable": True,
        "cluster": "cutting",
    },
    {
        "name": "band_saw",
        "display": "band saw",
        "confusable": True,
        "cluster": "cutting",
    },
    {
        "name": "miter_saw",
        "display": "miter saw",
        "confusable": True,
        "cluster": "cutting",
    },
    {
        "name": "panel_saw",
        "display": "panel saw",
        "confusable": True,
        "cluster": "cutting",
    },
    # ── Surfacing and shaping ────────────────────────────────────────────────
    {
        "name": "planer",
        "display": "woodworking planer",
        "confusable": True,
        "cluster": "surfacing",
    },
    {
        "name": "jointer",
        "display": "woodworking jointer",
        "confusable": True,
        "cluster": "surfacing",
    },
    {
        "name": "spindle_moulder",
        "display": "spindle moulder",
        "confusable": True,
        "cluster": "surfacing",
    },
    {
        "name": "wood_lathe",
        "display": "wood lathe",
        "confusable": True,
        "cluster": "surfacing",
    },
    # ── Boring and finishing ─────────────────────────────────────────────────
    {
        "name": "sanding_machines",
        "display": "industrial sanding machine",
        "confusable": False,
        "cluster": None,
    },
    {
        "name": "cnc_router",
        "display": "CNC router",
        "confusable": True,
        "cluster": "cnc_group",
    },
    # ── Safety and material handling ─────────────────────────────────────────
    {
        "name": "fire_extinguisher",
        "display": "fire extinguisher",
        "confusable": False,
        "cluster": None,
    },
    {
        "name": "crane",
        "display": "industrial crane",
        "confusable": False,
        "cluster": None,
    },
]

# ---------------------------------------------------------------------------
# Derived helpers (used by multiple pipeline stages)
# ---------------------------------------------------------------------------

def class_by_name(name: str) -> dict:
    """Return the class dict for a given slug, or raise KeyError."""
    for c in CLASSES:
        if c["name"] == name:
            return c
    raise KeyError(f"Unknown class slug: {name!r}")


def confusable_classes() -> list[dict]:
    """Return only the classes that are in a confusable cluster."""
    return [c for c in CLASSES if c["confusable"]]


def standard_classes() -> list[dict]:
    """Return classes NOT in any confusable cluster."""
    return [c for c in CLASSES if not c["confusable"]]


ALL_CLASS_NAMES: list[str] = [c["name"] for c in CLASSES]

QUERY_TEMPLATES: list[str] = [
    "{machine}",
    "industrial {machine} equipment",
    "{machine} in workshop factory",
    "heavy duty {machine}",
    "{machine} detail",
    "professional {machine} machine",
    "{machine} plant setup",
    "commercial {machine} tools",
    "5 axis {machine}",
    "vertical {machine}",
    "horizontal {machine}",
    "precision {machine}",
    "{machine} metalworking",
    "large {machine} machine",
    "modern {machine} workshop",
]

