"""Configuration and filesystem paths for the image picker."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset" / "elephants-alive"
CODED_ROOT = DATASET_ROOT / "coded"
CSV_PATH = DATASET_ROOT / "images.csv"
FILTERED_CROPS_ROOT = REPO_ROOT / "outputs" / "ear_segmentation_filtered"
HIGH_QUALITY_ROOT = REPO_ROOT / "outputs" / "high_quality"
HIGH_QUALITY_IMAGES_ROOT = HIGH_QUALITY_ROOT / "images"
HIGH_QUALITY_MANIFEST = HIGH_QUALITY_ROOT / "manifest.csv"

QUEUE_SIZE = 125
TARGET_DONE_IDENTITIES = 100
MIN_SIGHTINGS = 3
MIN_IDENTITY_PHOTOS = 25
MIN_SIDE_CANDIDATES = 25
MIN_SELECTIONS_PER_IDENTITY = 5
MAX_SELECTIONS_PER_IDENTITY = 5
MAX_PHOTOS_PER_IDENTITY_ANALYSIS = 100
MIN_EAR_BOX_AREA = 360_000.0
MIN_EAR_BOX_HEIGHT_WIDTH = 1.0
MAX_EAR_BOX_HEIGHT_WIDTH = 1.5
CROP_PREVIEW_PAD = 0.06
QUEUE_SEED = "elephant-id-high-quality-picker-v1"

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"})
SIDES = ("left", "right")
