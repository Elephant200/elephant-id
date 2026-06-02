"""Paths and constants for the sighting reviewer.

Single source of truth for filesystem layout. Nothing here performs I/O
at import time.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = REPO_ROOT / "dataset" / "elephants-alive"
CODED_ROOT = DATASET_ROOT / "coded"
CSV_PATH = DATASET_ROOT / "images.csv"

SAMPLES_ROOT = REPO_ROOT / "dataset" / "samples"
SAMPLES_SIGHTINGS_ROOT = SAMPLES_ROOT / "sightings"
# Flat priority copies for external tooling (basename only); kept in sync
# incrementally by ``samples.sync_starred_for_basenames``.
STARRED_SAMPLES_ROOT = SAMPLES_ROOT / "starred"

# Priority-starred files in ``samples/sightings/<folder>/`` use this prefix
# (two asterisks + space).
PRIORITY_STAR_PREFIX = "** "

PAGE_SIZE_DEFAULT = 18
PAGE_SIZE_MAX = 60
THUMB_MAX_SIZE = 420  # px, long side
THUMB_MIN_SIZE = 64
THUMB_HARD_MAX = 1024

IMG_EXTS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        # Case variants are checked explicitly because some source trees
        # ship uppercase extensions on case-sensitive filesystems.
        ".JPG",
        ".JPEG",
        ".PNG",
    }
)
