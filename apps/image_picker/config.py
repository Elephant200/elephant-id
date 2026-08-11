"""Configuration and filesystem paths for the matching image picker."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset" / "elephants-alive"
CODED_ROOT = DATASET_ROOT / "coded"
CSV_PATH = DATASET_ROOT / "images.csv"

HIGH_QUALITY_ROOT = REPO_ROOT / "outputs" / "high_quality"
HIGH_QUALITY_MANIFEST = HIGH_QUALITY_ROOT / "manifest.csv"

SIDES: tuple[str, str] = ("left", "right")

# Eligibility rule. A sighting qualifies when it has at least one left-side and
# one right-side ear candidate scoring above ``QUALITY_THRESHOLD`` on the
# anchored-ear quality prior; an elephant is presented once it has at least
# ``MIN_QUALIFYING_SIGHTINGS`` qualifying sightings.
QUALITY_THRESHOLD = 0.5
MIN_QUALIFYING_SIGHTINGS = 5
