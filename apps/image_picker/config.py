"""Configuration and filesystem paths for the matching image picker."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "dataset" / "elephants-alive"
CODED_ROOT = DATASET_ROOT / "coded"
CSV_PATH = DATASET_ROOT / "images.csv"

HIGH_QUALITY_ROOT = REPO_ROOT / "outputs" / "high_quality"
HIGH_QUALITY_MANIFEST = HIGH_QUALITY_ROOT / "manifest.csv"

# The separate segmentation-annotation batch. A sighting is flagged (never
# excluded) when any of its photos already appears here, so the reviewer is
# aware of pipeline overlap. Crops are named ``{photo_identifier}_{side}.jpg``.
SEGMENTATION_BATCH_ROOT = REPO_ROOT / "outputs" / "ear_segmentation_batch_1"

SIDES: tuple[str, str] = ("left", "right")

# Eligibility rule. A sighting qualifies when it has at least one left-side and
# one right-side ear candidate scoring above ``QUALITY_THRESHOLD`` on the
# anchored-ear quality prior; an elephant is presented once it has at least
# ``MIN_QUALIFYING_SIGHTINGS`` qualifying sightings.
QUALITY_THRESHOLD = 0.5
MIN_QUALIFYING_SIGHTINGS = 5

# Selection rule. Within an eligible elephant the reviewer selects between
# ``MIN_SELECTED_SIGHTINGS`` and ``MAX_SELECTED_SIGHTINGS`` sightings inclusive,
# picking one canonical ear per side for each. The maximum is a hard limit; an
# elephant is "done" once it has between the minimum and maximum sightings that
# are complete (both sides picked) with no partially picked sighting left over.
MIN_SELECTED_SIGHTINGS = 3
MAX_SELECTED_SIGHTINGS = 5
