"""Filesystem locations used by the Alphaphant sidecar."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

GALLERY_PROFILES_NPZ = REPO_ROOT / "outputs" / "tear_matching_eval" / "hq_profiles.npz"
GALLERY_MANIFEST_CSV = REPO_ROOT / "outputs" / "high_quality" / "manifest.csv"
MODEL_CACHE_ROOT = REPO_ROOT / ".cache"
DATASET_ROOT = REPO_ROOT / "dataset"


def gallery_profiles_path() -> Path:
    """Return the gallery profile cache, honoring ``ALPHAPHANT_GALLERY_PROFILES``.

    The override lets a demo run against a filtered gallery (for example with
    held-out sightings removed) without touching the evaluation outputs.
    """
    override = os.getenv("ALPHAPHANT_GALLERY_PROFILES")
    if override:
        return Path(override)
    return GALLERY_PROFILES_NPZ


def default_data_dir() -> Path:
    """Return the sidecar's writable data directory.

    Honors ``ALPHAPHANT_DATA_DIR`` so tests and packaged builds can relocate
    application state.
    """
    override = os.getenv("ALPHAPHANT_DATA_DIR")
    if override:
        return Path(override)
    return REPO_ROOT / "outputs" / "alphaphant"
