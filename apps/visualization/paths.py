"""Filesystem path safety and resolution helpers.

All public helpers raise :class:`ValueError` for malformed or
out-of-tree input. They never silently return ``None`` for traversal
attempts.
"""

from __future__ import annotations

import re
from pathlib import Path

from .config import (
    CODED_ROOT,
    IMG_EXTS,
    SAMPLES_ROOT,
    SAMPLES_SIGHTINGS_ROOT,
)

SAVED_SIGHTING_DIR_RE = re.compile(r"^(.+)_(\d{4}-\d{2}-\d{2})(?:_dup\d+)?$")


def is_image(path: Path) -> bool:
    return path.suffix in IMG_EXTS


def parse_sighting_folder_name(folder_name: str) -> tuple[str, str] | None:
    """Return ``(name, date)`` for saved sighting folder names."""
    m = SAVED_SIGHTING_DIR_RE.match(folder_name)
    if not m:
        return None
    return m.group(1), m.group(2)


def _normalize_rel(rel: str) -> str:
    rel = (rel or "").replace("\\", "/").strip().lstrip("/")
    if ".." in rel.split("/"):
        raise ValueError("Invalid path")
    return rel


def safe_coded_rel_image(rel: str) -> str:
    """Validate and normalize a relative image path under ``coded/``."""
    return _normalize_rel(rel)


def safe_saved_sighting_dir(rel: str) -> Path:
    """Resolve a saved sighting folder under ``samples/``."""
    rel = _normalize_rel(rel)
    if not rel.startswith("sightings/"):
        raise ValueError("Invalid sighting path")
    full = (SAMPLES_ROOT / rel).resolve()
    full.relative_to(SAMPLES_SIGHTINGS_ROOT.resolve())
    return full


def safe_saved_sighting_file(rel: str) -> Path:
    """Resolve a saved sighting image under ``samples/``."""
    rel = _normalize_rel(rel)
    if not rel.startswith("sightings/"):
        raise ValueError("Invalid path")
    full = (SAMPLES_ROOT / rel).resolve()
    full.relative_to(SAMPLES_SIGHTINGS_ROOT.resolve())
    if not full.is_file():
        raise ValueError("Not a file")
    if not is_image(full):
        raise ValueError("Not an image")
    return full


def samples_folder_rel(folder: Path) -> str:
    return str(folder.relative_to(SAMPLES_ROOT)).replace("\\", "/")


def coded_sighting_dir(name: str, date: str) -> Path:
    return CODED_ROOT / name / date
