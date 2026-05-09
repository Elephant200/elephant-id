"""On-disk thumbnail cache."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from .config import (
    CODED_ROOT,
    THUMB_CACHE_ROOT,
    THUMB_HARD_MAX,
    THUMB_MIN_SIZE,
)
from .paths import safe_coded_rel_image


def _clamp_size(size: int) -> int:
    return max(THUMB_MIN_SIZE, min(int(size), THUMB_HARD_MAX))


def _write_thumb(src: Path, dst: Path, size: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        im.save(dst, format="JPEG", quality=85, optimize=True)


def coded_thumb(rel: str, size: int) -> Path:
    """Return a thumbnail path for an image under ``coded/``, generating if needed."""
    rel = safe_coded_rel_image(rel)
    size = _clamp_size(size)
    src = CODED_ROOT / rel
    if not src.exists():
        raise FileNotFoundError(f"Missing image: {src}")
    dst = THUMB_CACHE_ROOT / str(size) / Path(rel).with_suffix(".jpg")
    if dst.exists():
        return dst
    _write_thumb(src, dst, size)
    return dst


def absolute_thumb(src_abs: Path, namespace: str, size: int) -> Path:
    """Return a thumbnail path for an absolute source path, generating if needed.

    The destination is keyed by ``namespace`` and a hash of the resolved source
    path. Used for files outside ``coded/`` (e.g. saved samples).
    """
    size = _clamp_size(size)
    safe_key = (namespace + "_" + str(src_abs.resolve())).replace("/", "_").replace("\\", "_")[:400]
    h = hashlib.md5(safe_key.encode()).hexdigest()
    dst = THUMB_CACHE_ROOT / "saved_preview" / str(size) / f"{h}.jpg"
    if dst.exists():
        return dst
    _write_thumb(src_abs, dst, size)
    return dst
