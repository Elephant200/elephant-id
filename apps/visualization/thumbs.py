"""On-the-fly thumbnail generation.

Thumbnails are produced in memory and returned as ``BytesIO`` buffers.
There is no on-disk cache: disk reads on modern SSDs are fast enough,
and the Dataset's image cache handles the hot path for full-resolution
reads done elsewhere.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

from .config import (
    CODED_ROOT,
    THUMB_HARD_MAX,
    THUMB_MIN_SIZE,
)
from .paths import safe_coded_rel_image


def _clamp_size(size: int) -> int:
    return max(THUMB_MIN_SIZE, min(int(size), THUMB_HARD_MAX))


def _render_thumb(src: Path, size: int) -> io.BytesIO:
    buf = io.BytesIO()
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.Resampling.LANCZOS)
        im.save(buf, format="JPEG", quality=85, optimize=True)
    buf.seek(0)
    return buf


def coded_thumb(rel: str, size: int) -> io.BytesIO:
    """Return an in-memory JPEG thumbnail for an image under ``coded/``."""
    rel = safe_coded_rel_image(rel)
    size = _clamp_size(size)
    src = CODED_ROOT / rel
    if not src.exists():
        raise FileNotFoundError(f"Missing image: {src}")
    return _render_thumb(src, size)


def absolute_thumb(src_abs: Path, size: int) -> io.BytesIO:
    """Return an in-memory JPEG thumbnail for an absolute source path."""
    size = _clamp_size(size)
    return _render_thumb(src_abs, size)
