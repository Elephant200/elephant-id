"""On-the-fly thumbnail generation.

Thumbnails are produced in memory and returned as `BytesIO` buffers.
There is no on-disk cache: disk reads on modern SSDs are fast enough,
and the Dataset's image cache handles the hot path for full-resolution
reads done elsewhere.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2

from .config import (
    CODED_ROOT,
    THUMB_HARD_MAX,
    THUMB_MIN_SIZE,
)
from .paths import safe_coded_rel_image


def _clamp_size(size: int) -> int:
    return max(THUMB_MIN_SIZE, min(int(size), THUMB_HARD_MAX))


def _render_thumb(src: Path, size: int) -> io.BytesIO:
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {src}")
    height, width = image.shape[:2]
    # Fit within a size x size box, downscaling only (like PIL's thumbnail).
    scale = min(size / width, size / height, 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    _ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return io.BytesIO(encoded.tobytes())


def coded_thumb(rel: str, size: int) -> io.BytesIO:
    """Return an in-memory JPEG thumbnail for a `coded/` image."""
    rel = safe_coded_rel_image(rel)
    size = _clamp_size(size)
    src = CODED_ROOT / rel
    if not src.exists():
        raise FileNotFoundError(f"Missing image: {src}")
    return _render_thumb(src, size)


def absolute_thumb(src_abs: Path, size: int) -> io.BytesIO:
    """Return an in-memory JPEG thumbnail for an absolute path."""
    size = _clamp_size(size)
    return _render_thumb(src_abs, size)
