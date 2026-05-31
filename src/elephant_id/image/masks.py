"""COCO RLE mask decoding and mask geometry."""

from typing import Any

import numpy as np
from pycocotools import mask as coco_mask


def decode_rle_mask(rle_mask: dict[str, Any]) -> np.ndarray:
    """Decode a COCO-style RLE mask into a 2D boolean array."""
    encoded = {
        "size": rle_mask["size"],
        "counts": rle_mask["counts"].encode("utf-8")
        if isinstance(rle_mask["counts"], str)
        else rle_mask["counts"],
    }
    decoded = coco_mask.decode(encoded)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return decoded.astype(bool)


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return xyxy bounds around True mask pixels."""
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Cannot compute bounds for an empty mask")
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )
