"""COCO RLE mask decoding and mask geometry."""

from typing import TypedDict

import numpy as np
from pycocotools import mask as coco_mask


class RleMask(TypedDict):
    size: list[int] # [height, width]
    counts: str | bytes # COCO RLE, utf-8 encoded

def decode_rle_mask(rle_mask: RleMask) -> np.ndarray:
    """Decode a COCO-style RLE mask into a contiguous 2D boolean array.

    Raises:
        ValueError: If `size`/`counts` is missing, or a field
            has the wrong type or shape.
    """
    if "size" not in rle_mask or "counts" not in rle_mask:
        raise ValueError("Invalid RLE mask: must have size and counts")
    size = rle_mask["size"]
    counts = rle_mask["counts"]
    if not size or not counts:
        raise ValueError("Invalid RLE mask: size and counts must be non-empty")
    if not isinstance(counts, str | bytes):
        raise ValueError("Invalid RLE mask: counts must be a string or bytes")
    if not isinstance(size, list | tuple) or len(size) != 2:
        raise ValueError("Invalid RLE mask: size must be a list or tuple of length 2")
    if any(type(dim) is not int or dim <= 0 for dim in size):
        raise ValueError("Invalid RLE mask: size values must be positive integers")

    decoded = coco_mask.decode(rle_mask)
    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]
    return np.ascontiguousarray(decoded.astype(bool))


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return xyxy bounds (half-open) around True mask pixels.

    Raises:
        ValueError: If the mask is not 2D or contains no True pixels.
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("Cannot compute bounds for an empty mask")
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )
