"""Image, mask, and bounding-box utilities."""

from typing import Any

import numpy as np
import PIL.Image as Image
from pycocotools import mask as coco_mask


def center_to_xyxy(
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """Convert a center-format box to corner coordinates."""
    return (
        x - width / 2,
        y - height / 2,
        x + width / 2,
        y + height / 2,
    )


def clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Clip an xyxy box to image bounds.

    Coordinates use the half-open convention: ``x2``/``y2`` are exclusive, so a
    valid box has ``x2 > x1`` and ``y2 > y1`` and must overlap the image. A box
    that is reversed, zero-area, or entirely outside the image is rejected
    rather than silently collapsed to a one-pixel box.

    Raises:
        ValueError: If the box has non-positive area, or does not intersect the
            image after clipping.
    """
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Box must have positive width and height: xyxy={(x1, y1, x2, y2)}"
        )

    ix1 = max(0, min(image_width - 1, round(x1)))
    iy1 = max(0, min(image_height - 1, round(y1)))
    ix2 = max(0, min(image_width, round(x2)))
    iy2 = max(0, min(image_height, round(y2)))

    if ix2 <= ix1 or iy2 <= iy1:
        raise ValueError(
            f"Box does not intersect image bounds "
            f"{(image_width, image_height)}: xyxy={(x1, y1, x2, y2)}"
        )
    return ix1, iy1, ix2, iy2


def apply_crop(
    image: Image.Image,
    crop_xyxy: tuple[float, float, float, float],
) -> Image.Image:
    """Return a copy of image cropped to a clipped xyxy box."""
    return image.crop(
        clip_xyxy(
            crop_xyxy[0],
            crop_xyxy[1],
            crop_xyxy[2],
            crop_xyxy[3],
            image.width,
            image.height,
        )
    )


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


def apply_mask(
    image: Image.Image,
    mask: np.ndarray,
    background: tuple[int, int, int] = (0, 0, 0),
    crop: bool = False,
) -> Image.Image:
    """Return image with unmasked pixels replaced by background.

    If crop is True, the masked image is cropped to the mask's True-pixel bounds.
    """
    rgb = image.convert("RGB")
    bool_mask = mask.astype(bool)
    if bool_mask.shape != (rgb.height, rgb.width):
        raise ValueError(
            f"mask shape {bool_mask.shape} does not match image size "
            f"{(rgb.height, rgb.width)}"
        )

    output = np.empty((rgb.height, rgb.width, 3), dtype=np.uint8)
    output[:, :] = np.array(background, dtype=np.uint8)
    source = np.array(rgb)
    output[bool_mask] = source[bool_mask]
    masked = Image.fromarray(output, mode="RGB")

    if crop:
        return apply_crop(masked, mask_bounds(bool_mask))
    return masked

