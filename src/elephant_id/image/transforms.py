"""Pixel-space transforms producing :data:`BgrImage` outputs."""

import numpy as np

from elephant_id.image.bgr import BgrImage
from elephant_id.image.boxes import clip_xyxy
from elephant_id.image.masks import mask_bounds


def apply_crop(
    image: BgrImage,
    crop_xyxy: tuple[float, float, float, float],
) -> BgrImage:
    """Return a copy of image cropped to a clipped xyxy box."""
    x1, y1, x2, y2 = clip_xyxy(
        crop_xyxy[0],
        crop_xyxy[1],
        crop_xyxy[2],
        crop_xyxy[3],
        image.shape[1],
        image.shape[0],
    )
    return image[y1:y2, x1:x2].copy()


def apply_mask(
    image: BgrImage,
    mask: np.ndarray,
    background: tuple[int, int, int] = (0, 0, 0),
    crop: bool = False,
) -> BgrImage:
    """Return image with unmasked pixels replaced by background.

    If crop is True, the masked image is cropped to the mask's True-pixel bounds.
    ``background`` is interpreted in BGR order.
    """
    bool_mask = mask.astype(bool)
    if bool_mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask shape {bool_mask.shape} does not match image size {image.shape[:2]}"
        )

    output = np.empty_like(image)
    output[:, :] = np.array(background, dtype=np.uint8)
    output[bool_mask] = image[bool_mask]

    if crop:
        return apply_crop(output, mask_bounds(bool_mask))
    return output
