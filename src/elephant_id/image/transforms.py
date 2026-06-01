"""Pixel-space transforms producing :data:`BgrImage` outputs."""

import numpy as np

from elephant_id.image.bgr import BgrImage
from elephant_id.image.boxes import clip_xyxy
from elephant_id.image.masks import RleMask, decode_rle_mask, mask_bounds


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
    mask: np.ndarray | RleMask,
    background: tuple[int, int, int] = (0, 0, 0),
    crop: bool = False,
) -> BgrImage:
    """Return image with unmasked pixels replaced by background.

    Args:
        image: A BGR image to apply the mask to.
        mask: A boolean mask to apply to the image. Either a numpy array or a RleMask.
        background: The background RGB color to use for the masked pixels. (Default: black)
        crop: Whether to crop the image to the mask's True-pixel bounds.

    Returns:
        A new BGR image with the mask applied.

    Raises:
        ValueError: If the background is not three values in ``[0, 255]``, the
            mask shape does not match the image, or cropping an empty mask.
    """
    if len(background) != 3 or any(not 0 <= c <= 255 for c in background):
        raise ValueError(
            f"background must be three values in [0, 255]: {background}"
        )

    if isinstance(mask, dict) and "size" in mask and "counts" in mask:
        mask = decode_rle_mask(mask)

    bool_mask = mask.astype(bool)
    if bool_mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask shape {bool_mask.shape} does not match image size {image.shape[:2]}"
        )

    output = np.empty_like(image)
    background = background[::-1] # BGR -> RGB
    output[:, :] = np.array(background, dtype=np.uint8)
    output[bool_mask] = image[bool_mask]

    if crop:
        return apply_crop(output, mask_bounds(bool_mask))
    return output
