"""
Visualization utilities for model predictions. This file will be deprecated soon.

Functions take and return a :data:`BgrImage`. ``color`` arguments and the
palette are authored in **RGB** (the human-facing convention) and flipped to BGR
only at the point they are written into the image buffer.

SAM3 prediction schema (per detection), as returned by ``Sam3Runner.run`` /
``Sam3Service.run`` inside ``response["predictions"]`` (a flat list).

    {
        "x1": float,           # bbox left edge in pixels
        "y1": float,           # bbox top edge in pixels
        "x2": float,           # bbox right edge in pixels
        "y2": float,           # bbox bottom edge in pixels
        "confidence": float,
        "class_id": int,
        "class": str,          # may have leading whitespace (e.g. " ear")
        "detection_id": str,
        "parent_id": str,
        "rle_mask": {
            "size": [H, W],    # COCO convention: [height, width]
            "counts": str,     # COCO RLE, utf-8 encoded
        },
    }
"""

from typing import Any

import cv2
import numpy as np

from elephant_id.image import BgrImage
from elephant_id.image.boxes import clip_xyxy
from elephant_id.image.masks import decode_rle_mask


def _blend_bgr(
    image: BgrImage,
    mask: np.ndarray,
    bgr: tuple[int, int, int],
    alpha: float,
) -> None:
    """Alpha-blend a solid BGR color into image where mask is True, in place.

    Raises:
        ValueError: If the alpha is not in the interval [0, 1], the mask shape
        does not match the image shape, or the bgr color is not three values in
        [0, 255].
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image size {image.shape[:2]}")
    if len(bgr) != 3 or any(not 0 <= c <= 255 for c in bgr):
        raise ValueError(f"bgr must be three values in [0, 255]: {bgr}")

    color = np.array(bgr, dtype=np.float32)
    image[mask] = ((1.0 - alpha) * image[mask] + alpha * color).astype(np.uint8)


def apply_alpha_mask(
    image: BgrImage,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> BgrImage:
    """Return a copy of image with a semi-transparent overlay where mask is True.

    Args:
        image: A BGR image to overlay the mask on.
        mask: A boolean mask to overlay on the image.
        color: The RGB color to use for the mask.
        alpha: The opacity of the mask in interval [0, 1].

    Returns:
        A new BGR image with the mask overlaid.
    """
    output = image.copy()
    _blend_bgr(output, mask.astype(bool), color[::-1], alpha)
    return output


def draw_rle_mask_overlay(
    image: BgrImage,
    rle: dict[str, Any],
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> BgrImage:
    """
    Overlay a single COCO RLE mask on an image.

    Args:
        image: A BGR image to overlay the mask on.
        rle: A COCO RLE mask to overlay on the image.
        color: The RGB color to use for the mask.
        alpha: The opacity of the mask in interval [0, 1].

    Returns:
        A new BGR image with the mask overlaid.
    """
    mask = decode_rle_mask(rle)
    return apply_alpha_mask(image, mask, color=color, alpha=alpha)


# Palette authored in RGB (with readable colour names); flipped to BGR at use.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 127, 14),  # orange
    (31, 119, 180),  # blue
    (44, 160, 44),  # green
    (214, 39, 40),  # red
    (148, 103, 189),  # purple
    (140, 86, 75),  # brown
)


def visualize_predictions(
    image: BgrImage,
    predictions: list[dict[str, Any]],
    mask_alpha: float = 0.35,
) -> BgrImage:
    """Draw SAM3 detections (RLE masks + boxes + labels) on an image.

    Args:
        image: Source BGR image.
        predictions: List of detection dicts following the schema in this
            module's docstring
        mask_alpha: Blend factor for the mask overlay in [0, 1].

    Returns:
        A new BGR image with all detections rendered.
    """
    output = image.copy()
    image_height, image_width = output.shape[:2]

    for prediction in predictions:
        class_id = int(prediction.get("class_id", 0))
        class_name = str(prediction.get("class", "unknown")).strip()
        confidence = float(prediction.get("confidence", 0.0))
        bgr = _PALETTE[class_id % len(_PALETTE)][::-1]  # RGB palette -> BGR

        rle_mask = prediction.get("rle_mask")
        if rle_mask:
            mask = decode_rle_mask(rle_mask)
            if mask.shape != (image_height, image_width):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            _blend_bgr(output, mask, bgr, mask_alpha)

        x1, y1, x2, y2 = clip_xyxy(
            prediction["x1"],
            prediction["y1"],
            prediction["x2"],
            prediction["y2"],
            image_width,
            image_height,
        )
        # clip_xyxy returns half-open coords (x2/y2 exclusive); cv2.rectangle
        # treats its second corner as inclusive, so step back one pixel.
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), bgr, 2)

        label = f"{class_name} {confidence:.2f}"
        text_anchor = (x1, max(20, y1 - 8))
        cv2.putText(
            output,
            label,
            text_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            text_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            bgr,
            1,
            cv2.LINE_AA,
        )

    return output
