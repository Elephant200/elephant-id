"""
Visualization utilities for model predictions.

SAM3 prediction schema (per detection), as returned by ``Sam3Runner.run`` /
``Sam3Service.run`` inside ``response["predictions"]`` (a flat list).

    {
        "x": float,            # bbox center x in pixels
        "y": float,            # bbox center y in pixels
        "width": float,        # bbox width in pixels
        "height": float,       # bbox height in pixels
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
import PIL.Image as Image
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


def _center_to_corners(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Convert center-format bbox to clipped corner coordinates."""
    x1 = int(round(x - width / 2))
    y1 = int(round(y - height / 2))
    x2 = int(round(x + width / 2))
    y2 = int(round(y + height / 2))

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(0, min(image_width - 1, x2))
    y2 = max(0, min(image_height - 1, y2))

    # Keep a minimum 1px bbox for OpenCV drawing.
    if x2 <= x1:
        x2 = min(image_width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(image_height - 1, y1 + 1)
    return x1, y1, x2, y2

def apply_alpha_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> Image.Image:
    """
    Return a copy of image with a semi-transparent color overlay where mask is True.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    base = image.convert("RGBA")

    if mask.shape != (base.height, base.width):
        raise ValueError(
            f"mask shape {mask.shape} does not match image size "
            f"{(base.height, base.width)}"
        )

    overlay = Image.new("RGBA", base.size, (*color, 0))

    alpha_channel = np.zeros((base.height, base.width), dtype=np.uint8)
    alpha_channel[mask] = int(255 * alpha)

    overlay.putalpha(Image.fromarray(alpha_channel, mode="L"))

    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_rle_mask_overlay(
    image: Image.Image,
    rle: dict[str, Any],
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> Image.Image:
    """Overlay a single COCO RLE mask on a PIL image."""
    mask = decode_rle_mask(rle)
    return apply_alpha_mask(image, mask, color=color, alpha=alpha)


_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 127, 14),   # orange
    (31, 119, 180),   # blue
    (44, 160, 44),    # green
    (214, 39, 40),    # red
    (148, 103, 189),  # purple
    (140, 86, 75),    # brown
)


def visualize_predictions(
    image: Image.Image,
    predictions: list[dict[str, Any]],
    mask_alpha: float = 0.35,
) -> Image.Image:
    """Draw SAM3 detections (RLE masks + boxes + labels) on a PIL image.

    Args:
        image: Source PIL image.
        predictions: List of detection dicts following the schema in this
            module's docstring
        mask_alpha: Blend factor for the mask overlay in [0, 1].

    Returns:
        New PIL RGB image with all detections rendered.
    """
    rgb = image.convert("RGB")
    output = np.array(rgb)
    image_height, image_width = output.shape[:2]

    for prediction in predictions:
        class_id = int(prediction.get("class_id", 0))
        class_name = str(prediction.get("class", "unknown")).strip()
        confidence = float(prediction.get("confidence", 0.0))
        color = _PALETTE[class_id % len(_PALETTE)]

        rle_mask = prediction.get("rle_mask")
        if rle_mask:
            mask = decode_rle_mask(rle_mask)
            if mask.shape != (image_height, image_width):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            output[mask] = (
                (1.0 - mask_alpha) * output[mask]
                + mask_alpha * np.array(color, dtype=np.float32)
            ).astype(np.uint8)

        x1, y1, x2, y2 = _center_to_corners(
            prediction["x"],
            prediction["y"],
            prediction["width"],
            prediction["height"],
            image_width,
            image_height,
        )
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

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
            color,
            1,
            cv2.LINE_AA,
        )

    return Image.fromarray(output)