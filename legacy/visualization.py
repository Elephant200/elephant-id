"""
Visualization utilities for model predictions.
"""

from typing import Any

import cv2
import numpy as np
from pycocotools import mask as coco_mask


def _decode_rle_mask(rle_mask: dict[str, Any]) -> np.ndarray:
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


def visualize_predictions(
    image: np.ndarray,
    predictions: list[dict],
    mask_alpha: float = 0.35,
) -> np.ndarray:
    """Draw detections (RLE masks + boxes + labels) on an image."""
    output = image.copy()
    image_height, image_width = output.shape[:2]

    palette = [
        (255, 127, 14),   # orange
        (31, 119, 180),   # blue
        (44, 160, 44),    # green
        (214, 39, 40),    # red
        (148, 103, 189),  # purple
        (140, 86, 75),    # brown
    ]

    for prediction in predictions:
        class_id = int(prediction.get("class_id", 0))
        class_name = str(prediction.get("class", "unknown")).strip()
        confidence = float(prediction.get("confidence", 0.0))
        color = palette[class_id % len(palette)]

        rle_mask = prediction.get("rle_mask")
        if rle_mask:
            mask = _decode_rle_mask(rle_mask)
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

    return output
