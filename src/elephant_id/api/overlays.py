"""Render annotated evidence images for the review UI.

Draws the analyzer's intermediate outputs — body mask and box, trunk and tusk
boxes, ear contours and anchor keypoints — onto copies of the photo so the
reviewer can see exactly what the models found.
"""

import cv2
import numpy as np
from loguru import logger

from elephant_id.image import BgrImage
from elephant_id.image.boxes import clip_xyxy
from elephant_id.image.masks import decode_rle_mask

BODY_TINT_RGB = (70, 130, 255)
BODY_BOX_RGB = (70, 130, 255)
TRUNK_RGB = (220, 220, 80)
TUSK_RGB = (255, 105, 65)
EAR_RGB = (80, 220, 130)
ANCHOR_UPPER_RGB = (255, 255, 255)
ANCHOR_LOWER_RGB = (255, 220, 80)
EAR_CROP_PAD_FRACTION = 0.08


def annotate_photo(image: BgrImage, analysis: dict) -> BgrImage:
    """Return a copy of the photo with all detection evidence drawn on."""
    annotated = image.copy()
    body = analysis["shared_data"]["body"]
    if body.rle_mask is not None:
        body_mask = decode_rle_mask(body.rle_mask)
        annotated[body_mask] = (
            0.8 * annotated[body_mask] + 0.2 * np.array(BODY_TINT_RGB[::-1])
        ).astype(np.uint8)
    _draw_box(annotated, body.xyxy, "body", BODY_BOX_RGB)

    for trunk in analysis["shared_data"]["trunks"]:
        _draw_box(annotated, trunk.xyxy, f"trunk {trunk.confidence:.0%}", TRUNK_RGB)
    for tusk in analysis["tusks"]:
        _draw_box(
            annotated,
            (tusk["x1"], tusk["y1"], tusk["x2"], tusk["y2"]),
            f"{tusk['side']} tusk {tusk['confidence']:.0%}",
            TUSK_RGB,
        )
    for ear_data in analysis["ears"]:
        ear = ear_data["ear"]
        _draw_box(annotated, ear.xyxy, f"{ear.side} ear", EAR_RGB)
        _draw_ear_contour(annotated, ear, origin=(0.0, 0.0))
    return annotated


def annotate_ear_crop(image: BgrImage, ear: object) -> BgrImage:
    """Return a padded ear crop with the anchored contour and anchors drawn."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = ear.xyxy
    pad = EAR_CROP_PAD_FRACTION * max(x2 - x1, y2 - y1)
    cx1, cy1, cx2, cy2 = clip_xyxy(
        x1 - pad, y1 - pad, x2 + pad, y2 + pad, width, height
    )
    crop = image[cy1:cy2, cx1:cx2].copy()
    _draw_ear_contour(crop, ear, origin=(float(cx1), float(cy1)))
    return crop


def _draw_ear_contour(
    image: BgrImage,
    ear: object,
    origin: tuple[float, float],
) -> None:
    """Draw the anchored contour and anchor keypoints, shifted by origin."""
    try:
        contour = ear.contour - np.array(origin)
        points = np.round(contour).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, EAR_RGB[::-1], 2, cv2.LINE_AA)
        for point, rgb in zip(
            ear.anchor_points, (ANCHOR_UPPER_RGB, ANCHOR_LOWER_RGB), strict=True
        ):
            center = tuple(np.round(np.array(point) - np.array(origin)).astype(int))
            cv2.circle(image, center, 5, rgb[::-1], -1, cv2.LINE_AA)
    except Exception as error:
        logger.warning(f"Could not draw ear contour: {error}")


def _draw_box(
    image: BgrImage,
    xyxy: tuple[float, float, float, float],
    label: str,
    rgb: tuple[int, int, int],
) -> None:
    """Draw a box with its label on a solid chip so it stays legible."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clip_xyxy(*xyxy, width, height)
    bgr = rgb[::-1]
    thickness = max(2, round(min(width, height) / 500))
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), bgr, thickness, cv2.LINE_AA)

    font_scale = 0.45 * thickness
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    pad = 4 * thickness
    chip_top = max(0, y1 - text_height - baseline - 2 * pad)
    chip_left = max(0, min(x1, width - text_width - 2 * pad))
    cv2.rectangle(
        image,
        (chip_left, chip_top),
        (chip_left + text_width + 2 * pad, chip_top + text_height + baseline + 2 * pad),
        bgr,
        -1,
    )
    cv2.putText(
        image,
        label,
        (chip_left + pad, chip_top + text_height + pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
