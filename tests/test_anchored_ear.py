import numpy as np

from elephant_id.ai.detection import Detection
from elephant_id.coding.ears.anchored_ear import AnchoredEar


def _rectangle_mask(
    height: int,
    width: int,
    x0: int,
    y0: int,
    box_width: int,
    box_height: int,
) -> np.ndarray:
    """Build a boolean mask with one solid rectangle at the given position."""
    mask = np.zeros((height, width), dtype=bool)
    mask[y0 : y0 + box_height, x0 : x0 + box_width] = True
    return mask


def _anchored_ear(
    rle_from_mask,
    height: int,
    width: int,
    x0: int,
    y0: int,
    box_width: int,
    box_height: int,
) -> AnchoredEar:
    """Anchor a rectangular ear whose cut contour spans the whole rectangle.

    The anchors sit on the top-left and bottom-left corners, so the longer
    contour path wraps around the right side and its bounding box equals the
    full rectangle.
    """
    mask = _rectangle_mask(height, width, x0, y0, box_width, box_height)
    ear = Detection(
        xyxy=(float(x0), float(y0), float(x0 + box_width), float(y0 + box_height)),
        class_name="ear",
        class_id=0,
        confidence=0.9,
        rle_mask=rle_from_mask(mask),
    )
    anchor = Detection(
        xyxy=(float(x0), float(y0), float(x0 + box_width), float(y0 + box_height)),
        class_name="anchor",
        class_id=0,
        confidence=0.9,
        keypoints=(
            (float(x0), float(y0)),
            (float(x0), float(y0 + box_height - 1)),
        ),
    )
    return AnchoredEar(ear, anchor)


def test_quality_scores_ideal_centered_crop_near_one(rle_from_mask):
    ear = _anchored_ear(
        rle_from_mask, height=1000, width=1000, x0=300, y0=250, box_width=400, box_height=500
    )

    assert ear.xyxy == (300.0, 250.0, 700.0, 750.0)
    assert ear.quality > 0.99


def test_quality_scores_extreme_aspect_near_zero(rle_from_mask):
    ear = _anchored_ear(
        rle_from_mask, height=1000, width=1000, x0=100, y0=400, box_width=800, box_height=100
    )

    assert ear.quality < 0.01


def test_quality_is_zero_for_edge_touching_crop_regardless_of_aspect(rle_from_mask):
    ear = _anchored_ear(
        rle_from_mask, height=1000, width=1000, x0=0, y0=250, box_width=400, box_height=500
    )

    assert ear.xyxy == (0.0, 250.0, 400.0, 750.0)
    assert ear.quality == 0.0
