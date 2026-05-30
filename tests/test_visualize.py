import numpy as np
import pytest
from PIL import Image

from elephant_id.visualize import (
    apply_alpha_mask,
    draw_rle_mask_overlay,
    visualize_predictions,
)


def test_draw_rle_mask_overlay_colors_masked_pixels(rle_from_mask):
    image = Image.new("RGB", (2, 1), (10, 20, 30))
    mask = np.array([[True, False]])

    output = draw_rle_mask_overlay(
        image,
        rle_from_mask(mask),
        color=(110, 120, 130),
        alpha=1.0,
    )

    assert output.getpixel((0, 0)) == (110, 120, 130)
    assert output.getpixel((1, 0)) == (10, 20, 30)


def test_apply_alpha_mask_rejects_invalid_alpha():
    image = Image.new("RGB", (1, 1))
    mask = np.array([[True]])

    with pytest.raises(ValueError, match="alpha"):
        apply_alpha_mask(image, mask, alpha=1.1)


def test_apply_alpha_mask_rejects_shape_mismatch():
    image = Image.new("RGB", (2, 1))
    mask = np.array([[True], [False]])

    with pytest.raises(ValueError, match="mask shape"):
        apply_alpha_mask(image, mask)


def test_visualize_predictions_draws_box_and_preserves_mode():
    image = Image.new("RGB", (12, 12), (10, 20, 30))
    predictions = [
        {
            "class": "ear",
            "class_id": 1,
            "confidence": 0.75,
            "x1": 1,
            "y1": 1,
            "x2": 8,
            "y2": 8,
        }
    ]

    output = visualize_predictions(image, predictions)

    assert output.mode == "RGB"
    assert output.getpixel((1, 1)) == (31, 119, 180)


def test_visualize_predictions_resizes_rle_mask_to_image(rle_from_mask):
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    predictions = [
        {
            "class": "tail",
            "class_id": 2,
            "confidence": 0.25,
            "x1": 1,
            "y1": 1,
            "x2": 2,
            "y2": 2,
            "rle_mask": rle_from_mask(np.array([[True]])),
        }
    ]

    output = visualize_predictions(image, predictions, mask_alpha=1.0)

    assert output.mode == "RGB"
    assert output.getpixel((0, 0)) == (44, 160, 44)


def test_visualize_predictions_uses_full_size_mask_without_resize(rle_from_mask):
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    predictions = [
        {
            "class": "tail",
            "class_id": 2,
            "confidence": 0.25,
            "x1": 1,
            "y1": 1,
            "x2": 2,
            "y2": 2,
            "rle_mask": rle_from_mask(mask),
        }
    ]

    output = visualize_predictions(image, predictions, mask_alpha=1.0)

    assert output.mode == "RGB"
    assert output.getpixel((0, 0)) == (44, 160, 44)


def test_visualize_predictions_fills_defaults_for_minimal_prediction():
    image = Image.new("RGB", (12, 12), (10, 20, 30))

    # Only the bounding box keys are present; class/class_id/confidence default.
    output = visualize_predictions(image, [{"x1": 1, "y1": 1, "x2": 8, "y2": 8}])

    assert output.mode == "RGB"
    # class_id defaults to 0 -> first palette colour (orange).
    assert output.getpixel((1, 1)) == (255, 127, 14)


def test_visualize_predictions_empty_list_returns_unchanged_copy():
    image = Image.new("RGB", (3, 2), (10, 20, 30))

    output = visualize_predictions(image, [])

    assert output.mode == "RGB"
    assert output.tobytes() == image.tobytes()
