import numpy as np
import pytest

from elephant_id.visualize import (
    apply_alpha_mask,
    draw_rle_mask_overlay,
    visualize_predictions,
)

# Pixel assertions are on the BGR buffer. Where a colour has a human (RGB) name,
# the stored value is reversed: e.g. palette blue RGB (31,119,180) -> (180,119,31).


def test_draw_rle_mask_overlay_colors_masked_pixels(rle_from_mask):
    image = np.full((1, 2, 3), (10, 20, 30), dtype=np.uint8)
    mask = np.array([[True, False]])

    output = draw_rle_mask_overlay(
        image,
        rle_from_mask(mask),
        color=(110, 120, 130),  # RGB
        alpha=1.0,
    )

    assert tuple(output[0, 0]) == (130, 120, 110)  # RGB color flipped to BGR
    assert tuple(output[0, 1]) == (10, 20, 30)


def test_apply_alpha_mask_rejects_invalid_alpha():
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    mask = np.array([[True]])

    with pytest.raises(ValueError, match="alpha"):
        apply_alpha_mask(image, mask, alpha=1.1)


def test_apply_alpha_mask_rejects_shape_mismatch():
    image = np.zeros((1, 2, 3), dtype=np.uint8)
    mask = np.array([[True], [False]])

    with pytest.raises(ValueError, match="mask shape"):
        apply_alpha_mask(image, mask)


def test_visualize_predictions_draws_box_and_preserves_shape():
    image = np.full((12, 12, 3), (10, 20, 30), dtype=np.uint8)
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

    assert output.shape == (12, 12, 3) and output.dtype == np.uint8
    # class_id 1 -> palette blue RGB (31,119,180) -> BGR (180,119,31).
    assert tuple(output[1, 1]) == (180, 119, 31)


def test_visualize_predictions_resizes_rle_mask_to_image(rle_from_mask):
    image = np.full((4, 4, 3), (10, 20, 30), dtype=np.uint8)
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

    assert output.shape == (4, 4, 3)
    # class_id 2 -> palette green (44,160,44); symmetric under BGR flip.
    assert tuple(output[0, 0]) == (44, 160, 44)


def test_visualize_predictions_uses_full_size_mask_without_resize(rle_from_mask):
    image = np.full((4, 4, 3), (10, 20, 30), dtype=np.uint8)
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

    assert output.shape == (4, 4, 3)
    assert tuple(output[0, 0]) == (44, 160, 44)


def test_visualize_predictions_fills_defaults_for_minimal_prediction():
    image = np.full((12, 12, 3), (10, 20, 30), dtype=np.uint8)

    # Only the bounding box keys are present; class/class_id/confidence default.
    output = visualize_predictions(image, [{"x1": 1, "y1": 1, "x2": 8, "y2": 8}])

    assert output.shape == (12, 12, 3)
    # class_id defaults to 0 -> palette orange RGB (255,127,14) -> BGR (14,127,255).
    assert tuple(output[1, 1]) == (14, 127, 255)


def test_visualize_predictions_empty_list_returns_unchanged_copy():
    image = np.full((2, 3, 3), (10, 20, 30), dtype=np.uint8)

    output = visualize_predictions(image, [])

    assert output is not image
    assert np.array_equal(output, image)


def test_visualize_predictions_mixes_masked_and_unmasked_detections(rle_from_mask):
    # One detection carries a mask, the other does not. The masked detection
    # still renders its mask; the maskless one gets an all-False plane and only
    # contributes a box/label, leaving its own corner pixel untouched by a mask.
    image = np.full((8, 8, 3), (10, 20, 30), dtype=np.uint8)
    masked_pixel = np.zeros((8, 8), dtype=bool)
    masked_pixel[0, 0] = True
    predictions = [
        {
            "class": "ear",
            "class_id": 2,  # green (44, 160, 44), symmetric under BGR flip
            "confidence": 0.9,
            "x1": 0,
            "y1": 0,
            "x2": 4,
            "y2": 4,
            "rle_mask": rle_from_mask(masked_pixel),
        },
        {
            "class": "tusk",
            "class_id": 3,
            "confidence": 0.5,
            "x1": 5,
            "y1": 5,
            "x2": 7,
            "y2": 7,
        },
    ]

    output = visualize_predictions(image, predictions, mask_alpha=1.0)

    assert output.shape == (8, 8, 3)
    assert tuple(output[0, 0]) == (44, 160, 44)  # masked detection's mask
