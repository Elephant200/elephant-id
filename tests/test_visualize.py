import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco_mask

from elephant_id.visualize import apply_alpha_mask, decode_rle_mask


def _rle_from_mask(mask: np.ndarray) -> dict:
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def test_decode_rle_mask_returns_boolean_mask():
    mask = np.array(
        [
            [False, True, False],
            [False, True, True],
        ]
    )

    decoded = decode_rle_mask(_rle_from_mask(mask))

    assert decoded.dtype == bool
    assert decoded.tolist() == mask.tolist()


def test_apply_alpha_mask_overlays_masked_pixels_only():
    image = Image.new("RGB", (2, 1), (10, 20, 30))
    mask = np.array([[True, False]])

    output = apply_alpha_mask(
        image,
        mask,
        color=(110, 120, 130),
        alpha=1.0,
    )

    assert output.getpixel((0, 0)) == (110, 120, 130)
    assert output.getpixel((1, 0)) == (10, 20, 30)


def test_apply_alpha_mask_rejects_bad_alpha():
    image = Image.new("RGB", (1, 1))
    mask = np.array([[True]])

    with pytest.raises(ValueError, match="alpha"):
        apply_alpha_mask(image, mask, alpha=1.5)


def test_apply_alpha_mask_rejects_shape_mismatch():
    image = Image.new("RGB", (2, 1))
    mask = np.array([[True], [False]])

    with pytest.raises(ValueError, match="mask shape"):
        apply_alpha_mask(image, mask)
