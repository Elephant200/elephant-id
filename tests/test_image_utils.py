import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco_mask

from elephant_id.image_utils import (
    apply_crop,
    apply_mask,
    center_to_xyxy,
    clip_xyxy,
    decode_rle_mask,
)


def _rle_from_mask(mask: np.ndarray) -> dict:
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def test_center_to_xyxy_matches_cached_sam3_geometry():
    # Derived from .cache/sam3/body/Adam_2011-03-31_02.
    assert center_to_xyxy(
        x=1072.5,
        y=1319.5,
        width=315.0,
        height=357.0,
    ) == (915.0, 1141.0, 1230.0, 1498.0)


def test_clip_xyxy_clips_to_image_bounds_and_preserves_area():
    assert clip_xyxy(
        x1=-5,
        y1=1.2,
        x2=10,
        y2=12,
        image_width=8,
        image_height=6,
    ) == (0, 1, 8, 6)


def test_clip_xyxy_expands_collapsed_boxes():
    assert clip_xyxy(
        x1=4,
        y1=4,
        x2=4,
        y2=4,
        image_width=8,
        image_height=6,
    ) == (4, 4, 5, 5)


def test_apply_crop_uses_clipped_xyxy_box():
    image = Image.new("RGB", (4, 3), (10, 20, 30))

    cropped = apply_crop(image, (-10, 1, 3, 99))

    assert cropped.size == (3, 2)


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


def test_apply_mask_replaces_unmasked_pixels():
    image = Image.new("RGB", (2, 1), (10, 20, 30))
    mask = np.array([[True, False]])

    output = apply_mask(image, mask, background=(1, 2, 3), crop=False)

    assert output.size == (2, 1)
    assert output.getpixel((0, 0)) == (10, 20, 30)
    assert output.getpixel((1, 0)) == (1, 2, 3)


def test_apply_mask_can_crop_to_mask_bounds():
    image = Image.new("RGB", (4, 3), (10, 20, 30))
    mask = np.array(
        [
            [False, False, False, False],
            [False, True, True, False],
            [False, False, False, False],
        ]
    )

    output = apply_mask(image, mask, crop=True)

    assert output.size == (2, 1)


def test_apply_mask_rejects_shape_mismatch():
    image = Image.new("RGB", (2, 1))
    mask = np.array([[True], [False]])

    with pytest.raises(ValueError, match="mask shape"):
        apply_mask(image, mask)


def test_apply_mask_rejects_empty_crop_mask():
    image = Image.new("RGB", (2, 1))
    mask = np.array([[False, False]])

    with pytest.raises(ValueError, match="empty"):
        apply_mask(image, mask, crop=True)
