import numpy as np
import pytest

from elephant_id.image.transforms import apply_crop, apply_mask


def test_apply_crop_uses_clipped_xyxy_box():
    image = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)

    cropped = apply_crop(image, (-10, 1, 3, 99))

    assert cropped.shape == (2, 3, 3)


def test_apply_crop_returns_a_copy():
    image = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)

    cropped = apply_crop(image, (0, 0, 4, 3))
    cropped[0, 0] = (0, 0, 0)

    assert tuple(image[0, 0]) == (10, 20, 30)


def test_apply_crop_rejects_box_outside_image():
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="outside image bounds"):
        apply_crop(image, (-10, 0, 0, 3))


def test_apply_crop_rejects_zero_area_box():
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive width and height"):
        apply_crop(image, (1, 1, 1, 3))


def test_apply_mask_replaces_unmasked_pixels():
    image = np.full((1, 2, 3), (10, 20, 30), dtype=np.uint8)
    mask = np.array([[True, False]])

    output = apply_mask(image, mask, background=(1, 2, 3), crop=False)

    assert output.shape == (1, 2, 3)
    assert tuple(output[0, 0]) == (10, 20, 30)
    assert tuple(output[0, 1]) == (1, 2, 3)


def test_apply_mask_can_crop_to_mask_bounds():
    image = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)
    mask = np.array(
        [
            [False, False, False, False],
            [False, True, True, False],
            [False, False, False, False],
        ]
    )

    output = apply_mask(image, mask, crop=True)

    assert output.shape == (1, 2, 3)


def test_apply_mask_defaults_to_black_background():
    image = np.full((1, 2, 3), (10, 20, 30), dtype=np.uint8)
    mask = np.array([[True, False]])

    output = apply_mask(image, mask)

    assert tuple(output[0, 1]) == (0, 0, 0)


def test_apply_mask_rejects_shape_mismatch():
    image = np.zeros((1, 2, 3), dtype=np.uint8)
    mask = np.array([[True], [False]])

    with pytest.raises(ValueError, match="mask shape"):
        apply_mask(image, mask)


def test_apply_mask_rejects_empty_crop_mask():
    image = np.zeros((1, 2, 3), dtype=np.uint8)
    mask = np.array([[False, False]])

    with pytest.raises(ValueError, match="empty"):
        apply_mask(image, mask, crop=True)


@pytest.mark.parametrize(
    "background",
    [(0, 0), (0, 0, 0, 0), (300, 0, 0), (-1, 0, 0)],
)
def test_apply_mask_rejects_invalid_background(background):
    image = np.zeros((1, 2, 3), dtype=np.uint8)
    mask = np.array([[True, False]])

    with pytest.raises(ValueError, match="background must be three values"):
        apply_mask(image, mask, background=background)
