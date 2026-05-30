import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from elephant_id.visualize import draw_rle_mask_overlay


def _rle_from_mask(mask: np.ndarray) -> dict:
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


def test_draw_rle_mask_overlay_colors_masked_pixels():
    image = Image.new("RGB", (2, 1), (10, 20, 30))
    mask = np.array([[True, False]])

    output = draw_rle_mask_overlay(
        image,
        _rle_from_mask(mask),
        color=(110, 120, 130),
        alpha=1.0,
    )

    assert output.getpixel((0, 0)) == (110, 120, 130)
    assert output.getpixel((1, 0)) == (10, 20, 30)
