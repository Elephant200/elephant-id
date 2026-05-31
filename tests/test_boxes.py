import pytest

from elephant_id.image.boxes import center_to_xyxy, clip_xyxy


def test_center_to_xyxy_converts_center_box_to_corners():
    assert center_to_xyxy(
        x=10,
        y=20,
        width=6,
        height=8,
    ) == (7.0, 16.0, 13.0, 24.0)


def test_clip_xyxy_clips_to_image_bounds_and_preserves_area():
    assert clip_xyxy(
        x1=-5,
        y1=1.2,
        x2=10,
        y2=12,
        image_width=8,
        image_height=6,
    ) == (0, 1, 8, 6)


def test_clip_xyxy_rejects_zero_area_box():
    with pytest.raises(ValueError, match="positive width and height"):
        clip_xyxy(x1=4, y1=4, x2=4, y2=4, image_width=8, image_height=6)


def test_clip_xyxy_rejects_reversed_box():
    with pytest.raises(ValueError, match="positive width and height"):
        clip_xyxy(x1=6, y1=2, x2=3, y2=5, image_width=8, image_height=6)


def test_clip_xyxy_rejects_box_outside_image():
    with pytest.raises(ValueError, match="does not intersect"):
        clip_xyxy(x1=-20, y1=-20, x2=-5, y2=-5, image_width=8, image_height=6)
