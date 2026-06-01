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


def test_clip_xyxy_keeps_box_already_inside_image():
    assert clip_xyxy(x1=1, y1=2, x2=5, y2=6, image_width=8, image_height=6) == (
        1,
        2,
        5,
        6,
    )


def test_clip_xyxy_expands_subpixel_box_outward():
    assert clip_xyxy(
        x1=5.2,
        y1=5.2,
        x2=5.8,
        y2=5.8,
        image_width=100,
        image_height=100,
    ) == (5, 5, 6, 6)


def test_clip_xyxy_rejects_zero_area_box():
    with pytest.raises(ValueError, match="positive width and height"):
        clip_xyxy(x1=4, y1=4, x2=4, y2=4, image_width=8, image_height=6)


def test_clip_xyxy_rejects_reversed_box():
    with pytest.raises(ValueError, match="positive width and height"):
        clip_xyxy(x1=6, y1=2, x2=3, y2=5, image_width=8, image_height=6)


@pytest.mark.parametrize(
    "xyxy",
    [
        (float("nan"), 1, 4, 4),
        (1, float("nan"), 4, 4),
        (1, 1, float("nan"), 4),
        (1, 1, 4, float("nan")),
        (float("-inf"), 1, 4, 4),
        (1, float("-inf"), 4, 4),
        (1, 1, float("inf"), 4),
        (1, 1, 4, float("inf")),
    ],
)
def test_clip_xyxy_rejects_non_finite_coordinates(xyxy):
    with pytest.raises(ValueError, match="finite"):
        clip_xyxy(*xyxy, image_width=8, image_height=6)


def test_clip_xyxy_rejects_box_outside_image():
    with pytest.raises(ValueError, match="outside image bounds"):
        clip_xyxy(x1=-20, y1=-20, x2=-5, y2=-5, image_width=8, image_height=6)


@pytest.mark.parametrize(
    "xyxy",
    [
        (-5, 0, 0, 4),  # x2 touches left edge (exclusive -> no overlap)
        (8, 0, 12, 4),  # x1 sits on right edge (exclusive -> no overlap)
        (0, -5, 4, 0),  # y2 touches top edge
        (0, 6, 4, 10),  # y1 sits on bottom edge
    ],
)
def test_clip_xyxy_rejects_edge_touching_box(xyxy):
    with pytest.raises(ValueError, match="outside image bounds"):
        clip_xyxy(*xyxy, image_width=8, image_height=6)


@pytest.mark.parametrize(
    "image_width,image_height",
    [(0, 6), (8, 0), (-1, 6), (8, -1)],
)
def test_clip_xyxy_rejects_nonpositive_image_size(image_width, image_height):
    with pytest.raises(ValueError, match="positive width and height"):
        clip_xyxy(
            x1=1,
            y1=1,
            x2=4,
            y2=4,
            image_width=image_width,
            image_height=image_height,
        )
