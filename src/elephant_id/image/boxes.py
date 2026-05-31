"""Bounding-box coordinate utilities."""


from math import ceil, floor


def center_to_xyxy(
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """Convert a center-format box to corner coordinates."""
    return (
        x - width / 2,
        y - height / 2,
        x + width / 2,
        y + height / 2,
    )


def clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Clip an xyxy box to image bounds.

    Coordinates use the half-open convention: ``x2``/``y2`` are exclusive, so a
    valid box has ``x2 > x1`` and ``y2 > y1`` and must overlap the image.

    Args:
        x1: Left edge of the box (inclusive).
        y1: Top edge of the box (inclusive).
        x2: Right edge of the box (exclusive).
        y2: Bottom edge of the box (exclusive).
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        The integer-valued box clipped to the image, expanded outward to whole
        pixels. The result always has positive area.

    Raises:
        ValueError: If the image has non-positive size, the box has non-positive
            area, or the box does not overlap the image.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Image must have positive width and height: {(image_width, image_height)}"
        )
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Box must have positive width and height: xyxy={(x1, y1, x2, y2)}"
        )
    if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
        raise ValueError(
            f"Box is outside image bounds "
            f"{(image_width, image_height)}: xyxy={(x1, y1, x2, y2)}"
        )

    ix1 = floor(max(0, x1))
    iy1 = floor(max(0, y1))
    ix2 = ceil(min(image_width, x2))
    iy2 = ceil(min(image_height, y2))

    return ix1, iy1, ix2, iy2
