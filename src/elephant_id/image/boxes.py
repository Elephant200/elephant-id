"""Bounding-box coordinate utilities."""


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
    valid box has ``x2 > x1`` and ``y2 > y1`` and must overlap the image. A box
    that is reversed, zero-area, or entirely outside the image is rejected
    rather than silently collapsed to a one-pixel box.

    Raises:
        ValueError: If the box has non-positive area, or does not intersect the
            image after clipping.
    """
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Box must have positive width and height: xyxy={(x1, y1, x2, y2)}"
        )

    ix1 = max(0, min(image_width - 1, round(x1)))
    iy1 = max(0, min(image_height - 1, round(y1)))
    ix2 = max(0, min(image_width, round(x2)))
    iy2 = max(0, min(image_height, round(y2)))

    if ix2 <= ix1 or iy2 <= iy1:
        raise ValueError(
            f"Box does not intersect image bounds "
            f"{(image_width, image_height)}: xyxy={(x1, y1, x2, y2)}"
        )
    return ix1, iy1, ix2, iy2
