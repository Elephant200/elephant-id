"""
Bounding-box format conversions for model predictions.
"""


def center_to_xyxy(
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """
    Convert a center-format box to corner coordinates.

    Args:
        x: Bbox center x in pixels.
        y: Bbox center y in pixels.
        width: Bbox width in pixels.
        height: Bbox height in pixels.

    Returns:
        ``(x1, y1, x2, y2)`` with top-left and bottom-right corners in pixels.
    """
    return (
        x - width / 2,
        y - height / 2,
        x + width / 2,
        y + height / 2,
    )


def prediction_center_to_xyxy(prediction: dict) -> dict:
    """
    Return a copy of a prediction dict with ``x1``, ``y1``, ``x2``, ``y2`` instead of
    center-format ``x``, ``y``, ``width``, ``height``.

    Args:
        prediction: Detection dict using center-format ``x``, ``y``, ``width``, ``height`` keys.

    Returns:
        Copy of ``prediction`` with corner bbox keys and center keys removed.
    """
    x1, y1, x2, y2 = center_to_xyxy(
        float(prediction["x"]),
        float(prediction["y"]),
        float(prediction["width"]),
        float(prediction["height"]),
    )
    converted = {
        key: value
        for key, value in prediction.items()
        if key not in {"x", "y", "width", "height"}
    }
    converted["x1"] = x1
    converted["y1"] = y1
    converted["x2"] = x2
    converted["y2"] = y2
    return converted
