"""Geometry operations shared by contour representations."""

import numpy as np


def resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    """Sample an ordered two-dimensional polyline uniformly by arc length.

    Raises:
        ValueError: If the polyline has zero total length.
    """
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    arc_lengths = np.concatenate([[0.0], np.cumsum(distances)])
    if arc_lengths[-1] == 0:
        raise ValueError("Cannot resample a zero-length contour")
    sample_lengths = np.linspace(0.0, arc_lengths[-1], count)
    x = np.interp(sample_lengths, arc_lengths, points[:, 0])
    y = np.interp(sample_lengths, arc_lengths, points[:, 1])
    return np.column_stack([x, y])
