"""Tests for shared contour geometry."""

import numpy as np
import pytest

from elephant_id.image.geometry import resample_polyline


def test_resample_polyline_preserves_endpoints_and_arc_length() -> None:
    """Unequal segments and a repeated vertex produce equally spaced samples."""
    points = np.asarray([[0, 0], [0, 1], [0, 1], [3, 1]], dtype=float)
    original = points.copy()
    result = resample_polyline(points, 5)
    np.testing.assert_array_equal(result, [[0, 0], [0, 1], [1, 1], [2, 1], [3, 1]])
    np.testing.assert_array_equal(points, original)


def test_resample_polyline_rejects_zero_length() -> None:
    """A contour with no extent cannot define an arc-length coordinate."""
    with pytest.raises(ValueError, match="zero-length"):
        resample_polyline(np.ones((3, 2)), 5)
