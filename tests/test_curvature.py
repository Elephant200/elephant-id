import numpy as np
import pytest

from elephant_id.coding.curvature import oriented_curvature


def test_oriented_curvature_right_side_preserves_point_order():
    contour = np.array(
        [
            [0.0, 0.0],
            [-0.5, 1.0],
            [-0.8, 2.0],
            [-0.4, 3.0],
            [0.0, 4.0],
            [0.4, 5.0],
            [0.8, 6.0],
            [0.5, 7.0],
            [0.0, 8.0],
        ],
        dtype=np.float32,
    )
    radii = np.array([2.5], dtype=np.float32)
    weights = np.array([1.0], dtype=np.float32)

    right_curvature = oriented_curvature(contour, radii, weights, side="right")
    left_curvature = oriented_curvature(contour, radii, weights, side="left")
    expected = oriented_curvature(contour[::-1], radii, weights, side="left")[::-1]

    np.testing.assert_allclose(right_curvature, expected)
    assert not np.allclose(right_curvature, left_curvature)
    assert right_curvature.shape == (len(contour),)


def test_oriented_curvature_rejects_unknown_side():
    contour = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    radii = np.array([1.0], dtype=np.float32)
    weights = np.array([1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="side must be"):
        oriented_curvature(contour, radii, weights, side="middle")
