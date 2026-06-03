"""
Module that computes integral curvature for an elephant ear.
"""

import numpy as np

from elephant_id.constants import DEFAULT_CURVATURE_RADII, DEFAULT_CURVATURE_WEIGHTS


def resample2d(points: np.ndarray, num_points: int) -> np.ndarray:
    """Resample a 2D polyline to evenly spaced points by arc length.

    Args:
        points: Ordered ``(n, 2)`` contour coordinates.
        num_points: Number of evenly spaced output points.

    Returns:
        A ``(num_points, 2)`` array sampled at uniform arc-length
            intervals.

    Raises:
        ValueError: If the contour has zero total length.
    """
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    arc_lengths = np.concatenate([[0], np.cumsum(distances)])
    if arc_lengths[-1] == 0:
        raise ValueError("Cannot resample a zero-length contour")

    sample_lengths = np.linspace(0, arc_lengths[-1], num_points)
    x = np.interp(sample_lengths, arc_lengths, points[:, 0])
    y = np.interp(sample_lengths, arc_lengths, points[:, 1])
    return np.column_stack([x, y])


def rotate(radians: float) -> np.ndarray:
    """Build a 3x3 homogeneous rotation matrix for a clockwise angle."""
    rotation = np.eye(3)
    rotation[0, 0] = np.cos(radians)
    rotation[1, 1] = np.cos(radians)
    rotation[0, 1] = np.sin(radians)
    rotation[1, 0] = -np.sin(radians)
    return rotation


def reorient(points: np.ndarray, theta: float, center: np.ndarray) -> np.ndarray:
    """Rotate points by ``theta`` about ``center``.

    Args:
        points: ``(n, 2)`` coordinates to rotate.
        theta: Rotation angle in radians.
        center: ``(2,)`` center of rotation.

    Returns:
        The rotated ``(n, 2)`` coordinates.
    """
    matrix = rotate(theta)
    points_translated = points - center
    points_augmented = np.hstack(
        (points_translated, np.ones((points.shape[0], 1)))
    )
    points_transformed = np.dot(matrix, points_augmented.T).T[:, :2]
    return points_transformed + center


def oriented_curvature(
    contour: np.ndarray,
    radii: np.ndarray = DEFAULT_CURVATURE_RADII,
    weights: np.ndarray = DEFAULT_CURVATURE_WEIGHTS,
) -> np.ndarray:
    """
    Compute weighted mean multi-scale integral curvature along a
    contour.

    For each contour point and radius, the curvature is the normalized
    area under the locally reoriented neighborhood enclosed by the
    circle of that radius, weighted by the given weights.

    Args:
        contour: Ordered ``(n, 2)`` contour coordinates.
        radii: Physical radii defining the integration scales.
        weights: Weights for each radius.

    Returns:
        A ``(n,)`` array of curvature values in ``[0, 1]``.
    """
    if len(weights) != len(radii):
        raise ValueError("Number of radii weights must match number of radii")

    curvatures = np.zeros((len(radii), contour.shape[0]), dtype=np.float32)

    for i, (x, y) in enumerate(contour):
        center = np.array([x, y])
        distances = ((contour - center) ** 2).sum(axis=1)
        inside = distances[:, np.newaxis] <= radii * radii

        for j, radius in enumerate(radii):
            curve = contour[inside[:, j]]
            if curve.shape[0] == 1:
                curv = 0.5
            else:
                normal = curve[-1] - curve[0]
                theta = np.arctan2(normal[1], normal[0])

                curve_p = reorient(curve, theta, center)
                center_p = np.squeeze(reorient(center[None], theta, center))

                lower = center_p - radius
                upper = center_p + radius
                curve_p = np.clip(curve_p, lower, upper)

                area = np.trapezoid(curve_p[:, 1] - lower[1], curve_p[:, 0], axis=0)
                curv = area / ((2 * radius) ** 2)

            curvatures[j, i] = curv

    return np.average(curvatures, axis=0, weights=weights) # shape: (n,)


def contour_max_dimension(contour: np.ndarray) -> float:
    """Return the largest side length of the contour's bounding box."""
    minimum = contour.min(axis=0)
    maximum = contour.max(axis=0)
    return float(np.max(maximum - minimum))
