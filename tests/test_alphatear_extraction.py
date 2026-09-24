"""Geometric characterization of AlphaTear reference crossings and normals."""

import numpy as np
import shapely

from elephant_id.matching.alphaphant.extraction import (
    AlphaTearExtractor,
    _polar_directions,
    _reference_crossings,
    _signed_depths,
)
from elephant_id.preparation.ear import ear_flap_path


def test_standard_extractor_has_three_scales_and_distinct_cache_slug() -> None:
    """The settled representation cannot read older profile cache records."""
    extractor = AlphaTearExtractor()

    assert extractor.config.alpha_fractions == (0.35, 2.0, 10.0)
    assert extractor.producer_slug == "alpha-tear-v6-area-contour"


def test_opened_path_wraps_around_exterior() -> None:
    """An opened polygon keeps its ear flap across the array boundary."""
    exterior = np.asarray([(8, 5), (0, 10), (-1, 5), (0, 0)], dtype=float)

    path = ear_flap_path(exterior, np.asarray([0.1, 0.2]), np.asarray([0.1, 9.8]))

    np.testing.assert_array_equal(path, [(0, 0), (8, 5), (0, 10)])


def test_reference_crossings_point_inward_for_both_boundary_windings() -> None:
    """An intersected edge gives the same inward normal in either winding."""
    boundary = np.asarray([(-2, -1), (2, -1), (2, 1), (-2, 1)])
    directions = np.asarray([(1, 0), (0, 1), (-1, 0), (0, -1)])
    expected_origins = np.asarray([(2, 0), (0, 1), (-2, 0), (0, -1)])
    expected_normals = np.asarray([(-1, 0), (0, -1), (1, 0), (0, 1)])

    for exterior in (boundary, boundary[::-1]):
        origins, normals = _reference_crossings(
            np.zeros(2), directions, shapely.Polygon(exterior)
        )
        np.testing.assert_allclose(origins, expected_origins, atol=1e-12)
        np.testing.assert_allclose(normals, expected_normals, atol=1e-12)


def test_contour_crossing_keeps_the_nearest_signed_distance() -> None:
    """A contour crossing behind the origin can be closer than a forward crossing."""
    contour = np.asarray([(-1, -2), (-1, 2), (2, 2), (2, -2)])
    origins = np.zeros((2, 2))
    normals = np.asarray([(1.0, 0.0), (-1.0, 0.0)])

    distances = _signed_depths(origins, normals, contour)

    np.testing.assert_array_equal(distances, [-1.0, 1.0])


def test_polar_rays_follow_geometric_margin_direction() -> None:
    """Polar rays point toward the margin for each inferred ear side."""
    upper = np.asarray([0.0, 0.0])
    lower = np.asarray([0.0, 10.0])
    angles = np.asarray([0.0, 90.0, 180.0])

    _, margin_right = _polar_directions(upper, lower, "right", angles)
    _, margin_left = _polar_directions(upper, lower, "left", angles)

    np.testing.assert_allclose(
        margin_right, [[0, -1], [-1, 0], [0, 1]], atol=1e-15
    )
    np.testing.assert_allclose(
        margin_left, [[0, -1], [1, 0], [0, 1]], atol=1e-15
    )
    np.testing.assert_allclose(margin_right[[1]], -margin_left[[1]], atol=1e-15)
