"""Tear finding: an anchored ear contour -> angular tear-depth profile.

The profile has one value per angle from the upper anchor direction (0°) to
the lower anchor direction (180°). A ray from the anchor midpoint selects
the furthest alpha-reference crossing, then the local inward normal measures
depth to the original contour. Positive values are inward tears.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter1d

from elephant_id.coding.ears.geometry import (
    alpha_shape,
    densify,
    ear_side_path,
    furthest_ray_crossings,
    inward_normals_at_origins,
    nearest_crossing,
    opened_contour,
)
from elephant_id.constants import (
    TEAR_ALPHA_FRAC,
    TEAR_OPEN_FRAC,
    TEAR_PROFILE_BINS,
    TEAR_SMOOTH_SIGMA,
    TEAR_TRIM_DEGREES,
)


@dataclass
class TearProfile:
    """Angular profile plus the scan geometry needed to draw it."""

    profile: np.ndarray  # (TEAR_PROFILE_BINS,) signed depth / R
    scale: float  # Equal-area semicircle radius R, px
    reference: np.ndarray  # Densified alpha-reference path, px
    origins: np.ndarray  # Polar scan-ray origins; NaN in trimmed/missed bins, px
    normals: np.ndarray  # Inward unit normals; NaN in trimmed/missed bins


def polar_directions(
    upper_anchor: np.ndarray,
    lower_anchor: np.ndarray,
    side: Literal["left", "right"],
    angles_degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the anchor midpoint and unit ray directions for ear angles.

    The left/right side selects which semicircle runs between the anchors.
    Coordinates use the image convention where positive y points down.
    """
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown ear side: {side!r}")
    midpoint = (upper_anchor + lower_anchor) / 2.0
    upper_direction = upper_anchor - midpoint
    distance = float(np.linalg.norm(upper_direction))
    if distance == 0:
        raise ValueError("Ear anchors must be distinct")
    upper_direction /= distance
    right_of_chord = np.array((-upper_direction[1], upper_direction[0]))
    # The outer margin of the elephant's left ear is right of its anchor chord.
    margin_direction = right_of_chord if side == "left" else -right_of_chord
    radians = np.deg2rad(angles_degrees)[:, None]
    directions = np.cos(radians) * upper_direction + np.sin(radians) * margin_direction
    return midpoint, directions


def tear_profile(
    contour: np.ndarray,
    area: float,
    side: Literal["left", "right"],
    original_anchor_points: tuple[tuple[float, float], tuple[float, float]],
) -> TearProfile:
    """Return the angular tear profile for an upper-to-lower ear contour.

    ``original_anchor_points`` calibrate the polar coordinate before the
    anchors are snapped to the segmentation contour.
    """
    radius = float(np.sqrt(2.0 * area / np.pi))
    opened = opened_contour(contour, TEAR_OPEN_FRAC * radius)
    reference_hull = alpha_shape(opened, TEAR_ALPHA_FRAC * radius)
    reference_boundary = np.asarray(reference_hull.exterior.coords)[:-1]
    reference_path = densify(ear_side_path(reference_boundary, opened[0], opened[-1]))

    angles_degrees = np.linspace(0.0, 180.0, TEAR_PROFILE_BINS)
    coded_angle_mask = (
        (angles_degrees > TEAR_TRIM_DEGREES)
        & (angles_degrees < 180.0 - TEAR_TRIM_DEGREES)
    )
    original_upper_anchor = np.asarray(original_anchor_points[0], dtype=float)
    original_lower_anchor = np.asarray(original_anchor_points[1], dtype=float)
    anchor_midpoint, ray_directions = polar_directions(
        original_upper_anchor,
        original_lower_anchor,
        side,
        angles_degrees[coded_angle_mask],
    )
    coded_origins = furthest_ray_crossings(
        anchor_midpoint,
        ray_directions,
        reference_boundary,
    )
    valid_coded_origins = np.isfinite(coded_origins).all(axis=1)
    coded_normals = np.full_like(coded_origins, np.nan)
    coded_depths = np.zeros(len(coded_origins))
    if valid_coded_origins.any():
        valid_origins = coded_origins[valid_coded_origins]
        coded_normals[valid_coded_origins] = inward_normals_at_origins(
            reference_path,
            reference_hull,
            valid_origins,
        )
        coded_depths[valid_coded_origins] = (
            nearest_crossing(
                valid_origins,
                coded_normals[valid_coded_origins],
                contour,
            )
            / radius
        )

    origins = np.full((TEAR_PROFILE_BINS, 2), np.nan)
    normals = np.full((TEAR_PROFILE_BINS, 2), np.nan)
    profile = np.zeros(TEAR_PROFILE_BINS)
    origins[coded_angle_mask] = coded_origins
    normals[coded_angle_mask] = coded_normals
    coded_profile = gaussian_filter1d(
        coded_depths,
        sigma=TEAR_SMOOTH_SIGMA,
    )
    coded_profile[~valid_coded_origins] = 0.0
    profile[coded_angle_mask] = coded_profile
    return TearProfile(
        profile=profile,
        scale=radius,
        reference=reference_path,
        origins=origins,
        normals=normals,
    )


def embed(
    contour: np.ndarray,
    area: float,
    side: Literal["left", "right"],
    original_anchor_points: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    """Return the angular tear-depth profile for an anchored ear contour."""
    return tear_profile(contour, area, side, original_anchor_points).profile
