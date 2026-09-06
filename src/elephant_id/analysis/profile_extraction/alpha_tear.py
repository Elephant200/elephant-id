"""AlphaTear profile extraction."""

from dataclasses import dataclass

import matplotlib.path as mpath
import numpy as np
import shapely
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import Delaunay

from elephant_id.analysis.ear_preparation import EarSide, PreparedEar
from elephant_id.analysis.tear_profile import TearProfile


@dataclass(frozen=True, slots=True)
class AlphaTearConfig:
    """Intentional research parameters controlling AlphaTear extraction."""

    contour_points: int = 1024
    profile_bins: int = 720
    trim_degrees: float = 5.0
    alpha_fraction: float = 0.35
    opening_fraction: float = 0.020
    smoothing_sigma: float = 2.0

    def __post_init__(self) -> None:
        """Validate the public research parameters."""
        if self.contour_points <= 1 or self.profile_bins <= 1:
            raise ValueError("AlphaTear sampling counts must be greater than one")
        if not 0 <= self.trim_degrees < 90:
            raise ValueError("trim_degrees must be in [0, 90)")
        if self.alpha_fraction <= 0:
            raise ValueError("alpha_fraction must be positive")
        if self.opening_fraction < 0 or self.smoothing_sigma < 0:
            raise ValueError("AlphaTear smoothing parameters must be non-negative")


@dataclass(frozen=True, slots=True)
class AlphaTearVersion:
    """One settled AlphaTear configuration and its stable producer slug."""

    slug: str
    config: AlphaTearConfig


DEFAULT_VERSION = AlphaTearVersion(
    slug="alpha-tear-v3",
    config=AlphaTearConfig(),
)

MULTISCALE_VERSIONS = tuple(
    AlphaTearVersion(
        slug=f"alpha-tear-v3-a{round(alpha * 100):03d}",
        config=AlphaTearConfig(alpha_fraction=alpha),
    )
    for alpha in (0.11, 0.22, 0.50, 1.10, 2.50, 5.00, 12.00)
)
"""Octave-spaced alpha-shape scales of the tuned representation.

A small `alpha_fraction` rolls a tight disk that follows single tears; a
large one approaches the convex envelope and shows overall margin shape.
"""


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    """Resample a two-dimensional polyline uniformly by arc length."""
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    arc_lengths = np.concatenate([[0], np.cumsum(distances)])
    if arc_lengths[-1] == 0:
        raise ValueError("Cannot resample a zero-length contour")
    sample_lengths = np.linspace(0, arc_lengths[-1], count)
    x = np.interp(sample_lengths, arc_lengths, points[:, 0])
    y = np.interp(sample_lengths, arc_lengths, points[:, 1])
    return np.column_stack([x, y])


def _densify(path: np.ndarray, max_spacing: float = 2.0) -> np.ndarray:
    """Resample a polyline so consecutive points do not exceed a spacing."""
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    pieces = []
    for index in range(len(path) - 1):
        count = int(segment_lengths[index] // max_spacing) + 1
        fractions = np.linspace(0, 1, count, endpoint=False)[:, None]
        pieces.append(path[index] + (path[index + 1] - path[index]) * fractions)
    pieces.append(path[-1:])
    return np.vstack(pieces)


def _ear_side_path(
    exterior: np.ndarray,
    upper_anchor: np.ndarray,
    lower_anchor: np.ndarray,
) -> np.ndarray:
    """Return the anchor path deviating farther from the anchor chord."""
    upper_index = int(np.argmin(np.linalg.norm(exterior - upper_anchor, axis=1)))
    exterior = np.roll(exterior, -upper_index, axis=0)
    lower_index = int(np.argmin(np.linalg.norm(exterior - lower_anchor, axis=1)))
    forward = exterior[: lower_index + 1]
    backward = np.vstack([exterior[lower_index:], exterior[:1]])[::-1]
    chord_direction = (lower_anchor - upper_anchor) / np.linalg.norm(
        lower_anchor - upper_anchor
    )

    def deviation(path: np.ndarray) -> float:
        """Return maximum absolute distance from the anchor chord."""
        if len(path) < 2:
            return -np.inf
        offsets = path - upper_anchor
        return float(
            np.abs(
                offsets[:, 0] * chord_direction[1]
                - offsets[:, 1] * chord_direction[0]
            ).max()
        )

    return forward if deviation(forward) > deviation(backward) else backward


def _opened_contour(contour: np.ndarray, radius: float) -> np.ndarray:
    """Morphologically open a contour and restore its original anchors."""
    polygon = shapely.Polygon(contour)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda geometry: geometry.area)
    opened = polygon.buffer(-radius).buffer(radius)
    if opened.is_empty:
        return contour
    if opened.geom_type == "MultiPolygon":
        opened = max(opened.geoms, key=lambda geometry: geometry.area)
    exterior = np.asarray(opened.exterior.coords)[:-1]
    path = _densify(_ear_side_path(exterior, contour[0], contour[-1]))
    path[0] = contour[0]
    path[-1] = contour[-1]
    return path


def _alpha_shape(contour: np.ndarray, radius: float) -> shapely.Polygon:
    """Return the rolling-disk alpha shape of an open ear contour."""
    median_spacing = float(
        np.median(np.linalg.norm(np.diff(contour, axis=0), axis=1))
    )
    chord_length = float(np.linalg.norm(contour[0] - contour[-1]))
    chord_point_count = max(int(chord_length // median_spacing), 1)
    fractions = np.linspace(0, 1, chord_point_count, endpoint=False)[1:, None]
    chord_points = contour[-1] + (contour[0] - contour[-1]) * fractions

    grid_spacing = radius / 3.0
    bbox_min, bbox_max = contour.min(0), contour.max(0)
    grid_x, grid_y = np.meshgrid(
        np.arange(bbox_min[0], bbox_max[0], grid_spacing),
        np.arange(bbox_min[1], bbox_max[1], grid_spacing),
    )
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    inside_contour = mpath.Path(contour).contains_points(grid_points)
    vertices = np.vstack([contour, chord_points, grid_points[inside_contour]])

    triangles = vertices[Delaunay(vertices).simplices]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    side_a = np.linalg.norm(b - c, axis=1)
    side_b = np.linalg.norm(a - c, axis=1)
    side_c = np.linalg.norm(a - b, axis=1)
    edge_ab, edge_ac = b - a, c - a
    four_times_area = 2.0 * np.abs(
        edge_ab[:, 0] * edge_ac[:, 1] - edge_ab[:, 1] * edge_ac[:, 0]
    )
    with np.errstate(divide="ignore"):
        circumradius = np.where(
            four_times_area > 0,
            side_a * side_b * side_c / four_times_area,
            np.inf,
        )

    hull = shapely.unary_union(shapely.polygons(triangles[circumradius < radius]))
    if isinstance(hull, shapely.MultiPolygon):
        hull = max(hull.geoms, key=lambda geometry: geometry.area)
    if hull.is_empty:
        hull = shapely.MultiPoint(contour).convex_hull
    return hull


def _nearest_crossing(
    origins: np.ndarray,
    normals: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:
    """Return signed nearest contour-crossing distances along local normals."""
    starts = contour[:-1]
    vectors = contour[1:] - contour[:-1]
    relative_starts = starts[None, :, :] - origins[:, None, :]
    denominator = (
        normals[:, 0:1] * vectors[None, :, 1]
        - normals[:, 1:2] * vectors[None, :, 0]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ray_distances = (
            relative_starts[:, :, 0] * vectors[None, :, 1]
            - relative_starts[:, :, 1] * vectors[None, :, 0]
        ) / denominator
        segment_positions = (
            relative_starts[:, :, 0] * normals[:, 1:2]
            - relative_starts[:, :, 1] * normals[:, 0:1]
        ) / denominator
    valid = (
        (segment_positions >= 0)
        & (segment_positions <= 1)
        & np.isfinite(ray_distances)
    )
    candidates = np.where(valid, ray_distances, np.inf)
    nearest = candidates[
        np.arange(len(candidates)),
        np.argmin(np.abs(candidates), axis=1),
    ]
    nearest[~np.isfinite(nearest)] = 0.0
    return nearest


def _furthest_ray_crossings(
    origin: np.ndarray,
    directions: np.ndarray,
    exterior: np.ndarray,
) -> np.ndarray:
    """Return the furthest forward boundary crossing for each ray."""
    vectors = np.roll(exterior, -1, axis=0) - exterior
    relative_starts = exterior[None, :, :] - origin
    denominator = (
        directions[:, 0:1] * vectors[None, :, 1]
        - directions[:, 1:2] * vectors[None, :, 0]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ray_distances = (
            relative_starts[:, :, 0] * vectors[None, :, 1]
            - relative_starts[:, :, 1] * vectors[None, :, 0]
        ) / denominator
        segment_positions = (
            relative_starts[:, :, 0] * directions[:, 1:2]
            - relative_starts[:, :, 1] * directions[:, 0:1]
        ) / denominator
    tolerance = 1e-9
    valid = (
        np.isfinite(ray_distances)
        & np.isfinite(segment_positions)
        & (ray_distances >= 0)
        & (segment_positions >= -tolerance)
        & (segment_positions <= 1 + tolerance)
    )
    furthest = np.where(valid, ray_distances, -np.inf).max(axis=1)
    crossings = np.full(directions.shape, np.nan, dtype=float)
    valid_rays = np.isfinite(furthest)
    crossings[valid_rays] = origin + furthest[valid_rays, None] * directions[valid_rays]
    return crossings


def _inward_normals(
    reference_path: np.ndarray,
    shape: shapely.Polygon,
    origins: np.ndarray,
) -> np.ndarray:
    """Return local unit normals pointing inward from reference origins."""
    tangents = np.gradient(reference_path, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    nearest_indices = np.argmin(
        np.sum((origins[:, None, :] - reference_path[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    normals = np.c_[tangents[nearest_indices, 1], -tangents[nearest_indices, 0]]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    probe_points = origins + 3.0 * normals
    normals[
        ~shapely.contains_xy(shape, probe_points[:, 0], probe_points[:, 1])
    ] *= -1
    return normals


def _polar_directions(
    upper: np.ndarray,
    lower: np.ndarray,
    side: EarSide,
    angles_degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the anchor midpoint and side-aware unit ray directions."""
    midpoint = (upper + lower) / 2.0
    upper_direction = upper - midpoint
    distance = float(np.linalg.norm(upper_direction))
    if distance == 0:
        raise ValueError("Ear landmarks must be distinct")
    upper_direction /= distance
    right_of_chord = np.array((-upper_direction[1], upper_direction[0]))
    margin_direction = right_of_chord if side == "left" else -right_of_chord
    radians = np.deg2rad(angles_degrees)[:, None]
    directions = (
        np.cos(radians) * upper_direction + np.sin(radians) * margin_direction
    )
    return midpoint, directions


def _depths(ear: PreparedEar, config: AlphaTearConfig) -> np.ndarray:
    """Compute normalized AlphaTear depths from one prepared ear."""
    contour = _resample(ear.contour, config.contour_points)
    radius = float(np.sqrt(2.0 * ear.cleaned_area / np.pi))
    opened = _opened_contour(contour, config.opening_fraction * radius)
    reference_hull = _alpha_shape(opened, config.alpha_fraction * radius)
    reference_boundary = np.asarray(reference_hull.exterior.coords)[:-1]
    reference_path = _densify(
        _ear_side_path(reference_boundary, opened[0], opened[-1])
    )

    angles = np.linspace(0.0, 180.0, config.profile_bins)
    coded = (angles > config.trim_degrees) & (
        angles < 180.0 - config.trim_degrees
    )
    upper = np.asarray(ear.original_landmarks[0], dtype=float)
    lower = np.asarray(ear.original_landmarks[1], dtype=float)
    midpoint, directions = _polar_directions(upper, lower, ear.inferred_side, angles[coded])
    origins = _furthest_ray_crossings(midpoint, directions, reference_boundary)
    valid_origins = np.isfinite(origins).all(axis=1)
    coded_depths = np.zeros(len(origins))
    if valid_origins.any():
        valid = origins[valid_origins]
        normals = _inward_normals(reference_path, reference_hull, valid)
        coded_depths[valid_origins] = _nearest_crossing(valid, normals, contour) / radius

    depths = np.zeros(config.profile_bins)
    smoothed = gaussian_filter1d(coded_depths, sigma=config.smoothing_sigma)
    smoothed[~valid_origins] = 0.0
    depths[coded] = smoothed
    return depths


class AlphaTearExtractor:
    """Extract normalized tear profiles with AlphaTear."""

    def __init__(
        self,
        configuration: AlphaTearConfig | AlphaTearVersion,
    ) -> None:
        """Configure settled or experimental extraction."""
        if isinstance(configuration, AlphaTearVersion):
            self.config = configuration.config
            self._producer_slug: str | None = configuration.slug
        else:
            self.config = configuration
            self._producer_slug = None

    @property
    def producer_slug(self) -> str | None:
        """Return the settled slug, or none for an experimental configuration."""
        return self._producer_slug

    def extract(self, ear: PreparedEar) -> TearProfile:
        """Extract one reusable tear profile from prepared ear geometry."""
        return TearProfile(_depths(ear, self.config))
