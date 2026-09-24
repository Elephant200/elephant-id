"""AlphaTear profile extraction."""

from dataclasses import dataclass

import matplotlib.path as mpath
import numpy as np
import shapely
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import Delaunay

from elephant_id.image.geometry import resample_polyline
from elephant_id.matching.alphaphant.profile import TearProfile
from elephant_id.preparation.ear import EarSide, PreparedEar


@dataclass(frozen=True, slots=True)
class AlphaTearConfig:
    """Numerical settings for one complete AlphaTear representation."""

    contour_points: int = 1024
    profile_bins: int = 720
    trim_degrees: float = 5.0
    alpha_fractions: tuple[float, ...] = (0.35, 2.0, 10.0)
    opening_fraction: float = 0.020
    smoothing_sigma: float = 2.0

    def __post_init__(self) -> None:
        """Validate the public research parameters."""
        if self.contour_points <= 1 or self.profile_bins <= 1:
            raise ValueError("AlphaTear sampling counts must be greater than one")
        if not 0 <= self.trim_degrees < 90:
            raise ValueError("trim_degrees must be in [0, 90)")
        if not self.alpha_fractions or any(
            alpha <= 0 for alpha in self.alpha_fractions
        ):
            raise ValueError("alpha_fractions must contain positive scales")
        if self.opening_fraction < 0 or self.smoothing_sigma < 0:
            raise ValueError("AlphaTear smoothing parameters must be non-negative")


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
                offsets[:, 0] * chord_direction[1] - offsets[:, 1] * chord_direction[0]
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
    median_spacing = float(np.median(np.linalg.norm(np.diff(contour, axis=0), axis=1)))
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


def _ray_segment_intersections(
    origins: np.ndarray,
    directions: np.ndarray,
    starts: np.ndarray,
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect every ray with every segment.

    Returns:
        Signed distances along each ray and fractional positions along each
        segment, both shaped `(rays, segments)`. Parallel pairs are not finite.
    """
    relative_starts = starts[None, :, :] - origins[:, None, :]
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
    return ray_distances, segment_positions


def _reference_crossings(
    origin: np.ndarray,
    directions: np.ndarray,
    reference: shapely.Polygon,
) -> tuple[np.ndarray, np.ndarray]:
    """Return where each ray last leaves the reference and that edge's inward normal.

    The inward normal is the crossed edge rotated a quarter turn toward the
    polygon interior, so no nearby-point search or inside test is needed.
    Rays that miss the reference get NaN rows.
    """
    exterior = np.asarray(reference.exterior.coords)[:-1]
    edges = np.roll(exterior, -1, axis=0) - exterior
    origins = np.broadcast_to(origin, directions.shape)
    distances, positions = _ray_segment_intersections(
        origins, directions, exterior, edges
    )
    tolerance = 1e-9
    forward = (
        np.isfinite(distances)
        & np.isfinite(positions)
        & (distances >= 0)
        & (positions >= -tolerance)
        & (positions <= 1 + tolerance)
    )
    distances = np.where(forward, distances, -np.inf)
    crossed_edges = np.argmax(distances, axis=1)
    furthest = distances[np.arange(len(directions)), crossed_edges]
    hit = np.isfinite(furthest)

    points = np.full(directions.shape, np.nan)
    points[hit] = origin + furthest[hit, None] * directions[hit]
    edge = edges[crossed_edges[hit]]
    inward = np.c_[-edge[:, 1], edge[:, 0]]
    if not reference.exterior.is_ccw:
        inward = -inward
    normals = np.full(directions.shape, np.nan)
    normals[hit] = inward / np.linalg.norm(inward, axis=1, keepdims=True)
    return points, normals


def _signed_depths(
    points: np.ndarray,
    normals: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:
    """Return the nearest contour crossing along each inward normal, in pixels.

    Positive values lie inside the reference, so a tear gives a positive depth.
    A normal that never meets the contour gives zero.
    """
    distances, positions = _ray_segment_intersections(
        points, normals, contour[:-1], np.diff(contour, axis=0)
    )
    on_contour = (positions >= 0) & (positions <= 1) & np.isfinite(distances)
    distances = np.where(on_contour, distances, np.inf)
    nearest = distances[np.arange(len(points)), np.argmin(np.abs(distances), axis=1)]
    nearest[~np.isfinite(nearest)] = 0.0
    return nearest


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
    directions = np.cos(radians) * upper_direction + np.sin(radians) * margin_direction
    return midpoint, directions


def _profiles(ear: PreparedEar, config: AlphaTearConfig) -> tuple[TearProfile, ...]:
    """Share contour preparation and polar rays across the extraction scales."""
    contour = resample_polyline(ear.contour, config.contour_points)
    radius = float(np.sqrt(2.0 * ear.cleaned_area / np.pi))
    opened = _opened_contour(contour, config.opening_fraction * radius)
    angles = np.linspace(0.0, 180.0, config.profile_bins)
    coded = (angles > config.trim_degrees) & (angles < 180.0 - config.trim_degrees)
    upper = np.asarray(ear.original_landmarks[0], dtype=float)
    lower = np.asarray(ear.original_landmarks[1], dtype=float)
    midpoint, directions = _polar_directions(
        upper, lower, ear.inferred_side, angles[coded]
    )
    profiles = []
    for alpha in config.alpha_fractions:
        reference = _alpha_shape(opened, alpha * radius)
        points, normals = _reference_crossings(midpoint, directions, reference)
        hit = np.isfinite(points).all(axis=1)
        coded_depths = np.zeros(len(points))
        if hit.any():
            coded_depths[hit] = (
                _signed_depths(points[hit], normals[hit], contour) / radius
            )

        depths = np.zeros(config.profile_bins)
        smoothed = gaussian_filter1d(coded_depths, sigma=config.smoothing_sigma)
        smoothed[~hit] = 0.0
        depths[coded] = smoothed
        profiles.append(TearProfile(depths))
    return tuple(profiles)


class AlphaTearExtractor:
    """Extract the complete multiscale representation from one prepared ear."""

    def __init__(
        self,
        configuration: AlphaTearConfig | None = None,
    ) -> None:
        """Use the fixed standard representation or an uncached research configuration."""
        self.config = configuration or AlphaTearConfig()
        self._producer_slug = (
            "alpha-tear-v6-area-contour" if configuration is None else None
        )

    @property
    def producer_slug(self) -> str | None:
        """Return the settled slug, or none for an experimental configuration."""
        return self._producer_slug

    def extract(self, ear: PreparedEar) -> tuple[TearProfile, ...]:
        """Return normalized profiles in the representation's fixed scale order."""
        return _profiles(ear, self.config)
