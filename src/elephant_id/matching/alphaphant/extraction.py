"""Tear-profile extraction with alpha shapes.

`AlphaShapeExtractor` makes one tear profile for each alpha scale.

The extractor makes the shared geometry one time for each ear:

1. Resample the ear contour to equal arc-length spacing.
2. Calculate the ear radius from the cleaned ear area.
3. Open the resampled contour morphologically.
4. Make one polar ray for each angle in the measured interval.

The extractor then does these steps for each alpha scale:

5. Build the alpha shape of the opened contour.
6. Find the intersection of each polar ray with the alpha-shape boundary
   that is farthest from the ray origin. This point is the crossing.
7. Turn the alpha-shape edge that each ray crosses by 90°, into the alpha
   shape. The result is the inward normal at the crossing.
8. Measure the signed distance from each crossing to the resampled contour,
   along the normal.
9. Divide each distance by the ear radius. The result is the depth.
10. Smooth the depths along the angles.
"""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

import matplotlib.path as mpath
import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import Delaunay

from elephant_id.domain import Photo
from elephant_id.image.boxes import BoundingBox
from elephant_id.preparation import EarSide, PreparedEar, ear_side_path


@dataclass(frozen=True, slots=True, eq=False)
class TearProfile:
    """The read-only depths of one ear at one alpha scale.

    The depths are at uniform angles from 0° to 180°, both ends included.
    Each depth is a fraction of the ear radius. A positive depth shows a tear
    whose ear contour is inside the alpha shape.

    Extraction measures the depths in the measured interval, from
    `start_degrees` to `end_degrees`, both ends excluded. The depths outside
    the measured interval are zero.

    Raises:
        ValueError: If the depths are empty, not finite, or not
            one-dimensional. Also if the measured interval is not a nonempty
            part of 0° to 180°.
    """

    depths: NDArray[np.float64]
    start_degrees: float
    end_degrees: float

    def __post_init__(self) -> None:
        """Validate the depths and the measured interval, and store a read-only copy of the depths."""
        depths = np.array(self.depths, dtype=np.float64, copy=True)
        if depths.ndim != 1 or len(depths) == 0:
            raise ValueError("Tear-profile depths must be a non-empty 1-D array")
        if not np.isfinite(depths).all():
            raise ValueError("Tear-profile depths must be finite")
        if not 0.0 <= self.start_degrees < self.end_degrees <= 180.0:
            raise ValueError("The tear-profile measured interval must be a nonempty part of 0° to 180°")
        depths.setflags(write=False)
        object.__setattr__(self, "depths", depths)


@dataclass(frozen=True, slots=True)
class AnalyzedEar:
    """One ear's tear profiles for each alpha scale, along with its source."""

    source_photo: Photo
    side: EarSide
    source_box: BoundingBox
    profiles_by_scale: tuple[TearProfile, ...]


@dataclass(frozen=True, slots=True)
class AnalyzedSightingEarPair:
    """The Alphaphant-analyzed counterpart of a SightingEarPair."""

    sighting_id: UUID
    left: AnalyzedEar
    right: AnalyzedEar


class TearProfileExtractor(Protocol):
    """Extract one tear profile for each alpha scale from one prepared ear."""

    @property
    def producer_slug(self) -> str | None:
        """Return the settled producer identity, or None for an experiment."""
        ...

    @property
    def alpha_scales(self) -> tuple[float, ...]:
        """Return the alpha scale of each extracted tear profile."""
        ...

    def extract(self, ear: PreparedEar) -> tuple[TearProfile, ...]:
        """Return one tear profile of signed, normalized depths for each alpha scale."""
        ...


@dataclass(frozen=True, slots=True)
class AlphaShapeExtractorConfig:
    """Settings for alpha-shape extraction geometry and sampling.

    The measured interval is from `trim_degrees` to `180 - trim_degrees`.

    Raises:
        ValueError: If sampling counts are at most one, trim is outside
            [0, 90), no alpha scale is given, an alpha scale is nonpositive,
            or opening or smoothing is negative.
    """

    contour_points: int = 1024
    profile_bins: int = 720
    trim_degrees: float = 5.0
    alpha_scales: tuple[float, ...] = (0.2, 1.0, 5.0)
    morph_opening_fraction: float = 0.020
    smoothing_sigma: float = 2.0

    def __post_init__(self) -> None:
        """Validate the public research parameters."""
        if self.contour_points <= 1 or self.profile_bins <= 1:
            raise ValueError("Extraction sampling counts must be greater than one")
        if not 0 <= self.trim_degrees < 90:
            raise ValueError("trim_degrees must be in [0, 90)")
        if not self.alpha_scales or min(self.alpha_scales) <= 0:
            raise ValueError("alpha_scales must contain at least one alpha scale, and each must be positive")
        if self.morph_opening_fraction < 0 or self.smoothing_sigma < 0:
            raise ValueError("Extraction smoothing parameters must be non-negative")


@dataclass(frozen=True, slots=True)
class AlphaShapeExtractorVersion:
    """One settled extractor configuration and its stable producer slug."""

    slug: str
    config: AlphaShapeExtractorConfig


DEFAULT_VERSION = AlphaShapeExtractorVersion(
    slug="alpha-tear-v1",
    config=AlphaShapeExtractorConfig(),
)


class AlphaShapeExtractor:
    """Extract normalized tear profiles with alpha shapes."""

    def __init__(
        self,
        configuration: AlphaShapeExtractorConfig | AlphaShapeExtractorVersion,
    ) -> None:
        """Configure settled or experimental extraction."""
        if isinstance(configuration, AlphaShapeExtractorVersion):
            self._config = configuration.config
            self._producer_slug: str | None = configuration.slug
        else:
            self._config = configuration
            self._producer_slug = None

    @property
    def producer_slug(self) -> str | None:
        """Return the settled slug, or none for an experimental configuration."""
        return self._producer_slug

    @property
    def alpha_scales(self) -> tuple[float, ...]:
        """Return the configured alpha scales, as fractions of the ear radius."""
        return self._config.alpha_scales

    def extract(self, ear: PreparedEar) -> tuple[TearProfile, ...]:
        """Return one tear profile for each alpha scale.

        The method makes the shared geometry (module steps 1 to 4) one time.
        For each alpha scale, the method builds the alpha shape (step 5) and
        measures the depths (steps 6 to 10). The method puts the depths on
        the grid from 0° to 180°. The depths outside the measured interval
        are zero.

        Args:
            ear: The prepared ear to extract tear profiles from.

        Returns:
            One tear profile for each configured alpha scale, in the same order.
        """
        config = self._config
        contour = _resample(ear.contour, config.contour_points)
        ear_radius = float(np.sqrt(2.0 * ear.cleaned_area / np.pi))
        opened_contour = _opened_contour(contour, config.morph_opening_fraction * ear_radius)

        angles = np.linspace(0.0, 180.0, config.profile_bins)
        start_degrees = config.trim_degrees
        end_degrees = 180.0 - config.trim_degrees
        measured_bins = (angles > start_degrees) & (angles < end_degrees)
        ray_origin, ray_directions = _polar_directions(
            np.asarray(ear.original_landmarks[0], dtype=float),
            np.asarray(ear.original_landmarks[1], dtype=float),
            ear.inferred_side,
            angles[measured_bins],
        )

        profiles = []
        for alpha_scale in config.alpha_scales:
            alpha_shape = _alpha_shape(opened_contour, alpha_scale * ear_radius)
            depths = np.zeros(config.profile_bins)
            depths[measured_bins] = _measure_depths(
                alpha_shape,
                contour,
                ear_radius,
                ray_origin,
                ray_directions,
                config.smoothing_sigma,
            )
            profiles.append(TearProfile(depths, start_degrees, end_degrees))
        return tuple(profiles)


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    """Resample a polyline to `count` points at equal arc-length spacing (step 1).

    Raises:
        ValueError: If the polyline has zero length.
    """
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    arc_lengths = np.concatenate([[0], np.cumsum(distances)])
    if arc_lengths[-1] == 0:
        raise ValueError("Cannot resample a zero-length contour")
    sample_lengths = np.linspace(0, arc_lengths[-1], count)
    x = np.interp(sample_lengths, arc_lengths, points[:, 0])
    y = np.interp(sample_lengths, arc_lengths, points[:, 1])
    return np.column_stack([x, y])


def _opened_contour(contour: np.ndarray, radius: float) -> np.ndarray:
    """Return the morphologically opened contour (step 3).

    The function does these steps:

    1. Morphologically open the contour by `radius`, removing narrow parts.
    2. If the result has more than one part, use the largest part.
    3. Cut the boundary to its ear-flap path between the two contour anchors.
    4. Add points until adjacent points are 2 px apart or less.
    5. Set the two ends of the path to the two contour anchors.

    If step 1 removes the full polygon, the function returns `contour`.

    Args:
        contour: The resampled contour.
        radius: The opening radius, in pixels.
    """
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
    path = _densify(ear_side_path(exterior, contour[0], contour[-1]), max_spacing=2.0)
    path[0] = contour[0]
    path[-1] = contour[-1]
    return path


def _densify(path: np.ndarray, max_spacing: float) -> np.ndarray:
    """Add points to a polyline until adjacent points are at most `max_spacing` apart.

    The output contains all the input points, in the same order.
    """
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    pieces = []
    for index in range(len(path) - 1):
        count = int(segment_lengths[index] // max_spacing) + 1
        fractions = np.linspace(0, 1, count, endpoint=False)[:, None]
        pieces.append(path[index] + (path[index + 1] - path[index]) * fractions)
    pieces.append(path[-1:])
    return np.vstack(pieces)


def _polar_directions(
    upper: np.ndarray,
    lower: np.ndarray,
    side: EarSide,
    angles_degrees: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the origin and the unit directions of the polar rays (step 4).

    The origin is the midpoint of the two landmarks. The ray angles
    are:

    - 0°: toward `upper`.
    - 90°: perpendicular to the landmark chord, toward the ear flap.
    - 180°: toward `lower`.

    The 90° direction changes with `side` so that one angle has the same
    anatomical position on a left ear and on a right ear.

    Raises:
        ValueError: If `upper` and `lower` are the same point.
    """
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


def _alpha_shape(contour: np.ndarray, radius: float) -> shapely.Polygon:
    """Return the alpha shape of an open ear contour for a disk of `radius` (step 5).

    The function does these steps:

    1. Make the vertices. The vertices are the contour points, points on the
       chord between the contour ends, and grid points inside the contour.
    2. Calculate the Delaunay triangulation of the vertices.
    3. Select the triangles whose circumradius is less than `radius`.
    4. Merge the selected triangles. If the result has more than one part,
       use the largest part.

    If no triangle is selected, the function returns the convex hull of the
    contour.

    Args:
        contour: The opened contour.
        radius: The disk radius, in pixels: the alpha scale times the ear radius.
    """
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


def _measure_depths(
    alpha_shape: shapely.Polygon,
    contour: np.ndarray,
    ear_radius: float,
    ray_origin: np.ndarray,
    ray_directions: np.ndarray,
    smoothing_sigma: float,
) -> np.ndarray:
    """Return the smoothed depth of each polar ray at one alpha scale (steps 6 to 10).

    If a ray does not cross the alpha shape, its depth is zero. The function
    sets the depth of these rays to zero after the smoothing.

    Args:
        alpha_shape: The alpha shape at this alpha scale.
        contour: The resampled contour. The function measures the depths to
            this contour.
        ear_radius: The ear radius, in pixels.
        ray_origin: The origin of the polar rays.
        ray_directions: The unit direction of each polar ray.
        smoothing_sigma: The standard deviation of the Gaussian smoothing, in
            bins.
    """
    crossings, normals = _alpha_shape_crossings(ray_origin, ray_directions, alpha_shape)
    crossed = np.isfinite(crossings).all(axis=1)
    depths = np.zeros(len(crossings))
    if crossed.any():
        depths[crossed] = (
            _contour_distances(crossings[crossed], normals[crossed], contour) / ear_radius
        )
    smoothed = gaussian_filter1d(depths, sigma=smoothing_sigma)
    smoothed[~crossed] = 0.0
    return smoothed


def _alpha_shape_crossings(
    ray_origin: np.ndarray,
    ray_directions: np.ndarray,
    alpha_shape: shapely.Polygon,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the crossing of each polar ray and the inward normal there (steps 6 and 7).

    Each ray starts at `ray_origin` and points away from it. The function does
    these steps:

    1. Find the crossing: the intersection of the ray with the alpha-shape
       boundary that is farthest from `ray_origin`.
    2. Turn the crossed boundary edge by 90° to make the normal. The winding
       order of the boundary sets the direction of the turn, so that the
       normal points into the alpha shape.

    Returns:
        The xy crossings and the inward unit normals, in ray order. The rows
        of a ray are NaN if the ray does not cross the alpha shape.
    """
    boundary = np.asarray(alpha_shape.exterior.coords)[:-1]
    edges = np.roll(boundary, -1, axis=0) - boundary
    distances, positions = _ray_segment_intersections(
        ray_origin[None, :], ray_directions, boundary, edges
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
    furthest = distances[np.arange(len(ray_directions)), crossed_edges]
    crossed = np.isfinite(furthest)

    crossings = np.full(ray_directions.shape, np.nan)
    crossings[crossed] = ray_origin + furthest[crossed, None] * ray_directions[crossed]
    crossed_edge_vectors = edges[crossed_edges[crossed]]
    inward = np.c_[-crossed_edge_vectors[:, 1], crossed_edge_vectors[:, 0]]
    if not alpha_shape.exterior.is_ccw:
        inward = -inward
    normals = np.full(ray_directions.shape, np.nan)
    normals[crossed] = inward / np.linalg.norm(inward, axis=1, keepdims=True)
    return crossings, normals


def _contour_distances(
    crossings: np.ndarray,
    normals: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:
    """Return the signed distance from each crossing to the contour along its normal, in pixels (step 8).

    The function searches the line through the crossing in the two directions.
    It uses the intersection with the contour that is nearest to the crossing.

    A positive distance is in the direction of the inward normal. Thus, the
    contour is inside the alpha shape, and the distance shows a tear. If the
    line does not cross the contour, the distance is zero.
    """
    distances, positions = _ray_segment_intersections(
        crossings, normals, contour[:-1], contour[1:] - contour[:-1]
    )
    on_contour = (positions >= 0) & (positions <= 1) & np.isfinite(distances)
    candidates = np.where(on_contour, distances, np.inf)
    nearest = candidates[
        np.arange(len(candidates)),
        np.argmin(np.abs(candidates), axis=1),
    ]
    nearest[~np.isfinite(nearest)] = 0.0
    return nearest


def _ray_segment_intersections(
    origins: np.ndarray,
    directions: np.ndarray,
    starts: np.ndarray,
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the intersection of each line with each segment.

    A line is `o + t d`. A segment is `s + u v`. The function calculates:

    - `t = cross(s - o, v) / cross(d, v)`, the signed distance along the line.
    - `u = cross(s - o, d) / cross(d, v)`, the position along the segment.

    In these equations, `cross(a, b) = a_x b_y - a_y b_x`. The callers select
    the valid intersections.

    Args:
        origins: The `(lines, 2)` line origins, or one `(1, 2)` origin for all lines.
        directions: The `(lines, 2)` unit line directions.
        starts: The `(segments, 2)` segment start points.
        vectors: The `(segments, 2)` vectors from each segment start to its end.

    Returns:
        Two `(lines, segments)` arrays, `t` and `u`. The value of `u` is 0 at
        the segment start and 1 at the segment end. If a line and a segment
        are parallel, their `t` and `u` are not finite.
    """
    relative_starts = starts[None, :, :] - origins[:, None, :]
    denominator = (
        directions[:, 0:1] * vectors[None, :, 1]
        - directions[:, 1:2] * vectors[None, :, 0]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        distances = (
            relative_starts[:, :, 0] * vectors[None, :, 1]
            - relative_starts[:, :, 1] * vectors[None, :, 0]
        ) / denominator
        positions = (
            relative_starts[:, :, 0] * directions[:, 1:2]
            - relative_starts[:, :, 1] * directions[:, 0:1]
        ) / denominator
    return distances, positions
