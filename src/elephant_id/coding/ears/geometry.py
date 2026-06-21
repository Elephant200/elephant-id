"""Ear contour geometry.

Primitives shared by ear feature finders. Curves are ordered anchor-to-anchor
ear contours.
"""
import matplotlib.path as mpath
import numpy as np
import shapely
from scipy.spatial import Delaunay


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


def densify(path: np.ndarray, max_spacing: float = 2.0) -> np.ndarray:
    """Resample a polyline so consecutive points are at most ``max_spacing`` apart."""
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    pieces = []
    for i in range(len(path) - 1):
        count = int(segment_lengths[i] // max_spacing) + 1
        fractions = np.linspace(0, 1, count, endpoint=False)[:, None]
        pieces.append(path[i] + (path[i + 1] - path[i]) * fractions)
    pieces.append(path[-1:])
    return np.vstack(pieces)


def ear_side_path(
    exterior_boundary: np.ndarray,
    upper_anchor: np.ndarray,
    lower_anchor: np.ndarray,
) -> np.ndarray:
    """Path from ``upper_anchor`` to ``lower_anchor`` along the ear side of a
    boundary: the half that deviates farther from the chord between them."""
    upper_index = int(np.argmin(np.linalg.norm(exterior_boundary - upper_anchor, axis=1)))
    exterior_boundary = np.roll(exterior_boundary, -upper_index, axis=0)
    lower_index = int(np.argmin(np.linalg.norm(exterior_boundary - lower_anchor, axis=1)))
    forward_half = exterior_boundary[:lower_index + 1]
    backward_half = np.vstack([exterior_boundary[lower_index:], exterior_boundary[:1]])[::-1]
    chord = lower_anchor - upper_anchor
    chord_direction = chord / np.linalg.norm(chord)

    def chord_deviation(path: np.ndarray) -> float:
        if len(path) < 2:
            return -np.inf
        offsets = path - upper_anchor
        return float(np.abs(
            offsets[:, 0] * chord_direction[1] - offsets[:, 1] * chord_direction[0]
        ).max())

    if chord_deviation(forward_half) > chord_deviation(backward_half):
        return forward_half
    return backward_half


def opened_contour(contour: np.ndarray, radius: float) -> np.ndarray:
    """Morphological opening of the contour as a densified ear-side path.

    The original anchors are restored so opening cannot move the polar frame.
    """
    polygon = shapely.Polygon(contour)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
        if polygon.geom_type == "MultiPolygon":
            polygon = max(polygon.geoms, key=lambda geom: geom.area)
    opened_polygon = polygon.buffer(-radius).buffer(radius)
    if opened_polygon.is_empty:
        return contour
    if opened_polygon.geom_type == "MultiPolygon":
        opened_polygon = max(opened_polygon.geoms, key=lambda geom: geom.area)
    exterior_boundary = np.asarray(opened_polygon.exterior.coords)[:-1]
    opened_path = densify(ear_side_path(exterior_boundary, contour[0], contour[-1]))
    opened_path[0] = contour[0]
    opened_path[-1] = contour[-1]
    return opened_path


def alpha_shape(contour: np.ndarray, radius: float) -> shapely.Polygon:
    """Rolling-disk (alpha) hull of an open contour, closed across the anchor chord.

    Concavities a ``radius`` disk cannot enter are bridged; wider ones are followed.
    """
    median_spacing = float(np.median(np.linalg.norm(np.diff(contour, axis=0), axis=1)))
    chord_length = float(np.linalg.norm(contour[0] - contour[-1]))
    chord_point_count = max(int(chord_length // median_spacing), 1)
    fractions = np.linspace(0, 1, chord_point_count, endpoint=False)[1:, None]
    chord_points = contour[-1] + (contour[0] - contour[-1]) * fractions

    # Seed the interior so the triangulation cannot fragment.
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
            four_times_area > 0, side_a * side_b * side_c / four_times_area, np.inf
        )

    hull = shapely.unary_union(shapely.polygons(triangles[circumradius < radius]))
    if isinstance(hull, shapely.MultiPolygon):
        hull = max(hull.geoms, key=lambda geom: geom.area)
    if hull.is_empty:
        hull = shapely.MultiPoint(contour).convex_hull
    return hull


def nearest_crossing(origins: np.ndarray, normals: np.ndarray,
                     contour: np.ndarray) -> np.ndarray:
    """Signed distance from each origin to the nearest ``contour`` crossing along its normal.

    Positive is ahead of the origin (inward), negative behind it (the contour
    pokes outside the reference). Origins whose normal never crosses read zero.
    """
    segment_starts = contour[:-1]
    segment_vectors = contour[1:] - contour[:-1]
    relative_starts = segment_starts[None, :, :] - origins[:, None, :]
    denominator = (
        normals[:, 0:1] * segment_vectors[None, :, 1]
        - normals[:, 1:2] * segment_vectors[None, :, 0]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ray_distances = (
            relative_starts[:, :, 0] * segment_vectors[None, :, 1]
            - relative_starts[:, :, 1] * segment_vectors[None, :, 0]
        ) / denominator
        segment_positions = (
            relative_starts[:, :, 0] * normals[:, 1:2]
            - relative_starts[:, :, 1] * normals[:, 0:1]
        ) / denominator
    valid_intersections = (
        (segment_positions >= 0)
        & (segment_positions <= 1)
        & np.isfinite(ray_distances)
    )
    candidate_distances = np.where(valid_intersections, ray_distances, np.inf)
    # Nearest by |distance|, so an origin a hair inside reads the near wall.
    nearest_distances = candidate_distances[
        np.arange(len(candidate_distances)),
        np.argmin(np.abs(candidate_distances), axis=1),
    ]
    nearest_distances[~np.isfinite(nearest_distances)] = 0.0
    return nearest_distances


def furthest_ray_crossings(
    origin: np.ndarray,
    directions: np.ndarray,
    exterior_boundary: np.ndarray,
) -> np.ndarray:
    """Return the furthest forward boundary crossing for each ray direction.

    Raises:
        ValueError: If a ray has no forward intersection.
    """
    segment_vectors = np.roll(exterior_boundary, -1, axis=0) - exterior_boundary
    relative_starts = exterior_boundary[None, :, :] - origin
    denominator = (
        directions[:, 0:1] * segment_vectors[None, :, 1]
        - directions[:, 1:2] * segment_vectors[None, :, 0]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ray_distances = (
            relative_starts[:, :, 0] * segment_vectors[None, :, 1]
            - relative_starts[:, :, 1] * segment_vectors[None, :, 0]
        ) / denominator
        segment_positions = (
            relative_starts[:, :, 0] * directions[:, 1:2]
            - relative_starts[:, :, 1] * directions[:, 0:1]
        ) / denominator
    tolerance = 1e-9
    valid_intersections = (
        np.isfinite(ray_distances)
        & np.isfinite(segment_positions)
        & (ray_distances >= 0)
        & (segment_positions >= -tolerance)
        & (segment_positions <= 1 + tolerance)
    )
    furthest_distances = np.where(valid_intersections, ray_distances, -np.inf).max(axis=1)
    if not np.isfinite(furthest_distances).all():
        missing = np.flatnonzero(~np.isfinite(furthest_distances)).tolist()
        raise ValueError(f"Polar tear-profile rays miss the alpha reference: {missing}")
    return origin + furthest_distances[:, None] * directions


def inward_normals_at_origins(
    reference_path: np.ndarray,
    shape: shapely.Polygon,
    origins: np.ndarray,
    probe: float = 3.0,
) -> np.ndarray:
    """Inward unit normals at each origin, from the nearest ``reference_path`` tangent.

    Inward points into ``shape``.
    """
    tangents = np.gradient(reference_path, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    nearest_path_indices = np.argmin(
        np.sum((origins[:, None, :] - reference_path[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    normals = np.c_[
        tangents[nearest_path_indices, 1],
        -tangents[nearest_path_indices, 0],
    ]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    # Flip normals whose probe lands outside; a centroid test is
    # ill-conditioned near the anchors.
    probe_points = origins + probe * normals
    normals[~shapely.contains_xy(shape, probe_points[:, 0], probe_points[:, 1])] *= -1
    return normals
