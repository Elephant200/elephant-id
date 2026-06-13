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


def densify(path: np.ndarray, ds: float = 2.0) -> np.ndarray:
    """Resample a polyline so consecutive points are at most ``ds`` apart."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    out = []
    for i in range(len(path) - 1):
        n = int(seg[i] // ds) + 1
        out.append(path[i] + (path[i + 1] - path[i])
                   * np.linspace(0, 1, n, endpoint=False)[:, None])
    out.append(path[-1:])
    return np.vstack(out)


def ear_side_path(env: np.ndarray, a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    """Path from ``a0`` to ``a1`` along the ear side of an envelope ring:
    the half that deviates farther from the anchor chord."""
    i0 = int(np.argmin(np.linalg.norm(env - a0, axis=1)))
    env = np.roll(env, -i0, axis=0)
    i1 = int(np.argmin(np.linalg.norm(env - a1, axis=1)))
    half_a = env[:i1 + 1]
    half_b = np.vstack([env[i1:], env[:1]])[::-1]
    d = (a1 - a0) / np.linalg.norm(a1 - a0)

    def chord_dev(path: np.ndarray) -> float:
        if len(path) < 2:
            return -np.inf
        rel = path - a0
        return float(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]).max())

    return half_a if chord_dev(half_a) > chord_dev(half_b) else half_b


def opened_contour(P: np.ndarray, r: float) -> np.ndarray:
    """Run morphological opening on the contour.

    Envelopes must contain every point, so one outward segmentation spur
    would lift the reference and perturb readings far away; build the
    reference from the opened contour, measure against the original.
    """
    poly = shapely.Polygon(P)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    op = poly.buffer(-r).buffer(r)
    if op.is_empty:
        return P
    if op.geom_type == "MultiPolygon":
        op = max(op.geoms, key=lambda g: g.area)
    ring = np.asarray(op.exterior.coords)[:-1]
    return densify(ear_side_path(ring, P[0], P[-1]))


def alpha_shape(P: np.ndarray, radius: float) -> shapely.Polygon:
    """Rolling-disk (alpha) hull of an open contour: union of Delaunay
    triangles with circumradius below ``radius``.

    Concavities a ``radius`` disk cannot enter are bridged; wider ones are
    followed. The open end is closed with a densified anchor chord, and the
    interior is seeded with a radius/3 grid so the triangulation cannot
    fragment.
    """
    step = float(np.median(np.linalg.norm(np.diff(P, axis=0), axis=1)))
    chord_len = float(np.linalg.norm(P[0] - P[-1]))
    n = max(int(chord_len // step), 1)
    t = np.linspace(0, 1, n, endpoint=False)[1:, None]
    chord = P[-1] + (P[0] - P[-1]) * t

    g = radius / 3.0
    mn, mx = P.min(0), P.max(0)
    gx, gy = np.meshgrid(np.arange(mn[0], mx[0], g), np.arange(mn[1], mx[1], g))
    grid = np.c_[gx.ravel(), gy.ravel()]
    inside = mpath.Path(P).contains_points(grid)
    pts = np.vstack([P, chord, grid[inside]])

    tri = pts[Delaunay(pts).simplices]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    la = np.linalg.norm(b - c, axis=1)
    lb = np.linalg.norm(a - c, axis=1)
    lc = np.linalg.norm(a - b, axis=1)
    u, v = b - a, c - a
    area4 = 2.0 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    with np.errstate(divide="ignore"):
        circum_r = np.where(area4 > 0, la * lb * lc / area4, np.inf)

    shape = shapely.unary_union(shapely.polygons(tri[circum_r < radius]))
    if isinstance(shape, shapely.MultiPolygon):
        shape = max(shape.geoms, key=lambda s: s.area)
    if shape.is_empty:
        shape = shapely.MultiPoint(P).convex_hull
    return shape


def nearest_crossing(origins: np.ndarray, normals: np.ndarray,
                     poly: np.ndarray) -> np.ndarray:
    """Signed nearest crossing of ``poly`` along each ray line (min |t|).

    Positive = ahead of the origin along the normal, negative = behind it
    (the contour pokes outside the reference). Nearest, not
    first-nonnegative: an origin a hair inside the target must read the
    nearby crossing, not the far side of the ear.
    """
    A, E = poly[:-1], poly[1:] - poly[:-1]
    rel = A[None, :, :] - origins[:, None, :]
    denom = (normals[:, 0:1] * E[None, :, 1]
             - normals[:, 1:2] * E[None, :, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (rel[:, :, 0] * E[None, :, 1]
             - rel[:, :, 1] * E[None, :, 0]) / denom
        u = (rel[:, :, 0] * normals[:, 1:2]
             - rel[:, :, 1] * normals[:, 0:1]) / denom
    ok = (u >= 0) & (u <= 1) & np.isfinite(t)
    tt = np.where(ok, t, np.inf)
    near = tt[np.arange(len(tt)), np.argmin(np.abs(tt), axis=1)]
    near[~np.isfinite(near)] = 0.0
    return near


def inward_normals(path: np.ndarray, shape: shapely.Polygon,
                   grid: np.ndarray,
                   probe: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Ray origins and inward unit normals at each arclength in ``grid``.

    Inward is decided by probing ``probe`` px along the normal for
    containment in ``shape`` (a centroid test is ill-conditioned near the
    anchors).
    """
    s = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    s /= s[-1]
    origins = np.c_[np.interp(grid, s, path[:, 0]),
                    np.interp(grid, s, path[:, 1])]
    tan = np.gradient(path, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    normals = np.c_[np.interp(grid, s, tan[:, 1]),
                    np.interp(grid, s, -tan[:, 0])]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    p = origins + probe * normals
    normals[~shapely.contains_xy(shape, p[:, 0], p[:, 1])] *= -1
    return origins, normals
