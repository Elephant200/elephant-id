"""Tear finding: ear contour -> 1-D tear-depth profile.

Input: the cut ear contour P (N x 2 px, ordered, anchor P[0] -> anchor
P[-1]; any point spacing of a few px works -- fixed-count resampling is
not required).

Output: TEAR_PROFILE_BINS signed values on x in [0, 1], where x is
normalized arclength along an estimate of the intact ear edge. The profile
reads as the ear unrolled flat: ~0 where intact, a positive bump per tear
(height = depth, area = size), and small negative dips where the contour
pokes outside the reference (shaved spurs / outward texture -- a useful
diagnostic, bounded by roughly the opening radius). Two photos of the same
ear produce bumps at the same x.

Pipeline (tunables in elephant_id.constants; lengths in units of S):
  1. S     = convex-hull arc length between the anchors
             (rotation- and tear-invariant scale)
  2. open  = morphological opening, radius TEAR_OPEN_FRAC * S
             (outward segmentation spurs cannot lift the reference)
  3. ref   = alpha hull of the opened contour, radius TEAR_ALPHA_FRAC * S
             (bridges tears, follows anatomical bays)
  4. scan  = signed nearest contour crossing along the reference's inward
             normals, divided by S
  5. clean = light gaussian smoothing, then zero the SEEK-uncoded ends
             (TEAR_TRIM_LO / TEAR_TRIM_HI)

The reasoning behind every choice: docs/tear-embedding.md. Production
entry point: EarFieldAnalyzer calls compute_tear_profile() per anchored ear; the
standalone research path lives in scripts/evaluate.py. Constants are
calibrated on a 17-photo pilot set; re-validate at scale.
"""
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.ndimage import gaussian_filter1d

from elephant_id.coding.ears.geometry import (
    alpha_shape,
    densify,
    ear_side_path,
    inward_normals,
    nearest_crossing,
    opened_contour,
)
from elephant_id.constants import (
    TEAR_ALPHA_FRAC,
    TEAR_OPEN_FRAC,
    TEAR_PROFILE_BINS,
    TEAR_SMOOTH_SIGMA,
    TEAR_TRIM_HI,
    TEAR_TRIM_LO,
)

# Tunables live in elephant_id.constants (TEAR_*); derived values here.
PROFILE_GRID = (np.arange(TEAR_PROFILE_BINS) + 0.5) / TEAR_PROFILE_BINS
_LO = int(TEAR_TRIM_LO * TEAR_PROFILE_BINS)
_HI = int(TEAR_TRIM_HI * TEAR_PROFILE_BINS)


@dataclass
class TearProfile:
    """Profile plus the scan geometry needed to draw it on the photo."""

    profile: np.ndarray    # (TEAR_PROFILE_BINS,) signed depth / S
    scale: float           # S, px
    reference: np.ndarray  # densified reference path, px
    origins: np.ndarray    # scan-ray origins on the reference, px
    normals: np.ndarray    # inward unit normals, one per bin


def hull_arclength(points: np.ndarray) -> float:
    """S: arc length of the convex hull between the anchors, px."""
    ring = np.asarray(shapely.MultiPoint(points).convex_hull.exterior.coords)[:-1]
    path = ear_side_path(ring, points[0], points[-1])
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def tear_profile(points: np.ndarray) -> TearProfile:
    """The pipeline, returning the profile with its scan geometry."""
    S = hull_arclength(points)  # 1. scale
    # TODO: Consider using the alpha shape length instead of the hull arclength for the scale
    src = opened_contour(points, TEAR_OPEN_FRAC * S)  # 2. opening
    shape = alpha_shape(src, TEAR_ALPHA_FRAC * S)  # 3. reference
    path = densify(ear_side_path(
        np.asarray(shape.exterior.coords)[:-1], src[0], src[-1]))
    origins, normals = inward_normals(path, shape, PROFILE_GRID)
    depth = nearest_crossing(origins, normals, points) / S  # 4. signed depth scan
    profile = gaussian_filter1d(depth, sigma=TEAR_SMOOTH_SIGMA)  # 5. cleanup
    profile[:_LO] = 0
    profile[-_HI:] = 0
    return TearProfile(profile=profile, scale=S, reference=path,
                       origins=origins, normals=normals)


def embed(points: np.ndarray) -> np.ndarray:
    """Margin polyline -> 1-D tear-depth profile (TEAR_PROFILE_BINS,)."""
    return tear_profile(points).profile
