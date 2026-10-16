"""CurvRank v2 elephant-ear catalog matcher."""

from collections.abc import Callable, Mapping
from functools import cache
from itertools import combinations

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from sklearn.neighbors import NearestNeighbors

from elephant_id.domain import SightingEarPair
from elephant_id.matching.protocol import CandidateKey, CandidateScores, MatchingError
from elephant_id.preparation import EarSide, PreparedEar

SCALES = np.array([0.02, 0.04, 0.06, 0.08], dtype=np.float32)
CROP_PADDING = 0.1
FINE_WIDTH = 1024
FINE_HEIGHT = 1024
CURV_LENGTH = 1024
NUM_KEYPOINTS = 32
FEAT_DIM = 32
LNBNN_K = 2
_SMOOTHING_SIGMA = 5.0
_EXTREMA_ORDER = 3
_RASTER_STEP = 0.25

Descriptors = tuple[NDArray[np.float32], ...]
"""One `(n, FEAT_DIM)` descriptor array for each curvature scale."""


def padded_crop(ear: PreparedEar) -> tuple[int, int, int, int]:
    """Return the plugin's padded, image-clipped ear crop as `(x0, y0, x1, y1)`."""
    box = ear.source_box
    width = box.x2 - box.x1
    height = box.y2 - box.y1
    image_width, image_height = ear.image_size
    pad_x = int(CROP_PADDING * width)
    pad_y = int(CROP_PADDING * height)
    return (
        max(0, box.x1 - pad_x),
        max(0, box.y1 - pad_y),
        min(image_width, box.x2 + pad_x),
        min(image_height, box.y2 + pad_y),
    )


def _pixel_path(points: NDArray[np.float64]) -> NDArray[np.int64]:
    """Rasterize a polyline into a minimal 8-connected integer pixel path.

    The result has the form of the plugin's diagonal A* path: each step
    moves to one of the eight neighbors, and no single diagonal move can skip
    a pixel of the path.
    """
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(lengths)])
    samples = np.linspace(0.0, arc[-1], int(np.ceil(arc[-1] / _RASTER_STEP)) + 1)
    dense = np.column_stack(
        [np.interp(samples, arc, points[:, 0]), np.interp(samples, arc, points[:, 1])]
    )
    path: list[tuple[int, int]] = []
    for x, y in np.rint(dense).astype(np.int64).tolist():
        while len(path) >= 2 and max(abs(x - path[-2][0]), abs(y - path[-2][1])) <= 1:
            path.pop()
        if not path or path[-1] != (x, y):
            path.append((x, y))
    return np.asarray(path, dtype=np.int64)


def fine_contour(ear: PreparedEar) -> NDArray[np.int64]:
    """Return the ear outline as the plugin's fine-frame contour.

    The plugin flips right-view images, so this function mirrors right ears
    inside their crop. It scales the crop to `FINE_WIDTH` and rasterizes the
    outline to an 8-connected path from the upper to the lower anchor. The
    points are in matrix `(row, column)` order because the elephant-ear
    configuration sets `transpose_dims`.
    """
    x0, y0, x1, _ = padded_crop(ear)
    points = np.array(ear.contour, dtype=np.float64)
    if ear.inferred_side == "right":
        points[:, 0] = (x0 + x1 - 1) - points[:, 0]
    ratio = FINE_WIDTH / (x1 - x0)
    fine = (points - np.array([x0, y0]) + 0.5) * ratio - 0.5
    return _pixel_path(fine)[:, ::-1]


def _reorient(
    points: NDArray[np.float64],
    theta: float,
    center: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate points by `-theta` about `center`, so direction `theta` becomes the x-axis."""
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return (points - center) @ rotation.T + center


def oriented_curvature(
    contour: NDArray[np.int64],
    radii: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute multi-scale integral curvature along a contour.

    For each point and radius, the contour inside the disk turns so that its
    chord is horizontal. The curvature is the area under it inside the
    chord-clipped `2r` square, divided by the area of that region. A straight
    contour scores `0.5`.

    Returns:
        A `(len(contour), len(radii))` float32 array.
    """
    curvature = np.zeros((len(contour), len(radii)), dtype=np.float32)
    for i, center in enumerate(contour):
        squared_distances = ((contour - center) ** 2).sum(axis=1)
        inside = squared_distances[:, np.newaxis] <= radii * radii
        for j, radius in enumerate(radii):
            curve = contour[inside[:, j]]
            if len(curve) == 1:
                curvature[i, j] = 0.5
                continue
            chord = curve[-1] - curve[0]
            theta = float(np.arctan2(chord[1], chord[0]))
            curve_p = _reorient(curve, theta, center)
            lower = center - radius
            upper = center + radius
            lower[0] = max(curve_p[:, 0].min(), lower[0])
            upper[0] = min(curve_p[:, 0].max(), upper[0])
            area = np.trapezoid(curve_p[:, 1] - lower[1], curve_p[:, 0])
            curvature[i, j] = area / np.prod(upper - lower)
    return curvature


def resample(values: NDArray[np.floating], length: int) -> NDArray[np.float64]:
    """Resample rows with the plugin's `resample1d` convention.

    The input rows span `[0, length]`, and the output reads them at
    `0, 1, ..., length - 1`. Thus, the output never reaches the last input row.
    """
    support = np.linspace(0.0, length, num=len(values))
    positions = np.arange(length)
    if values.ndim == 1:
        return np.interp(positions, support, values)
    return np.column_stack([np.interp(positions, support, column) for column in values.T])


def curvature_descriptors(curvature: NDArray[np.float32]) -> Descriptors:
    """Build per-scale L2-normalized descriptors from a curvature matrix.

    1. Resample the curvature to `CURV_LENGTH` rows and smooth it.
    2. At each scale, keep the `NUM_KEYPOINTS` local extrema farthest from
       `0.5`. The start and the end are always keypoints.
    3. For each keypoint pair, resample the unsmoothed curvature between them
       to `FEAT_DIM` values. This is one descriptor.

    A scale without local extrema has no descriptors.
    """
    curvature = resample(curvature, CURV_LENGTH)
    smoothed = gaussian_filter1d(curvature, _SMOOTHING_SIGMA, axis=0)
    maxima = np.vstack(argrelextrema(smoothed, np.greater, axis=0, order=_EXTREMA_ORDER)).T
    minima = np.vstack(argrelextrema(smoothed, np.less, axis=0, order=_EXTREMA_ORDER)).T
    # The plugin checks endpoint proximity on maxima followed by minima, unsorted.
    extrema = np.vstack((maxima, minima))
    per_scale: list[NDArray[np.float32]] = []
    for j in range(smoothed.shape[1]):
        keypoints = extrema[extrema[:, 1] == j, 0]
        if keypoints.size == 0:
            per_scale.append(np.empty((0, FEAT_DIM), dtype=np.float32))
            continue
        if keypoints[0] > 1:
            keypoints = np.hstack((0, keypoints))
        if keypoints[-1] < smoothed.shape[0] - 2:
            keypoints = np.hstack((keypoints, smoothed.shape[0] - 1))
        strength = np.abs(smoothed[keypoints, j] - 0.5)
        strength[0] = np.inf
        strength[-1] = np.inf
        keypoints = np.sort(keypoints[np.argsort(strength)[::-1][:NUM_KEYPOINTS]])
        pairs = list(combinations(keypoints, 2))
        descriptors = np.empty((len(pairs), FEAT_DIM), dtype=np.float32)
        for row, (start, end) in enumerate(pairs):
            feature = resample(curvature[start : end + 1, j], FEAT_DIM)
            descriptors[row] = feature / np.linalg.norm(feature)
        per_scale.append(descriptors)
    return tuple(per_scale)


def extract_descriptors(ear: PreparedEar) -> Descriptors:
    """Extract one CurvRank descriptor array per scale from a prepared ear."""
    radii = SCALES * max(FINE_WIDTH, FINE_HEIGHT)
    return curvature_descriptors(oriented_curvature(fine_contour(ear), radii))


def lnbnn_scores(
    query: NDArray[np.float32],
    catalog: NDArray[np.float32],
    labels: list[CandidateKey],
) -> dict[CandidateKey, float]:
    """Accumulate the LNBNN margins of one scale. A more negative margin is stronger.

    Each query descriptor finds its `LNBNN_K + 1` exact nearest catalog
    descriptors. Each candidate among the first `LNBNN_K` adds its closest
    distance minus the distance of the last neighbor.
    """
    scores: dict[CandidateKey, float] = {}
    if len(query) == 0 or len(catalog) == 0:
        return scores
    neighbors = NearestNeighbors(n_neighbors=min(LNBNN_K + 1, len(catalog)))
    distances, indices = neighbors.fit(catalog).kneighbors(query)
    for row_distances, row_indices in zip(distances, indices, strict=True):
        normalizer = float(row_distances[-1])
        seen: set[CandidateKey] = set()
        for distance, neighbor in zip(row_distances[:-1], row_indices[:-1], strict=True):
            label = labels[neighbor]
            if label not in seen:
                seen.add(label)
                scores[label] = scores.get(label, 0.0) + float(distance) - normalizer
    return scores


class CurvRank:
    """Score catalog candidates with CurvRank v2 descriptors and LNBNN.

    For each ear side, the matcher builds one exact index per scale from the
    same-side catalog descriptors, labeled by candidate. It adds the LNBNN
    sums of all scales and both sides, as the plugin does when it stacks the
    descriptors of a sighting into one query. It negates the total, so a
    larger candidate score is a stronger match.
    """

    def __init__(
        self,
        *,
        prepare_ears: Callable[[SightingEarPair], tuple[PreparedEar, PreparedEar]],
    ) -> None:
        """Compose shared left/right preparation with CurvRank description.

        Args:
            prepare_ears: Returns the left and right prepared ears of a sighting
                ear pair. Raises `MatchingError` when preparation fails.
        """
        self._prepare_ears = prepare_ears
        self._descriptors_for = cache(self._describe_pair)

    def describe(self, pair: SightingEarPair) -> tuple[Descriptors, Descriptors]:
        """Return the left and right descriptors of the pair.

        This instance stores the descriptors and returns them for the same pair.

        Raises:
            MatchingError: If preparation or description fails.
        """
        return self._descriptors_for(pair)

    def _describe_pair(self, pair: SightingEarPair) -> tuple[Descriptors, Descriptors]:
        """Prepare both ears and describe each one."""
        left, right = self._prepare_ears(pair)
        return self._describe_ear(left, "left"), self._describe_ear(right, "right")

    def _describe_ear(self, ear: PreparedEar, side: EarSide) -> Descriptors:
        """Return the descriptors of one ear, with failures tagged by photo and side."""
        try:
            return extract_descriptors(ear)
        except Exception as error:
            raise MatchingError(
                photo=ear.source_photo,
                side=side,
                stage="curvrank description",
                message=str(error),
            ) from error

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Return the negated LNBNN total of each candidate.

        Returns:
            One finite candidate score per supplied candidate; larger is stronger.

        Raises:
            RuntimeError: If a candidate has no reference sightings.
            MatchingError: If query or reference description fails.
        """
        query_sides = self.describe(query)
        reference_sides: dict[CandidateKey, list[tuple[Descriptors, Descriptors]]] = {}
        for candidate_key, reference_pairs in catalog.items():
            if not reference_pairs:
                raise RuntimeError(f"{candidate_key} has no reference sightings")
            reference_sides[candidate_key] = [self.describe(pair) for pair in reference_pairs]

        totals = dict.fromkeys(catalog, 0.0)
        for side, query_scales in enumerate(query_sides):
            for scale, query_descriptors in enumerate(query_scales):
                rows: list[NDArray[np.float32]] = []
                labels: list[CandidateKey] = []
                for candidate_key, references in reference_sides.items():
                    for reference in references:
                        descriptors = reference[side][scale]
                        rows.append(descriptors)
                        labels.extend([candidate_key] * len(descriptors))
                for candidate_key, score in lnbnn_scores(query_descriptors, np.vstack(rows), labels).items():
                    totals[candidate_key] += score
        logger.debug("CurvRank matched sighting {} against {} candidates", query.sighting_id, len(catalog))
        return {candidate_key: -score for candidate_key, score in totals.items()}
