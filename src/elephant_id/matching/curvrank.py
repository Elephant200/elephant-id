"""CurvRank v2 baseline catalog matcher over prepared ear contours.

Ports the CurvRank v2 integral-curvature descriptor and LNBNN scoring
(reference: `curvrank_ref/curv.py` and `curvrank_ref/functional.py`)
onto the repo's prepared-ear seam. Contours come from
the shared ear-preparation callable, descriptors are matched per ear side, and
exact nearest-neighbor search replaces the reference Annoy index.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache, partial
from itertools import combinations

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from sklearn.neighbors import NearestNeighbors

from elephant_id.analysis import EarSide, PreparedEar
from elephant_id.domain import SightingEarPair
from elephant_id.matching.protocol import CandidateKey, CandidateScores

_SMOOTHING_SIGMA = 5.0
_EXTREMA_ORDER = 3


@dataclass(frozen=True, slots=True)
class CurvRankConfig:
    """Research parameters for CurvRank descriptor extraction and scoring.

    Defaults match the CurvRank v2 example workflow. `scales` are
    fractions of the contour's largest bounding-box side used as
    integral-curvature disk radii, and `lnbnn_k` is the neighbor count
    whose `k + 1`-th distance normalizes each LNBNN contribution.
    """

    curv_length: int = 1024
    scales: tuple[float, ...] = (0.04, 0.06, 0.08, 0.10)
    num_keypoints: int = 32
    feat_dim: int = 32
    lnbnn_k: int = 2

    def __post_init__(self) -> None:
        """Validate that every parameter is positive.

        Raises:
            ValueError: If any count is non-positive or any scale is
                missing or non-positive.
        """
        for name in ("curv_length", "num_keypoints", "feat_dim", "lnbnn_k"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.scales or any(scale <= 0 for scale in self.scales):
            raise ValueError("scales must be a non-empty tuple of positive fractions")


def _resample_polyline(
    points: NDArray[np.float64],
    num_points: int,
) -> NDArray[np.float64]:
    """Resample an ordered polyline to evenly spaced points by arc length.

    Raises:
        ValueError: If the polyline has zero total length.
    """
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    arc_lengths = np.concatenate([[0.0], np.cumsum(distances)])
    if arc_lengths[-1] == 0:
        raise ValueError("Cannot resample a zero-length contour")
    sample_lengths = np.linspace(0.0, arc_lengths[-1], num_points)
    x = np.interp(sample_lengths, arc_lengths, points[:, 0])
    y = np.interp(sample_lengths, arc_lengths, points[:, 1])
    return np.column_stack([x, y])


def _resample_signal(values: NDArray[np.float64], num_samples: int) -> NDArray[np.float64]:
    """Resample a one-dimensional signal to `num_samples` values."""
    positions = np.linspace(0.0, 1.0, num_samples)
    support = np.linspace(0.0, 1.0, len(values))
    return np.interp(positions, support, values)


def _reorient(
    points: NDArray[np.float64],
    theta: float,
    center: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate points clockwise by `theta` radians about `center`."""
    rotation = np.array(
        [
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ]
    )
    return (points - center) @ rotation.T + center


def _oriented_curvature(
    contour: NDArray[np.float64],
    radii: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute multi-scale oriented integral curvature along a contour.

    For each contour point and radius, the curvature is the fraction of
    the local chord-aligned bounding region of the disk that lies under
    the contour, following the reference `oriented_curvature`. Values
    near `0.5` indicate a locally straight contour.

    Returns:
        A `(len(contour), len(radii))` array of curvature values.
    """
    curvature = np.full((len(contour), len(radii)), 0.5, dtype=np.float64)
    squared_radii = radii * radii
    for i, center in enumerate(contour):
        squared_distances = np.sum((contour - center) ** 2, axis=1)
        inside = squared_distances[:, np.newaxis] <= squared_radii
        for j, radius in enumerate(radii):
            curve = contour[inside[:, j]]
            if len(curve) == 1:
                continue
            chord = curve[-1] - curve[0]
            theta = float(np.arctan2(chord[1], chord[0]))
            curve_p = _reorient(curve, theta, center)
            center_p = _reorient(center[np.newaxis], theta, center)[0]
            lower = center_p - radius
            upper = center_p + radius
            lower[0] = max(float(curve_p[:, 0].min()), lower[0])
            upper[0] = min(float(curve_p[:, 0].max()), upper[0])
            region_area = float(np.prod(upper - lower))
            if region_area <= 0:
                continue
            area = float(np.trapezoid(curve_p[:, 1] - lower[1], curve_p[:, 0]))
            curvature[i, j] = area / region_area
    return curvature


def _ear_curvature(
    contour: NDArray[np.float64],
    radii: NDArray[np.float64],
    side: EarSide,
) -> NDArray[np.float64]:
    """Compute curvature with left-ear handedness.

    Right-ear contours are mirrored relative to left ears, so their
    curvature is computed on the reversed traversal and re-reversed to
    keep the upper-to-lower point order, mirroring the reference
    left-view flip.
    """
    if side == "right":
        return _oriented_curvature(contour[::-1], radii)[::-1]
    return _oriented_curvature(contour, radii)


def _keypoint_indices(
    smoothed: NDArray[np.float64],
    num_keypoints: int,
) -> NDArray[np.intp]:
    """Select keypoint indices from one smoothed curvature scale.

    Local extrema are ranked by their magnitude around the straight-line
    value `0.5`; both endpoints are always included and the strongest
    `num_keypoints` are returned sorted by contour index. An empty array
    is returned when the scale has no local extrema, matching the
    reference behavior.
    """
    maxima = argrelextrema(smoothed, np.greater, order=_EXTREMA_ORDER)[0]
    minima = argrelextrema(smoothed, np.less, order=_EXTREMA_ORDER)[0]
    extrema = np.sort(np.concatenate([maxima, minima])).astype(np.intp)
    if extrema.size == 0:
        return extrema
    if extrema[0] > 1:
        extrema = np.concatenate([[np.intp(0)], extrema])
    if extrema[-1] < len(smoothed) - 2:
        extrema = np.concatenate([extrema, [np.intp(len(smoothed) - 1)]])
    magnitude = np.abs(smoothed[extrema] - 0.5)
    magnitude[0] = np.inf
    magnitude[-1] = np.inf
    strongest = np.argsort(magnitude)[::-1][:num_keypoints]
    return np.sort(extrema[strongest])


def _scale_descriptors(
    curvature: NDArray[np.float64],
    keypoints: NDArray[np.intp],
    feat_dim: int,
) -> NDArray[np.float32]:
    """Build L2-normalized descriptors for every keypoint pair at one scale."""
    pairs = list(combinations(keypoints.tolist(), 2))
    descriptors = np.empty((len(pairs), feat_dim), dtype=np.float32)
    for row, (start, end) in enumerate(pairs):
        section = _resample_signal(curvature[start : end + 1], feat_dim)
        descriptors[row] = section / np.linalg.norm(section)
    return descriptors


def extract_descriptors(
    ear: PreparedEar,
    config: CurvRankConfig,
) -> tuple[NDArray[np.float32], ...]:
    """Extract per-scale CurvRank descriptors from one prepared ear.

    The contour is resampled to `config.curv_length` points by arc
    length, oriented integral curvature is computed with disk radii of
    `scale * max bounding-box side`, and each scale contributes one
    `(n_descriptors, config.feat_dim)` float32 array of L2-normalized
    curvature sections between keypoint pairs.
    """
    contour = _resample_polyline(
        np.asarray(ear.contour, dtype=np.float64),
        config.curv_length,
    )
    max_dimension = float(np.max(contour.max(axis=0) - contour.min(axis=0)))
    radii = np.asarray(config.scales, dtype=np.float64) * max_dimension
    curvature = _ear_curvature(contour, radii, ear.inferred_side)
    smoothed = gaussian_filter1d(curvature, _SMOOTHING_SIGMA, axis=0)
    per_scale: list[NDArray[np.float32]] = []
    for j in range(len(config.scales)):
        keypoints = _keypoint_indices(smoothed[:, j], config.num_keypoints)
        if keypoints.size < 2:
            per_scale.append(np.empty((0, config.feat_dim), dtype=np.float32))
        else:
            per_scale.append(
                _scale_descriptors(curvature[:, j], keypoints, config.feat_dim)
            )
    return tuple(per_scale)


class CurvRankMatcher:
    """Score catalog candidates with CurvRank descriptors and LNBNN.

    Implements the `CatalogMatcher` protocol. Per ear side and scale,
    one exact nearest-neighbor index is built over all catalog
    descriptors and every query descriptor contributes the reference
    LNBNN margin `dist(nearest in candidate) - dist(k + 1-th neighbor)`
    to the candidates in its top `k`. The reference convention is
    more-negative-is-stronger; this matcher negates the accumulated
    sums so that larger output scores indicate stronger matches.
    """

    def __init__(
        self,
        *,
        prepare_ears: Callable[[SightingEarPair], tuple[PreparedEar, PreparedEar]],
        config: CurvRankConfig | None = None,
    ) -> None:
        """Initialize the matcher with shared ear preparation."""
        self._config = config if config is not None else CurvRankConfig()
        self._prepare = cache(prepare_ears)
        self._describe = cache(partial(extract_descriptors, config=self._config))

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Return one similarity score per catalog candidate.

        Each side is scored independently with LNBNN over that side's
        catalog descriptors; the final score is the mean of the left
        and right side scores.

        Raises:
            RuntimeError: If a candidate has no catalog evidence.
        """
        query_left, query_right = self._prepare(query)
        if not catalog:
            return {}
        left_evidence: dict[CandidateKey, tuple[PreparedEar, ...]] = {}
        right_evidence: dict[CandidateKey, tuple[PreparedEar, ...]] = {}
        for candidate_key, evidence in catalog.items():
            if not evidence:
                raise RuntimeError(f"{candidate_key} has no catalog evidence")
            prepared = tuple(self._prepare(pair) for pair in evidence)
            left_evidence[candidate_key] = tuple(pair[0] for pair in prepared)
            right_evidence[candidate_key] = tuple(pair[1] for pair in prepared)
        left_scores = self._score_side(query_left, left_evidence)
        right_scores = self._score_side(query_right, right_evidence)
        scores = {
            candidate_key: (left_scores[candidate_key] + right_scores[candidate_key])
            / 2.0
            for candidate_key in catalog
        }
        logger.debug(
            "CurvRank matched sighting {} against {} candidates "
            "({} query descriptors, {} catalog descriptors)",
            query.sighting_id,
            len(catalog),
            self._descriptor_count((query_left, query_right)),
            self._descriptor_count(
                ear
                for ears in (*left_evidence.values(), *right_evidence.values())
                for ear in ears
            ),
        )
        return scores

    def _descriptor_count(self, ears: Iterable[PreparedEar]) -> int:
        """Count cached descriptors across all scales of the given ears."""
        return sum(
            len(descriptors) for ear in ears for descriptors in self._describe(ear)
        )

    def _score_side(
        self,
        query_ear: PreparedEar,
        catalog_ears: Mapping[CandidateKey, tuple[PreparedEar, ...]],
    ) -> dict[CandidateKey, float]:
        """Score one ear side with LNBNN summed over scales.

        Candidates that never appear among a query descriptor's nearest
        neighbors keep their neutral `0.0` accumulation. Scales with
        fewer than two catalog descriptors are skipped because the
        LNBNN normalizing distance is undefined there.
        """
        query_scales = self._describe(query_ear)
        accumulated: dict[CandidateKey, float] = dict.fromkeys(catalog_ears, 0.0)
        for scale_index, query_descriptors in enumerate(query_scales):
            rows: list[NDArray[np.float32]] = []
            labels: list[CandidateKey] = []
            for candidate_key, ears in catalog_ears.items():
                for ear in ears:
                    descriptors = self._describe(ear)[scale_index]
                    if len(descriptors):
                        rows.append(descriptors)
                        labels.extend([candidate_key] * len(descriptors))
            if len(query_descriptors) == 0 or len(labels) < 2:
                continue
            neighbor_count = min(self._config.lnbnn_k + 1, len(labels))
            index = NearestNeighbors(n_neighbors=neighbor_count).fit(np.vstack(rows))
            distances, indices = index.kneighbors(query_descriptors)
            for row_distances, row_indices in zip(distances, indices, strict=True):
                normalizer = float(row_distances[-1])
                scored: set[CandidateKey] = set()
                for distance, neighbor in zip(
                    row_distances[:-1],
                    row_indices[:-1],
                    strict=True,
                ):
                    candidate_key = labels[neighbor]
                    if candidate_key in scored:
                        continue
                    scored.add(candidate_key)
                    accumulated[candidate_key] += float(distance) - normalizer
        return {
            candidate_key: -value for candidate_key, value in accumulated.items()
        }
