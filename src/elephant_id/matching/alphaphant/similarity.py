"""Compute ear similarities between a query ear and reference ears."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance as scipy_distance

from elephant_id.matching.alphaphant.extraction import AnalyzedEar, TearProfile


@dataclass(frozen=True, slots=True)
class EarComparatorConfig:
    """Settings for the comparison grid, the alignments, and the shift penalty."""

    resampled_bins: int = 240
    max_shift_fraction: float = 0.15
    shift_penalty_scale: float = 0.08
    shift_penalty_power: float = 4.0
    stretches: tuple[float, ...] = tuple(
        round(0.8 + index * 0.025, 3) for index in range(17)
    )


@dataclass(frozen=True, slots=True)
class EarSimilarity:
    """An ear similarity and the query alignment that gives it.

    A positive shift moves the query to larger angles. An ear similarity of
    zero has stretch 1 and shift 0.
    """

    score: float
    stretch: float
    shift_bins: int


class EarComparator:
    """Compute the ear similarity of a query ear with each reference ear."""

    def __init__(self, config: EarComparatorConfig | None = None) -> None:
        """Set the comparison settings; use the defaults if no config is supplied."""
        self._config = config or EarComparatorConfig()
        max_shift_bins = round(self._config.max_shift_fraction * self._config.resampled_bins)
        self._alignments = [
            (stretch, shift_bins)
            for stretch in self._config.stretches
            for shift_bins in range(-max_shift_bins, max_shift_bins + 1)
        ]
        self._shift_penalties = np.array(
            [self._compute_shift_penalty(shift_bins) for _, shift_bins in self._alignments]
        )

    def compare_ears(
        self,
        query_ear: AnalyzedEar,
        reference_ears: Sequence[AnalyzedEar],
    ) -> tuple[EarSimilarity, ...]:
        """Return the ear similarity of the query ear with each reference ear.

        For each alignment, the comparator calculates the mean real-valued
        Jaccard of the depth rows and multiplies it by the shift penalty. The
        ear similarity is the highest result.

        Returns:
            One ear similarity for each reference ear, in the same order.

        Raises:
            ValueError: If an ear does not have one tear profile for each alpha
                scale of the query, or if the tear profiles have different lengths.
        """
        if not _has_same_scale_count(query_ear, reference_ears):
            raise ValueError("Every ear needs one tear profile for each alpha scale")
        if not _has_same_length(query_ear, reference_ears):
            raise ValueError("Compared tear profiles must have the same length")
        if not reference_ears:
            return ()

        query_rows = self._build_depth_rows(query_ear.profiles_by_scale)
        reference_rows = np.array([self._build_depth_rows(ear.profiles_by_scale) for ear in reference_ears])
        scores = self._score_alignments(query_rows, reference_rows)
        return tuple(self._select_best_alignment(reference_scores) for reference_scores in scores)

    def _build_depth_rows(self, profiles: Sequence[TearProfile]) -> np.ndarray:
        """Return one depth row for each tear profile.

        For each tear profile, this method sets negative depths to zero,
        calculates the square root of each depth, and resamples the depths to
        `resampled_bins`.

        Returns:
            Depth rows with shape `(scale_count, resampled_bins)`.
        """
        rows = []
        for profile in profiles:
            compressed = np.sqrt(np.maximum(profile.depths, 0.0))
            grid_positions = np.linspace(0.0, len(compressed) - 1, self._config.resampled_bins)
            rows.append(np.interp(grid_positions, np.arange(len(compressed)), compressed))
        return np.array(rows)

    def _align_row(self, row: np.ndarray) -> np.ndarray:
        """Return the row at every alignment, with shape `(alignment_count, bins)`."""
        stretched = {stretch: _stretch(row, stretch) for stretch in self._config.stretches}
        return np.array([_shift(stretched[stretch], shift_bins) for stretch, shift_bins in self._alignments])

    def _score_alignments(self, query_rows: np.ndarray, reference_rows: np.ndarray) -> np.ndarray:
        """Return the alignment score of every alignment against each reference ear.

        For each alignment, this method calculates the real-valued Jaccard of
        each depth row, calculates the mean over the alpha scales, and
        multiplies the mean by the shift penalty. Column `j` contains the score of
        alignment `self._alignments[j]`.

        Args:
            query_rows: The query's depth rows, with shape `(scale_count, bins)`.
            reference_rows: The reference ears' depth rows, with shape
                `(reference_count, scale_count, bins)`.

        Returns:
            Scores with shape `(reference_count, alignment_count)`.
        """
        jaccard_by_row = [
            _real_valued_jaccard(reference_rows[:, scale_index], self._align_row(query_row))
            for scale_index, query_row in enumerate(query_rows)
        ]
        return np.mean(jaccard_by_row, axis=0) * self._shift_penalties

    def _compute_shift_penalty(self, shift_bins: int) -> float:
        """Return the shift penalty for a shift of `shift_bins`."""
        shift_fraction = abs(shift_bins) / self._config.resampled_bins
        return float(
            np.exp(-((shift_fraction / self._config.shift_penalty_scale) ** self._config.shift_penalty_power))
        )

    def _select_best_alignment(self, scores: np.ndarray) -> EarSimilarity:
        """Return the highest alignment score as the ear similarity, with its alignment."""
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score <= 1e-12: # approximately zero
            return EarSimilarity(score=0.0, stretch=1.0, shift_bins=0)
        stretch, shift_bins = self._alignments[best]
        return EarSimilarity(score=score, stretch=float(stretch), shift_bins=shift_bins)


def _has_same_scale_count(query_ear: AnalyzedEar, reference_ears: Sequence[AnalyzedEar]) -> bool:
    """Return whether the query and all reference ears have the same number of alpha scales."""
    scale_count = len(query_ear.profiles_by_scale)
    return scale_count > 0 and all(len(ear.profiles_by_scale) == scale_count for ear in reference_ears)


def _has_same_length(query_ear: AnalyzedEar, reference_ears: Sequence[AnalyzedEar]) -> bool:
    """Return whether all tear profiles of all ears have the same length."""
    lengths = {
        len(profile.depths)
        for ear in (query_ear, *reference_ears)
        for profile in ear.profiles_by_scale
    }
    return len(lengths) == 1


def _stretch(row: np.ndarray, stretch: float) -> np.ndarray:
    """Return the row stretched about its center. Bins with no source value are zero."""
    output_fractions = np.linspace(0.0, 1.0, len(row))
    source_positions = ((output_fractions - 0.5) / stretch + 0.5) * (len(row) - 1)
    return np.interp(source_positions, np.arange(len(row)), row, left=0.0, right=0.0)


def _shift(row: np.ndarray, shift_bins: int) -> np.ndarray:
    """Return the row moved by `shift_bins`. Bins with no source value are zero.

    A positive shift moves the row to larger angles. A negative shift moves
    the row to smaller angles.
    """
    shifted = np.zeros_like(row)
    if 0 <= shift_bins < len(row):
        shifted[shift_bins:] = row[: len(row) - shift_bins]
    elif -len(row) < shift_bins < 0:
        shifted[:shift_bins] = row[-shift_bins:]
    return shifted


def _real_valued_jaccard(reference_rows: np.ndarray, aligned_query_rows: np.ndarray) -> np.ndarray:
    """Return the real-valued Jaccard of every reference row and aligned query row.

    The value is zero when both rows contain only zeros.

    Returns:
        Values with shape `(reference_count, alignment_count)`.
    """
    # With L1 norms, (|x| + |y| - |x - y|) / (|x| + |y| + |x - y|) equals
    # sum(min) / sum(max) for nonnegative rows.
    norm_sum = reference_rows.sum(axis=1)[:, None] + aligned_query_rows.sum(axis=1)[None, :]
    distance = scipy_distance.cdist(reference_rows, aligned_query_rows, metric="cityblock")
    denominator = norm_sum + distance
    return np.divide(norm_sum - distance, denominator, out=np.zeros_like(denominator), where=denominator > 0)
