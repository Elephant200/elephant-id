"""Compare tear profiles and return query alignments."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass(frozen=True, slots=True)
class EarProfileSimilarityConfig:
    """Settings for profile sampling, alignment, and score penalties."""

    resampled_bins: int = 240
    max_shift_fraction: float = 0.15
    shift_penalty_scale: float = 0.08
    shift_penalty_power: float = 4.0
    depth_exponent: float = 0.5
    stretches: tuple[float, ...] = tuple(
        round(0.8 + index * 0.025, 3) for index in range(17)
    )


@dataclass(frozen=True, slots=True)
class EarComparison:
    """A similarity score and its query alignment.

    Positive shifts move the query right. A zero score has stretch 1 and shift 0.
    """

    score: float
    stretch: float
    shift_bins: int


class EarProfileSimilarity:
    """Compare a query ear's tear profiles with reference ears' tear profiles."""

    def __init__(self, config: EarProfileSimilarityConfig | None = None) -> None:
        """Set comparison parameters; use defaults if no config is supplied."""
        self._config = config or EarProfileSimilarityConfig()
        max_shift_bins = round(self._config.max_shift_fraction * self._config.resampled_bins)
        self._alignments = [
            (stretch, shift_bins)
            for stretch in self._config.stretches
            for shift_bins in range(-max_shift_bins, max_shift_bins + 1)
        ]
        self._penalties = np.array([self._penalty(shift_bins) for _, shift_bins in self._alignments])

    def compare(
        self,
        query_profiles: Sequence[np.ndarray],
        reference_profiles: Sequence[np.ndarray],
    ) -> EarComparison:
        """Return the highest similarity score and its query alignment for one reference ear."""
        return self.compare_many(query_profiles, (reference_profiles,))[0]

    def compare_many(
        self,
        query_profiles: Sequence[np.ndarray],
        reference_profiles: Sequence[Sequence[np.ndarray]],
    ) -> tuple[EarComparison, ...]:
        """Compare a query ear with each reference ear.

        At each query alignment, the score is the mean overlap across alpha
        scales times the shift penalty. Each reference gets its best alignment.

        Args:
            query_profiles: The query ear's tear-profile depths, as nonempty,
                finite 1-D arrays for each alpha scale.
            reference_profiles: Each reference ear's tear-profile depths, as
                nonempty, finite 1-D arrays for each alpha scale.

        Returns:
            Scores and query alignments for each reference.

        Raises:
            ValueError: If an ear does not have one tear profile for each alpha
                scale of the query, or if the tear profiles have different lengths.
        """
        _validate_ears(query_profiles, reference_profiles)
        if not reference_profiles:
            return ()

        query_rows = self._comparison_rows(query_profiles)
        reference_rows = np.array([self._comparison_rows(ear) for ear in reference_profiles])
        scores = self._alignment_scores(query_rows, reference_rows)
        return tuple(self._best_comparison(reference_scores) for reference_scores in scores)

    def _penalty(self, shift_bins: int) -> float:
        """Return the score penalty for shifting the query by `shift_bins`."""
        shift_fraction = abs(shift_bins) / self._config.resampled_bins
        return float(
            np.exp(-((shift_fraction / self._config.shift_penalty_scale) ** self._config.shift_penalty_power))
        )

    def _comparison_rows(self, profiles: Sequence[np.ndarray]) -> np.ndarray:
        """Clip, compress, and resample each tear profile onto the common grid.

        Returns:
            One row for each alpha scale, with shape `(row_count, resampled_bins)`.
        """
        rows = []
        for depths in profiles:
            clipped = np.maximum(np.asarray(depths, dtype=np.float64), 0.0)
            compressed = clipped ** self._config.depth_exponent
            grid_positions = np.linspace(0.0, len(compressed) - 1, self._config.resampled_bins)
            rows.append(np.interp(grid_positions, np.arange(len(compressed)), compressed))
        return np.array(rows)

    def _alignment_scores(self, query_rows: np.ndarray, reference_rows: np.ndarray) -> np.ndarray:
        """Return the mean row overlap at every query alignment, times its shift penalty.

        Args:
            query_rows: Query comparison rows, with shape `(row_count, bins)`.
            reference_rows: Reference comparison rows, with shape
                `(reference_count, row_count, bins)`.

        Returns:
            Scores with shape `(reference_count, alignment_count)`.
        """
        overlaps_by_row = [
            _overlaps(reference_rows[:, row_index], self._aligned_query_rows(query_row))
            for row_index, query_row in enumerate(query_rows)
        ]
        return np.mean(overlaps_by_row, axis=0) * self._penalties

    def _aligned_query_rows(self, row: np.ndarray) -> np.ndarray:
        """Return the query row at every alignment, with shape `(alignment_count, bins)`."""
        stretched = {stretch: _stretched(row, stretch) for stretch in self._config.stretches}
        return np.array([_shifted(stretched[stretch], shift_bins) for stretch, shift_bins in self._alignments])

    def _best_comparison(self, scores: np.ndarray) -> EarComparison:
        """Return the highest score and its alignment, or a neutral alignment for zero."""
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score == 0.0:
            return EarComparison(score=0.0, stretch=1.0, shift_bins=0)
        stretch, shift_bins = self._alignments[best]
        return EarComparison(score=score, stretch=float(stretch), shift_bins=shift_bins)


def _validate_ears(
    query_profiles: Sequence[np.ndarray],
    reference_profiles: Sequence[Sequence[np.ndarray]],
) -> None:
    """Check that the ears have comparable tear profiles.

    Raises:
        ValueError: If an ear does not have one tear profile for each alpha
            scale of the query, or if the tear profiles have different lengths.
    """
    scale_count = len(query_profiles)
    if scale_count == 0 or any(len(ear) != scale_count for ear in reference_profiles):
        raise ValueError("Every ear needs one tear profile for each alpha scale")
    lengths = {len(profile) for ear in (query_profiles, *reference_profiles) for profile in ear}
    if len(lengths) != 1:
        raise ValueError("Compared tear profiles must have the same length")


def _stretched(row: np.ndarray, stretch: float) -> np.ndarray:
    """Return the row scaled about its center, with zero in bins from outside the row."""
    output_fractions = np.linspace(0.0, 1.0, len(row))
    source_positions = ((output_fractions - 0.5) / stretch + 0.5) * (len(row) - 1)
    return np.interp(source_positions, np.arange(len(row)), row, left=0.0, right=0.0)


def _shifted(row: np.ndarray, shift_bins: int) -> np.ndarray:
    """Return the row moved right by `shift_bins`, with zero in the vacated bins."""
    shifted = np.zeros_like(row)
    if 0 <= shift_bins < len(row):
        shifted[shift_bins:] = row[: len(row) - shift_bins]
    elif -len(row) < shift_bins < 0:
        shifted[:shift_bins] = row[-shift_bins:]
    return shifted


def _overlaps(reference_rows: np.ndarray, aligned_query_rows: np.ndarray) -> np.ndarray:
    """Return the overlap `sum(min) / sum(max)` of every reference row and aligned query row.

    Each output row is a reference. Each column is one query alignment.
    """
    distance = cdist(reference_rows, aligned_query_rows, metric="cityblock")
    total = aligned_query_rows.sum(axis=1)[None, :] + reference_rows.sum(axis=1)[:, None]
    # For nonnegative depths: sum(min(q, r)) = (sum(q) + sum(r) - sum(abs(q - r))) / 2.
    intersection = (total - distance) * 0.5
    # Subtraction loses precision when total and distance are almost equal.
    # Estimate the rounding error from the bin count and total depth.
    # Recalculate small intersections directly, including exact zero overlaps.
    cancellation_bound = 4 * np.finfo(np.float64).eps * aligned_query_rows.shape[1] * total
    for index, reference in enumerate(reference_rows):
        uncertain = intersection[index] <= cancellation_bound[index]
        intersection[index, uncertain] = np.minimum(aligned_query_rows[uncertain], reference).sum(axis=1)
    union = total - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
