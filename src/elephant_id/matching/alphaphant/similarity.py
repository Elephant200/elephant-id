"""Directional similarity between two ears' tear-profile stacks.

Each ear contributes two channels built from its per-scale tear profiles:

- depth: one row per scale;
- depth change: a rising row and a falling row per scale.

Rows are compressed, resampled to a common grid, and weighted toward the lower
margin. The query is stretched and shifted over a fixed grid. At each alignment
the score is the weighted channel mean of Ruzicka row overlaps, times a shift
penalty; the best alignment gives the similarity.
"""

from collections.abc import Sequence

import numpy as np
from scipy.spatial.distance import cdist

_RESAMPLED_BINS = 240
_MAGNITUDE_EXPONENT = 0.50
_ANGLE_WEIGHT_CENTER_DEGREES = 120.0
_ANGLE_WEIGHT_WIDTH_DEGREES = 35.0
_DEPTH_WEIGHT = 0.40
_DEPTH_CHANGE_WEIGHT = 0.60
_STRETCHES = tuple(round(0.8 + index * 0.025, 3) for index in range(17))
_MAX_SHIFT_FRACTION = 0.10
_SHIFT_PENALTY_SCALE = 0.16
_SHIFT_PENALTY_POWER = 4.0
_OVERLAP_WORKSPACE_BYTES = 16 * 1024 * 1024


class EarProfileSimilarity:
    """Score one query ear against reference ears with one shared alignment."""

    def __init__(self) -> None:
        """Precompute the angle weights and the stretch and shift grid."""
        angles = np.linspace(0.0, 180.0, _RESAMPLED_BINS)
        self._angle_weights = np.exp(
            -0.5
            * ((angles - _ANGLE_WEIGHT_CENTER_DEGREES) / _ANGLE_WEIGHT_WIDTH_DEGREES)
            ** 2
        )
        self._precompute_alignments()

    def compare_many(
        self,
        query: Sequence[np.ndarray],
        references: Sequence[Sequence[np.ndarray]],
    ) -> np.ndarray:
        """Return one similarity per reference ear.

        Args:
            query: The query ear's depth profiles in scale order.
            references: Each reference ear's depth profiles in the same order.

        Raises:
            ValueError: If an ear has no profiles or scale counts differ.
        """
        if not references:
            return np.empty(0, dtype=np.float64)
        if not query or any(len(ear) != len(query) for ear in references):
            raise ValueError("Profile stacks must be nonempty and pair one-to-one")

        query_alignments = [
            [self._alignments(row) for row in rows] for rows in self._channels(query)
        ]
        weights = (_DEPTH_WEIGHT, _DEPTH_CHANGE_WEIGHT)
        batch_size = max(
            1, _OVERLAP_WORKSPACE_BYTES // (len(self._alignment_penalties) * 8 * 4)
        )
        scores = np.empty(len(references), dtype=np.float64)
        for start in range(0, len(references), batch_size):
            batch = [self._channels(ear) for ear in references[start : start + batch_size]]
            combined = np.zeros((len(batch), len(self._alignment_penalties)))
            for channel, (weight, alignments) in enumerate(
                zip(weights, query_alignments, strict=True)
            ):
                rows = np.asarray([channels[channel] for channels in batch])
                for row, row_alignments in enumerate(alignments):
                    combined += (weight / len(alignments)) * _overlaps(
                        rows[:, row, :], row_alignments
                    )
            combined *= self._alignment_penalties
            scores[start : start + len(batch)] = combined.max(axis=1)
        return scores

    def _channels(self, profiles: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Return comparison rows for the depth and depth-change channels.

        Depth change is taken from the original signed depths, before clipping.
        """
        depth_rows = [np.asarray(profile, dtype=np.float64) for profile in profiles]
        change_rows = []
        for depths in depth_rows:
            change = np.gradient(depths) if len(depths) > 1 else np.zeros_like(depths)
            change_rows.extend((np.maximum(change, 0.0), np.maximum(-change, 0.0)))
        return self._comparison_rows(depth_rows), self._comparison_rows(change_rows)

    def _comparison_rows(self, rows: Sequence[np.ndarray]) -> np.ndarray:
        """Clip, compress, resample, and angle-weight rows of possibly unequal length."""
        output = np.empty((len(rows), _RESAMPLED_BINS))
        for index, row in enumerate(rows):
            compressed = np.maximum(row, 0.0) ** _MAGNITUDE_EXPONENT
            positions = np.linspace(0.0, len(compressed) - 1, _RESAMPLED_BINS)
            output[index] = np.interp(positions, np.arange(len(compressed)), compressed)
        output *= self._angle_weights
        return output

    def _alignments(self, row: np.ndarray) -> np.ndarray:
        """Return the row under every stretch, then every shift, zero-padded."""
        stretched = (
            row[self._stretch_lower_bins] * (1.0 - self._stretch_upper_weights)
            + row[self._stretch_upper_bins] * self._stretch_upper_weights
        )
        stretched[~self._stretch_in_bounds] = 0.0
        shifted = stretched[:, self._shift_source_bins] * self._shift_in_bounds
        return np.ascontiguousarray(shifted.reshape(-1, _RESAMPLED_BINS))

    def _precompute_alignments(self) -> None:
        """Precompute stretch interpolation, shift indices, and shift penalties."""
        stretches = np.asarray(_STRETCHES)
        positions = (np.linspace(0.0, 1.0, _RESAMPLED_BINS) - 0.5) / stretches[
            :, None
        ] + 0.5
        sources = positions * (_RESAMPLED_BINS - 1)
        self._stretch_in_bounds = (sources >= 0) & (sources <= _RESAMPLED_BINS - 1)
        lower = np.floor(sources).astype(np.int32)
        self._stretch_upper_weights = sources - lower
        self._stretch_lower_bins = np.clip(lower, 0, _RESAMPLED_BINS - 1)
        self._stretch_upper_bins = np.minimum(
            self._stretch_lower_bins + 1, _RESAMPLED_BINS - 1
        )

        max_shift = round(_MAX_SHIFT_FRACTION * _RESAMPLED_BINS)
        shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
        shift_sources = np.arange(_RESAMPLED_BINS)[None, :] - shifts[:, None]
        self._shift_in_bounds = (shift_sources >= 0) & (shift_sources < _RESAMPLED_BINS)
        self._shift_source_bins = np.clip(shift_sources, 0, _RESAMPLED_BINS - 1)
        penalties = np.exp(
            -(
                (np.abs(shifts / _RESAMPLED_BINS) / _SHIFT_PENALTY_SCALE)
                ** _SHIFT_PENALTY_POWER
            )
        )
        self._alignment_penalties = np.tile(penalties, len(stretches))


def _overlaps(references: np.ndarray, query_alignments: np.ndarray) -> np.ndarray:
    """Return Ruzicka overlap `sum(min) / sum(max)` for every row pair.

    For nonnegative rows, `sum(min) / sum(max) = (total - L1) / (total + L1)`,
    where `total` is the sum of both rows' masses.
    """
    references = np.ascontiguousarray(references)
    distance = cdist(references, query_alignments, metric="cityblock")
    reference_mass = references.sum(axis=1)
    query_mass = query_alignments.sum(axis=1)
    total = reference_mass[:, None] + query_mass[None, :]
    denominator = total + distance
    overlap = np.divide(
        total - distance,
        denominator,
        out=np.zeros_like(distance),
        where=denominator > 0,
    )
    np.clip(overlap, 0.0, 1.0, out=overlap)
    overlap[reference_mass == 0.0, :] = 0.0
    overlap[:, query_mass == 0.0] = 0.0
    return overlap
