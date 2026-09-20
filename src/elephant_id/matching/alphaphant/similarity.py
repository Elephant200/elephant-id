"""Compare tear profiles and return query alignments."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

_OVERLAP_WORKSPACE_BYTES = 16 * 1024 * 1024


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

    Positive shifts move the query right. A zero score has stretch 1 and shift 0."""

    score: float
    stretch: float
    shift_bins: int


class EarProfileSimilarity:
    """Compare a query profile with reference profiles."""

    def __init__(self, config: EarProfileSimilarityConfig | None = None) -> None:
        """Set comparison parameters; use defaults if no config is supplied."""
        self._config = config or EarProfileSimilarityConfig()
        self._precompute_search()

    def compare(
        self, query_profile: np.ndarray, reference_profile: np.ndarray
    ) -> EarComparison:
        """Return the highest similarity score and its query alignment."""
        return self.compare_many(query_profile, (reference_profile,))[0]

    def compare_many(
        self,
        query_profile: np.ndarray,
        reference_profiles: Sequence[np.ndarray],
    ) -> tuple[EarComparison, ...]:
        """Compare a query with each reference without changing the inputs.

        Args:
            query_profile: A nonempty, finite 1-D depth array.
            reference_profiles: Nonempty, finite 1-D depth arrays. Lengths may differ.

        Returns:
            Scores and query alignments in reference order. No references gives
            an empty tuple."""
        if len(reference_profiles) == 0:
            return ()

        prepared_query = self._prepare_profiles((query_profile,))[0]
        reference_batches = self._prepare_reference_batches(reference_profiles)

        query_variants = self._build_query_variants(prepared_query)

        return self._score_reference_batches(query_variants, reference_batches)

    def _build_query_variants(self, prepared_query: np.ndarray) -> np.ndarray:
        """Return aligned query rows with shape `(variant_count, bin_count)`.

        Rows follow stretch order, then shift order."""
        stretched_profiles = self._stretch_query(prepared_query)
        shifted_profiles = (
            stretched_profiles[:, self._shift_source_bins] * self._shift_in_bounds
        )

        return shifted_profiles.reshape(-1, self._config.resampled_bins)

    def _score_reference_batches(
        self,
        query_variants: np.ndarray,
        reference_batches: Iterator[np.ndarray],
    ) -> tuple[EarComparison, ...]:
        """Return the best comparison for each reference, in input order."""
        variant_depth_sums = query_variants.sum(axis=1)
        matches: list[EarComparison] = []

        for reference_batch in reference_batches:
            scores = self._match_profile_batch(
                query_variants, variant_depth_sums, reference_batch
            )
            matches.extend(self._select_matches(scores))

        return tuple(matches)

    def _match_profile_batch(
        self,
        query_variants: np.ndarray,
        variant_depth_sums: np.ndarray,
        reference_batch: np.ndarray,
    ) -> np.ndarray:
        """Return penalized scores with reference rows and query-variant columns."""
        overlap = np.minimum(query_variants[None, :, :], reference_batch[:, None, :]).sum(axis=2)

        # min + max = sum
        reference_depth_sums = reference_batch.sum(axis=1)
        union = variant_depth_sums[None, :] + reference_depth_sums[:, None] - overlap
        scores = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
        scores *= self._variant_penalties

        return scores

    def _select_matches(self, scores: np.ndarray) -> tuple[EarComparison, ...]:
        """Return the highest score and its alignment for each reference row."""
        best_indices = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(len(scores)), best_indices]

        stretches = np.where(best_scores > 0.0, self._variant_stretches[best_indices], 1.0)
        shifts = np.where(best_scores > 0.0, self._variant_shifts[best_indices], 0)

        return tuple(
            EarComparison(score=float(score), stretch=float(stretch), shift_bins=int(shift))
            for score, stretch, shift in zip(
                best_scores, stretches, shifts, strict=True
            )
        )

    def _prepare_reference_batches(
        self, reference_profiles: Sequence[np.ndarray]
    ) -> Iterator[np.ndarray]:
        """Yield prepared reference batches in input order."""
        variant_count = len(self._variant_stretches)
        bytes_per_profile = (
            variant_count * self._config.resampled_bins * np.dtype(np.float64).itemsize
        )
        batch_size = max(1, _OVERLAP_WORKSPACE_BYTES // bytes_per_profile)

        for start in range(0, len(reference_profiles), batch_size):
            yield self._prepare_profiles(reference_profiles[start : start + batch_size])

    def _prepare_profiles(self, source_profiles: Sequence[np.ndarray]) -> np.ndarray:
        """Return nonnegative, compressed profiles on the common grid, in input order."""
        indices_by_length: dict[int, list[int]] = {}
        for profile_index, profile in enumerate(source_profiles):
            indices_by_length.setdefault(len(profile), []).append(profile_index)

        resampled_profiles = np.empty((len(source_profiles), self._config.resampled_bins))

        for profile_indices in indices_by_length.values():
            source_depths = np.asarray(
                np.stack([source_profiles[index] for index in profile_indices]),
                dtype=np.float64,
            )
            compressed_depths = np.maximum(source_depths, 0.0) ** self._config.depth_exponent
            resampled_profiles[profile_indices] = self._resample_rows(compressed_depths)

        return resampled_profiles

    def _resample_rows(self, profiles: np.ndarray) -> np.ndarray:
        """Return each profile row sampled on the common grid."""
        last_bin = profiles.shape[1] - 1
        source_bins = np.linspace(0.0, last_bin, self._config.resampled_bins)
        lower_bins = np.floor(source_bins).astype(np.int32)
        upper_bins = np.minimum(lower_bins + 1, last_bin)
        upper_weights = source_bins - lower_bins

        return profiles[:, lower_bins] * (1.0 - upper_weights) + profiles[:, upper_bins] * upper_weights

    def _stretch_query(self, query: np.ndarray) -> np.ndarray:
        """Return query rows in configured stretch order, with zero outside the grid."""
        stretched = (
            query[self._stretch_lower_bins] * (1.0 - self._stretch_upper_weights)
            + query[self._stretch_upper_bins] * self._stretch_upper_weights
        )
        stretched[~self._stretch_in_bounds] = 0.0

        return stretched

    def _precompute_search(self) -> None:
        """Set query sampling indices and penalties for the configured alignments."""
        bin_count = self._config.resampled_bins

        max_shift_bins = round(self._config.max_shift_fraction * bin_count)
        shift_offsets = np.arange(-max_shift_bins, max_shift_bins + 1, dtype=np.int32)
        shift_fractions = shift_offsets / bin_count
        shift_penalties = np.exp(
            -(
                (np.abs(shift_fractions) / self._config.shift_penalty_scale)
                ** self._config.shift_penalty_power
            )
        )

        shift_source_bins = np.arange(bin_count)[None, :] - shift_offsets[:, None]
        self._shift_in_bounds = (shift_source_bins >= 0) & (shift_source_bins < bin_count)
        self._shift_source_bins = np.clip(shift_source_bins, 0, bin_count - 1)

        stretch_factors = np.asarray(self._config.stretches)
        normalized_output_positions = np.linspace(0.0, 1.0, bin_count)
        normalized_source_positions = (
            normalized_output_positions - 0.5
        ) / stretch_factors[:, None] + 0.5

        self._stretch_source_bins = normalized_source_positions * (bin_count - 1)
        self._stretch_in_bounds = (self._stretch_source_bins >= 0.0) & (
            self._stretch_source_bins <= bin_count - 1
        )

        lower_bins = np.floor(self._stretch_source_bins).astype(np.int32)
        self._stretch_upper_weights = self._stretch_source_bins - lower_bins
        self._stretch_lower_bins = np.clip(lower_bins, 0, bin_count - 1)
        self._stretch_upper_bins = np.minimum(self._stretch_lower_bins + 1, bin_count - 1)

        stretch_grid, shift_grid = np.meshgrid(stretch_factors, shift_offsets, indexing="ij")
        self._variant_stretches = stretch_grid.ravel()
        self._variant_shifts = shift_grid.ravel()
        self._variant_penalties = np.tile(shift_penalties, len(stretch_factors))
