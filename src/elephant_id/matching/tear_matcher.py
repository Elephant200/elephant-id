"""Forward-only bulk tear-profile matching for elephant ear re-identification."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

_OVERLAP_WORKSPACE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TearMatcherConfig:
    """Parameters controlling resampling, shift penalties, and stretch search."""

    resampled_bins: int = 240
    max_shift_fraction: float = 0.15
    shift_penalty_scale: float = 0.08
    shift_penalty_power: float = 4.0
    depth_exponent: float = 0.5
    stretches: tuple[float, ...] = tuple(
        round(0.8 + index * 0.025, 3) for index in range(17)
    )


@dataclass(frozen=True, slots=True)
class TearMatch:
    """Forward similarity and alignment; positive shifts move query depths right."""

    score: float
    stretch: float
    shift_bins: int


class TearMatcher:
    """Match one query profile against catalog profiles using forward alignment."""

    def __init__(self, config: TearMatcherConfig | None = None) -> None:
        """Configure the forward alignment search."""
        self._config = config or TearMatcherConfig()
        self._precompute_search()

    def match(
        self, query_profile: np.ndarray, catalog_profile: np.ndarray
    ) -> TearMatch:
        """Match one query profile to one catalog profile."""
        return self.match_many(query_profile, (catalog_profile,))[0]

    def match_many(
        self,
        query_profile: np.ndarray,
        catalog_profiles: Sequence[np.ndarray],
    ) -> tuple[TearMatch, ...]:
        """Match one query profile to many catalog profiles.

        Args:
            query_profile: The profile to match, a 1D array of depths.
            catalog_profiles: The profiles to match against, a sequence of 1D arrays of depths.

        Returns:
            A tuple of matches in catalog-profile order.
        """
        if len(catalog_profiles) == 0:
            return ()

        prepared_query = self._prepare_profiles((query_profile,))[0]
        catalog_batches = self._prepare_catalog_batches(catalog_profiles)

        query_variants = self._build_query_variants(prepared_query)

        return self._score_catalog_batches(query_variants, catalog_batches)

    def _build_query_variants(self, prepared_query: np.ndarray) -> np.ndarray:
        """Stretch the query, then shift each stretch across the search offsets."""
        stretched_profiles = self._stretch_query(prepared_query)
        shifted_profiles = (
            stretched_profiles[:, self._shift_source_bins] * self._shift_in_bounds
        )

        return shifted_profiles.reshape(-1, self._config.resampled_bins)

    def _score_catalog_batches(
        self,
        query_variants: np.ndarray,
        catalog_batches: Iterator[np.ndarray],
    ) -> tuple[TearMatch, ...]:
        """Score prepared catalog batches using the same query variants."""
        variant_depth_sums = query_variants.sum(axis=1)
        matches: list[TearMatch] = []

        for catalog_batch in catalog_batches:
            scores = self._match_profile_batch(
                query_variants, variant_depth_sums, catalog_batch
            )
            matches.extend(self._select_matches(scores))

        return tuple(matches)

    def _match_profile_batch(
        self,
        query_variants: np.ndarray,
        variant_depth_sums: np.ndarray,
        catalog_batch: np.ndarray,
    ) -> np.ndarray:
        """Return a `(catalog_profile_count, variant_count)` similarity matrix."""
        overlap = np.minimum(query_variants[None, :, :], catalog_batch[:, None, :]).sum(
            axis=2
        )

        # min + max = sum
        catalog_depth_sums = catalog_batch.sum(axis=1)
        union = variant_depth_sums[None, :] + catalog_depth_sums[:, None] - overlap
        scores = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
        scores *= self._variant_penalties

        return scores

    def _select_matches(self, scores: np.ndarray) -> tuple[TearMatch, ...]:
        """Select each profile's best variant, retaining neutral zero-score alignments."""
        best_indices = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(len(scores)), best_indices]

        stretches = np.where(
            best_scores > 0.0, self._variant_stretches[best_indices], 1.0
        )
        shifts = np.where(best_scores > 0.0, self._variant_shifts[best_indices], 0)

        return tuple(
            TearMatch(score=float(score), stretch=float(stretch), shift_bins=int(shift))
            for score, stretch, shift in zip(
                best_scores, stretches, shifts, strict=True
            )
        )

    def _prepare_catalog_batches(
        self, catalog_profiles: Sequence[np.ndarray]
    ) -> Iterator[np.ndarray]:
        """Resample catalog batches sized to fit the overlap workspace."""
        variant_count = len(self._variant_stretches)
        bytes_per_profile = (
            variant_count * self._config.resampled_bins * np.dtype(np.float64).itemsize
        )
        batch_size = max(1, _OVERLAP_WORKSPACE_BYTES // bytes_per_profile)

        for start in range(0, len(catalog_profiles), batch_size):
            yield self._prepare_profiles(catalog_profiles[start : start + batch_size])

    def _prepare_profiles(self, source_profiles: Sequence[np.ndarray]) -> np.ndarray:
        """Compress ragged source depths into a `(profile_count, resampled_bins)` matrix."""
        indices_by_length: dict[int, list[int]] = {}
        for profile_index, profile in enumerate(source_profiles):
            indices_by_length.setdefault(len(profile), []).append(profile_index)

        resampled_profiles = np.empty(
            (len(source_profiles), self._config.resampled_bins)
        )

        for profile_indices in indices_by_length.values():
            source_depths = np.asarray(
                np.stack([source_profiles[index] for index in profile_indices]),
                dtype=np.float64,
            )
            compressed_depths = (
                np.maximum(source_depths, 0.0) ** self._config.depth_exponent
            )
            resampled_profiles[profile_indices] = self._resample_rows(compressed_depths)

        return resampled_profiles

    def _resample_rows(self, profiles: np.ndarray) -> np.ndarray:
        """Sample each profile row on the same evenly spaced bin grid."""
        last_bin = profiles.shape[1] - 1
        source_bins = np.linspace(0.0, last_bin, self._config.resampled_bins)
        lower_bins = np.floor(source_bins).astype(np.int32)
        upper_bins = np.minimum(lower_bins + 1, last_bin)
        upper_weights = source_bins - lower_bins

        return (
            profiles[:, lower_bins] * (1.0 - upper_weights)
            + profiles[:, upper_bins] * upper_weights
        )

    def _stretch_query(self, query: np.ndarray) -> np.ndarray:
        """Sample the query at every configured centered stretch."""
        stretched = (
            query[self._stretch_lower_bins] * (1.0 - self._stretch_upper_weights)
            + query[self._stretch_upper_bins] * self._stretch_upper_weights
        )
        stretched[~self._stretch_in_bounds] = 0.0

        return stretched

    def _precompute_search(self) -> None:
        """Precompute sampling positions and the stretch-by-shift search grid."""
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
        self._shift_in_bounds = (shift_source_bins >= 0) & (
            shift_source_bins < bin_count
        )
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
        self._stretch_upper_bins = np.minimum(
            self._stretch_lower_bins + 1, bin_count - 1
        )

        stretch_grid, shift_grid = np.meshgrid(
            stretch_factors, shift_offsets, indexing="ij"
        )
        self._variant_stretches = stretch_grid.ravel()
        self._variant_shifts = shift_grid.ravel()
        self._variant_penalties = np.tile(shift_penalties, len(stretch_factors))
