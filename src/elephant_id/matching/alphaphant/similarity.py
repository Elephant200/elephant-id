"""Fixed directional similarity for AlphaTear profile stacks."""

from collections.abc import Sequence

import numpy as np
from scipy.spatial.distance import cdist

_OVERLAP_WORKSPACE_BYTES = 16 * 1024 * 1024
_RESAMPLED_BINS = 240
_MAX_SHIFT_FRACTION = 0.10
_SHIFT_PENALTY_SCALE = 0.16
_SHIFT_PENALTY_POWER = 4.0
_MAGNITUDE_EXPONENT = 0.50
_DEPTH_WEIGHT = 0.40
_SIGNED_DEPTH_CHANGE_WEIGHT = 0.60
_STRETCHES = tuple(round(0.8 + index * 0.025, 3) for index in range(17))


class TearMatcher:
    """Compare profile stacks with one alignment across all channels and scales."""

    def __init__(self) -> None:
        """Precompute the fixed stretch, shift, and penalty search grid."""
        self._precompute_search()

    def match_many(
        self,
        query: Sequence[np.ndarray],
        candidates: Sequence[Sequence[np.ndarray]],
    ) -> np.ndarray:
        """Return one combined similarity for each candidate profile stack.

        Every depth, rising-change, and falling-change row at every scale uses
        the same shift and stretch. Combine weighted row scores before selecting
        the best alignment for each query-candidate pair.

        Raises:
            ValueError: If a stack is empty or reference counts do not agree.
        """
        if not candidates:
            return np.empty(0, dtype=np.float64)
        if not query or any(len(stack) != len(query) for stack in candidates):
            raise ValueError("Profile stacks must be nonempty and pair one-to-one")
        channels = tuple(
            (
                signed,
                weight,
                tuple(
                    self._build_query_variants(row)
                    for row in self._prepare_stack(query, signed=signed)
                ),
            )
            for signed, weight in (
                (False, _DEPTH_WEIGHT),
                (True, _SIGNED_DEPTH_CHANGE_WEIGHT),
            )
        )
        variant_count = len(self._variant_penalties)
        batch_size = max(1, _OVERLAP_WORKSPACE_BYTES // (variant_count * 8 * 4))
        scores = np.empty(len(candidates), dtype=np.float64)
        for start in range(0, len(candidates), batch_size):
            stacks = candidates[start : start + batch_size]
            matches = np.zeros((len(stacks), variant_count))
            for signed, weight, variants in channels:
                batch = np.asarray(
                    [self._prepare_stack(stack, signed=signed) for stack in stacks]
                )
                for index, query_variants in enumerate(variants):
                    matches += (weight / len(variants)) * self._overlap_scores(
                        batch[:, index, :], query_variants
                    )
            matches *= self._variant_penalties
            scores[start : start + len(stacks)] = np.max(matches, axis=1)
        return scores

    @staticmethod
    def _overlap_scores(catalog: np.ndarray, query_variants: np.ndarray) -> np.ndarray:
        """Compute Ruzicka overlap as `(total mass - L1) / (total mass + L1)`."""
        catalog = np.ascontiguousarray(catalog)
        distance = cdist(catalog, query_variants, metric="cityblock")
        catalog_sum = catalog.sum(axis=1)
        variant_sum = query_variants.sum(axis=1)
        total = catalog_sum[:, None] + variant_sum[None, :]
        denominator = total + distance
        overlap = np.divide(
            total - distance,
            denominator,
            out=np.zeros_like(distance),
            where=denominator > 0,
        )
        np.clip(overlap, 0.0, 1.0, out=overlap)
        overlap[catalog_sum == 0.0, :] = 0.0
        overlap[:, variant_sum == 0.0] = 0.0
        return overlap

    def _prepare_stack(
        self,
        profiles: Sequence[np.ndarray],
        *,
        signed: bool,
    ) -> np.ndarray:
        """Prepare one stack for depth or signed-depth-change comparison."""
        rows: list[np.ndarray] = []
        for profile in profiles:
            values = np.asarray(profile, dtype=np.float64)
            if not signed:
                rows.append(values)
                continue
            gradient = np.gradient(values) if len(values) > 1 else np.zeros_like(values)
            rows.extend((np.maximum(gradient, 0.0), np.maximum(-gradient, 0.0)))
        prepared = self._prepare_profiles(rows)
        prepared *= self._bin_weights
        return prepared

    @staticmethod
    def _prepare_profiles(profiles: Sequence[np.ndarray]) -> np.ndarray:
        """Compress and resample possibly ragged one-dimensional profiles."""
        output = np.empty((len(profiles), _RESAMPLED_BINS))
        for index, profile in enumerate(profiles):
            values = np.maximum(profile, 0.0) ** _MAGNITUDE_EXPONENT
            positions = np.linspace(0.0, len(values) - 1, _RESAMPLED_BINS)
            output[index] = np.interp(positions, np.arange(len(values)), values)
        return output

    def _build_query_variants(self, query: np.ndarray) -> np.ndarray:
        """Apply every fixed centered stretch and zero-padded shift."""
        stretched = (
            query[self._stretch_lower_bins] * (1.0 - self._stretch_upper_weights)
            + query[self._stretch_upper_bins] * self._stretch_upper_weights
        )
        stretched[~self._stretch_in_bounds] = 0.0
        shifted = stretched[:, self._shift_source_bins] * self._shift_in_bounds
        return np.ascontiguousarray(shifted.reshape(-1, _RESAMPLED_BINS))

    def _precompute_search(self) -> None:
        """Precompute interpolation positions, shifts, and penalties."""
        max_shift = round(_MAX_SHIFT_FRACTION * _RESAMPLED_BINS)
        shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
        penalties = np.exp(
            -(
                (np.abs(shifts / _RESAMPLED_BINS) / _SHIFT_PENALTY_SCALE)
                ** _SHIFT_PENALTY_POWER
            )
        )
        sources = np.arange(_RESAMPLED_BINS)[None, :] - shifts[:, None]
        self._shift_in_bounds = (sources >= 0) & (sources < _RESAMPLED_BINS)
        self._shift_source_bins = np.clip(sources, 0, _RESAMPLED_BINS - 1)
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
        self._variant_penalties = np.tile(penalties, len(stretches))
        angles = np.linspace(0.0, 180.0, _RESAMPLED_BINS)
        self._bin_weights = np.exp(-0.5 * ((angles - 120.0) / 35.0) ** 2)
