"""Directional tear-profile alignment with shared alpha-shape scale support."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.distance import cdist

_OVERLAP_WORKSPACE_BYTES = 16 * 1024 * 1024
ProfileChannel = Literal["depth", "depth_change", "signed_depth_change"]


def angular_weights(
    bins: int, center_degrees: float, width_degrees: float
) -> tuple[float, ...]:
    """Return Gaussian angular weights over the canonical half-circle.

    Raises:
        ValueError: If the bin count or Gaussian width is invalid.
    """
    if bins < 2 or width_degrees <= 0:
        raise ValueError("Angular weights require at least two bins and positive width")
    angles = np.linspace(0.0, 180.0, bins)
    return tuple(np.exp(-0.5 * ((angles - center_degrees) / width_degrees) ** 2))


@dataclass(frozen=True, slots=True)
class TearMatcherConfig:
    """Profile preparation and the bounded directional alignment search.

    Defaults retain the single-scale main baseline. The publication
    composition supplies its measured settings explicitly. Signed depth
    change keeps rising and falling slopes separate under one alignment.
    """

    resampled_bins: int = 240
    max_shift_fraction: float = 0.15
    shift_penalty_scale: float = 0.08
    shift_penalty_power: float = 4.0
    depth_exponent: float = 0.5
    stretches: tuple[float, ...] = tuple(
        round(0.8 + index * 0.025, 3) for index in range(17)
    )
    bin_weights: tuple[float, ...] | None = None
    channel: ProfileChannel = "depth"

    def __post_init__(self) -> None:
        """Validate channel and angular weights.

        Raises:
            ValueError: If the channel is unknown or weights are invalid.
        """
        if self.channel not in ("depth", "depth_change", "signed_depth_change"):
            raise ValueError("Unknown profile channel")
        if self.bin_weights is not None:
            weights = np.asarray(self.bin_weights)
            if (
                weights.shape != (self.resampled_bins,)
                or not np.isfinite(weights).all()
                or np.any(weights < 0)
                or not np.any(weights > 0)
            ):
                raise ValueError(
                    "Angular weights must be finite, nonnegative, and match bins"
                )


@dataclass(frozen=True, slots=True)
class TearMatch:
    """Directional score and alignment; a positive shift moves query depths right."""

    score: float
    stretch: float
    shift_bins: int


class TearMatcher:
    """Align one ear's profile stack against catalog stacks in bounded memory."""

    def __init__(self, config: TearMatcherConfig | None = None) -> None:
        """Configure profile preparation and precompute the alignment grid."""
        self._config = config or TearMatcherConfig()
        self._precompute_search()

    def match(
        self, query_profile: np.ndarray, catalog_profile: np.ndarray
    ) -> TearMatch:
        """Match one query profile to one catalog profile."""
        return self.match_many(query_profile, (catalog_profile,))[0]

    def match_many(
        self, query_profile: np.ndarray, catalog_profiles: Sequence[np.ndarray]
    ) -> tuple[TearMatch, ...]:
        """Match a single-scale query against single-scale catalog profiles."""
        return self.match_stack_many(
            (query_profile,), tuple((profile,) for profile in catalog_profiles)
        )

    def match_stack(
        self, query: Sequence[np.ndarray], candidate: Sequence[np.ndarray]
    ) -> TearMatch:
        """Match two ears using one transformation across all alpha-shape scales."""
        return self.match_stack_many(query, (candidate,))[0]

    def match_stack_many(
        self,
        query: Sequence[np.ndarray],
        candidates: Sequence[Sequence[np.ndarray]],
    ) -> tuple[TearMatch, ...]:
        """Return directional matches under one alignment per candidate stack.

        Scales enter the mean overlap before its maximum is selected. In the
        signed channel, rising and falling slopes also share that alignment.

        Raises:
            ValueError: If a stack is empty or scale counts do not agree.
        """
        if not candidates:
            return ()
        if not query or any(len(stack) != len(query) for stack in candidates):
            raise ValueError("Profile stacks must be nonempty and pair one-to-one")
        prepared_query = self._prepare_stack(query)
        variants = tuple(self._build_query_variants(row) for row in prepared_query)
        variant_count = len(self._variant_penalties)
        batch_size = max(1, _OVERLAP_WORKSPACE_BYTES // (variant_count * 8 * 4))
        matches: list[TearMatch] = []
        for start in range(0, len(candidates), batch_size):
            batch = np.asarray(
                [
                    self._prepare_stack(stack)
                    for stack in candidates[start : start + batch_size]
                ]
            )
            scores = np.zeros((len(batch), variant_count))
            for index, query_variants in enumerate(variants):
                scores += self._overlap_scores(batch[:, index, :], query_variants)
            scores *= self._variant_penalties / len(variants)
            matches.extend(self._select_matches(scores))
        return tuple(matches)

    @staticmethod
    def _overlap_scores(catalog: np.ndarray, query_variants: np.ndarray) -> np.ndarray:
        """Compute Ruzicka overlap as (total mass - L1) / (total mass + L1)."""
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
        # Rounding can put a mathematically zero overlap just below zero.
        np.clip(overlap, 0.0, 1.0, out=overlap)
        overlap[catalog_sum == 0.0, :] = 0.0
        overlap[:, variant_sum == 0.0] = 0.0
        return overlap

    def _prepare_stack(self, profiles: Sequence[np.ndarray]) -> np.ndarray:
        """Select the channel before compression, resampling, and angular weights."""
        rows: list[np.ndarray] = []
        for profile in profiles:
            values = np.asarray(profile, dtype=np.float64)
            if self._config.channel == "depth":
                rows.append(values)
            else:
                gradient = (
                    np.gradient(values) if len(values) > 1 else np.zeros_like(values)
                )
                if self._config.channel == "depth_change":
                    rows.append(np.abs(gradient))
                else:
                    rows.extend((np.maximum(gradient, 0.0), np.maximum(-gradient, 0.0)))
        prepared = self._prepare_profiles(rows)
        if self._config.bin_weights is not None:
            prepared *= np.asarray(self._config.bin_weights)
        return prepared

    def _prepare_profiles(self, profiles: Sequence[np.ndarray]) -> np.ndarray:
        """Compress and resample possibly ragged one-dimensional profiles."""
        output = np.empty((len(profiles), self._config.resampled_bins))
        for index, profile in enumerate(profiles):
            values = np.maximum(profile, 0.0) ** self._config.depth_exponent
            positions = np.linspace(0.0, len(values) - 1, self._config.resampled_bins)
            lower = np.floor(positions).astype(np.int32)
            upper = np.minimum(lower + 1, len(values) - 1)
            weight = positions - lower
            output[index] = values[lower] * (1.0 - weight) + values[upper] * weight
        return output

    def _build_query_variants(self, query: np.ndarray) -> np.ndarray:
        """Apply every configured centered stretch and zero-padded shift."""
        stretched = (
            query[self._stretch_lower_bins] * (1.0 - self._stretch_upper_weights)
            + query[self._stretch_upper_bins] * self._stretch_upper_weights
        )
        stretched[~self._stretch_in_bounds] = 0.0
        shifted = stretched[:, self._shift_source_bins] * self._shift_in_bounds
        return np.ascontiguousarray(shifted.reshape(-1, self._config.resampled_bins))

    def _select_matches(self, scores: np.ndarray) -> tuple[TearMatch, ...]:
        """Select the first best transformation, with neutral zero-score alignment."""
        indices = np.argmax(scores, axis=1)
        values = scores[np.arange(len(scores)), indices]
        return tuple(
            TearMatch(
                score=float(score),
                stretch=float(self._variant_stretches[index]) if score > 0 else 1.0,
                shift_bins=int(self._variant_shifts[index]) if score > 0 else 0,
            )
            for index, score in zip(indices, values, strict=True)
        )

    def _precompute_search(self) -> None:
        """Precompute interpolation positions, shifts, and penalties."""
        bins = self._config.resampled_bins
        max_shift = round(self._config.max_shift_fraction * bins)
        shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
        penalties = np.exp(
            -(
                (np.abs(shifts / bins) / self._config.shift_penalty_scale)
                ** self._config.shift_penalty_power
            )
        )
        sources = np.arange(bins)[None, :] - shifts[:, None]
        self._shift_in_bounds = (sources >= 0) & (sources < bins)
        self._shift_source_bins = np.clip(sources, 0, bins - 1)
        stretches = np.asarray(self._config.stretches)
        positions = (np.linspace(0.0, 1.0, bins) - 0.5) / stretches[:, None] + 0.5
        sources = positions * (bins - 1)
        self._stretch_in_bounds = (sources >= 0) & (sources <= bins - 1)
        lower = np.floor(sources).astype(np.int32)
        self._stretch_upper_weights = sources - lower
        self._stretch_lower_bins = np.clip(lower, 0, bins - 1)
        self._stretch_upper_bins = np.minimum(self._stretch_lower_bins + 1, bins - 1)
        stretch_grid, shift_grid = np.meshgrid(stretches, shifts, indexing="ij")
        self._variant_stretches = stretch_grid.ravel()
        self._variant_shifts = shift_grid.ravel()
        self._variant_penalties = np.tile(penalties, len(stretches))


SELECTED_PROFILE_SETTINGS = TearMatcherConfig(
    depth_exponent=0.75,
    shift_penalty_scale=0.16,
    bin_weights=angular_weights(240, 120.0, 35.0),
)
"""Fixed profile settings selected on the frozen tuning sets."""
