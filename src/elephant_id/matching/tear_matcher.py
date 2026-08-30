"""Symmetric tear-profile matching for elephant ear re-identification."""

from dataclasses import dataclass

import numpy as np


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
    """Symmetric score and query-to-catalog alignment; reverse is discarded."""

    score: float
    stretch: float
    shift_bins: int


class TearMatcher:
    """Compute similarity score and alignment for two tear profiles."""

    def __init__(self, config: TearMatcherConfig | None = None) -> None:
        """Create a matcher and precompute its configured shift penalties."""
        self._config = config or TearMatcherConfig()
        max_shift = round(self._config.max_shift_fraction * self._config.resampled_bins)
        self._shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
        shift_fractions = self._shifts / self._config.resampled_bins
        self._shift_penalties = np.exp(
            -(
                (np.abs(shift_fractions) / self._config.shift_penalty_scale)
                ** self._config.shift_penalty_power
            )
        )

    def match(self, query: np.ndarray, candidate: np.ndarray) -> TearMatch:
        """Return symmetric similarity score with the forward direction's stretch and shift."""
        query_profile = self._prepare(query)
        candidate_profile = self._prepare(candidate)
        query_to_catalog = self._align(query_profile, candidate_profile)
        catalog_to_query = self._align(candidate_profile, query_profile)
        return TearMatch(
            score=(query_to_catalog.score + catalog_to_query.score) / 2.0, # Symmetrize
            stretch=query_to_catalog.stretch,
            shift_bins=query_to_catalog.shift_bins,
        )

    def _prepare(self, profile: np.ndarray) -> np.ndarray:
        """Clip, compress, and resample one tear profile."""
        depths = np.maximum(np.asarray(profile, dtype=np.float64), 0.0)
        depths = depths**self._config.depth_exponent
        return self._sample(
            depths,
            np.linspace(0.0, len(depths) - 1, self._config.resampled_bins),
        )

    def _align(
        self,
        query: np.ndarray,
        candidate: np.ndarray,
    ) -> TearMatch:
        """Return one directional score with its stretch and shift."""
        best_score = 0.0
        best_shift = 0
        best_stretch = 1.0
        for stretch in self._config.stretches:
            stretched = self._stretch(query, stretch)
            for shift, penalty in zip(self._shifts, self._shift_penalties, strict=True):
                shifted = self._shift(stretched, int(shift))
                score = self._overlap(shifted, candidate) * penalty
                if score > best_score:
                    best_score = score
                    best_shift = int(shift)
                    best_stretch = stretch

        return TearMatch(
            score=float(best_score),
            stretch=best_stretch,
            shift_bins=best_shift,
        )

    def _sample(self, profile: np.ndarray, source_bins: np.ndarray) -> np.ndarray:
        """Linearly interpolate `profile` at source-bin positions."""
        last_bin = len(profile) - 1
        unclipped_lower = np.floor(source_bins).astype(np.int32)
        lower = np.clip(unclipped_lower, 0, last_bin)
        upper = np.minimum(lower + 1, last_bin)
        upper_weight = source_bins - unclipped_lower
        return profile[lower] * (1.0 - upper_weight) + profile[upper] * upper_weight

    def _stretch(
        self,
        profile: np.ndarray,
        stretch: float,
    ) -> np.ndarray:
        """Sample a profile with a centered stretch transform."""
        output_arc = np.linspace(0.0, 1.0, self._config.resampled_bins)
        source_arc = (output_arc - 0.5) / stretch + 0.5
        source_bins = source_arc * (self._config.resampled_bins - 1)
        valid_source = (
            (source_bins >= 0.0)
            & (source_bins <= self._config.resampled_bins - 1)
        )
        stretched = self._sample(profile, source_bins)
        stretched[~valid_source] = 0.0
        return stretched

    def _shift(
        self,
        profile: np.ndarray,
        shift: int,
    ) -> np.ndarray:
        """Shift a resampled profile by whole bins."""
        if shift == 0:
            return profile
        shifted = np.zeros_like(profile)
        if shift > 0:
            shifted[shift:] = profile[:-shift]
        else:
            shifted[:shift] = profile[-shift:]
        return shifted

    def _overlap(
        self,
        query_profile: np.ndarray,
        candidate_profile: np.ndarray,
    ) -> float:
        """Return area overlap over area union for two profiles."""
        overlap = np.minimum(query_profile, candidate_profile).sum()
        union = np.maximum(query_profile, candidate_profile).sum()
        return float(overlap / union) if union > 0 else 0.0
