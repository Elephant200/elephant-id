"""Tear-profile matcher for elephant ear re-identification.

A tear profile is a 1-D depth signal along one ear arc. Larger positive values
mean deeper inward tears.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TearMatcherConfig:
    """Parameters controlling resampling, shift penalties, and stretch search.

    ``depth_exponent`` compresses tear depths (``profile ** depth_exponent``)
    before overlap scoring. Values below 1 tolerate depth mismatch from ear
    foreshortening while preserving angular structure.

    Defaults are the configuration validated on the high-quality and filtered
    evaluation sets: 240 bins preserve narrow scallops that 120 bins smear
    into accidental impostor overlap, and the fine stretch grid improves ear
    width normalization.
    """

    resampled_bins: int = 240
    max_shift_fraction: float = 0.15  # fraction of the resampled profile length
    shift_penalty_scale: float = 0.08
    shift_penalty_power: float = 4.0
    depth_exponent: float = 0.5
    stretches: tuple[float, ...] = field(
        default=(
            0.800, 0.825, 0.850, 0.875, 0.900, 0.925, 0.950, 0.975, 1.000,
            1.025, 1.050, 1.075, 1.100, 1.125, 1.150, 1.175, 1.200,
        )
    )

    def __post_init__(self) -> None:
        """Validate matcher parameters."""
        if self.resampled_bins <= 0:
            raise ValueError("resampled_bins must be positive")
        if self.max_shift_fraction < 0:
            raise ValueError("max_shift_fraction must be non-negative")
        if self.shift_penalty_scale <= 0:
            raise ValueError("shift_penalty_scale must be positive")
        if self.shift_penalty_power <= 0:
            raise ValueError("shift_penalty_power must be positive")
        if self.depth_exponent <= 0:
            raise ValueError("depth_exponent must be positive")
        if not self.stretches:
            raise ValueError("stretches must not be empty")
        if any(stretch <= 0 for stretch in self.stretches):
            raise ValueError("stretches must all be positive")


@dataclass(frozen=True)
class TearMatch:
    """Best alignment for one query/candidate pair."""

    score: float
    distance: float
    overlap_score: float
    shift_bins: int
    shift_fraction: float
    stretch: float
    penalty: float


@dataclass(frozen=True)
class TearMatchBatch:
    """Row-aligned match results for paired query and candidate profiles."""

    score: np.ndarray
    distance: np.ndarray
    overlap_score: np.ndarray
    shift_bins: np.ndarray
    shift_fraction: np.ndarray
    stretch: np.ndarray
    penalty: np.ndarray


@dataclass(frozen=True)
class TearMatchGallery:
    """Gallery match results plus descending score order."""

    score: np.ndarray
    distance: np.ndarray
    overlap_score: np.ndarray
    shift_bins: np.ndarray
    shift_fraction: np.ndarray
    stretch: np.ndarray
    penalty: np.ndarray
    order: np.ndarray


class TearMatcher:
    """Score sparse tear-depth profiles with centered stretch and penalized shift."""

    def __init__(self, config: TearMatcherConfig | None = None) -> None:
        """Create a matcher with the default configuration."""
        self.config = config or TearMatcherConfig()

    def match_pair(
        self,
        query: np.ndarray,
        candidate: np.ndarray,
    ) -> TearMatch:
        """Match one query/candidate pair and return scalar result values."""
        query_rows = self._as_profile_batch(query, "query")
        candidate_rows = self._as_profile_batch(candidate, "candidate")
        if len(query_rows) != 1 or len(candidate_rows) != 1:
            raise ValueError("match_pair expects one query profile and one candidate profile")
        result = self.match_row_pairs(query_rows, candidate_rows)
        return TearMatch(
            score=float(result.score[0]),
            distance=float(result.distance[0]),
            overlap_score=float(result.overlap_score[0]),
            shift_bins=int(result.shift_bins[0]),
            shift_fraction=float(result.shift_fraction[0]),
            stretch=float(result.stretch[0]),
            penalty=float(result.penalty[0]),
        )

    def match_row_pairs(
        self,
        queries: np.ndarray,
        candidates: np.ndarray,
    ) -> TearMatchBatch:
        """Match query rows to candidate rows.

        Returns row-aligned arrays for score, distance, overlap score, shift,
        stretch, and shift penalty.
        """
        query_rows = self._as_profile_batch(queries, "queries")
        candidate_rows = self._as_profile_batch(candidates, "candidates")
        if query_rows.shape != candidate_rows.shape:
            raise ValueError("queries and candidates must have the same shape")

        # Outward bulges are not useful tear evidence for this matcher.
        query_rows = self._compress_depths(self._clip_negative_depths(query_rows))
        candidate_rows = self._compress_depths(self._clip_negative_depths(candidate_rows))

        query_resampled_profile = self._resample_profiles(query_rows)
        candidate_resampled_profile = self._resample_profiles(candidate_rows)

        profile_count = len(query_resampled_profile)
        best_score = np.zeros(profile_count, dtype=np.float64)
        best_overlap_score = np.zeros(profile_count, dtype=np.float64)
        best_shift = np.zeros(profile_count, dtype=np.int32)
        best_stretch = np.ones(profile_count, dtype=np.float64)
        best_penalty = np.ones(profile_count, dtype=np.float64)

        max_shift = round(self.config.max_shift_fraction * self.config.resampled_bins)
        shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
        shift_fractions = shifts / self.config.resampled_bins
        shift_penalties = np.exp(
            -((np.abs(shift_fractions) / self.config.shift_penalty_scale) ** self.config.shift_penalty_power)
        )  # flat-topped super-Gaussian shift penalty

        for stretch in self.config.stretches:
            stretched = self._stretch_profile(query_resampled_profile, stretch)

            for shift, penalty in zip(shifts, shift_penalties, strict=True):
                shifted = self._shift_profile(stretched, int(shift))
                overlap_score = self._profile_overlap(shifted, candidate_resampled_profile)
                score = overlap_score * penalty

                update = score > best_score
                best_score[update] = score[update]
                best_overlap_score[update] = overlap_score[update]
                best_shift[update] = shift
                best_stretch[update] = stretch
                best_penalty[update] = penalty

        return TearMatchBatch(
            score=best_score,
            distance=1.0 - best_score,
            overlap_score=best_overlap_score,
            shift_bins=best_shift,
            shift_fraction=best_shift / self.config.resampled_bins,
            stretch=best_stretch,
            penalty=best_penalty,
        )

    def _as_profile_batch(self, profile: np.ndarray, name: str) -> np.ndarray:
        """Return one or more profiles as a 2-D float array."""
        rows = np.asarray(profile, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty 1-D or 2-D array")
        return rows

    def _clip_negative_depths(self, profile_rows: np.ndarray) -> np.ndarray:
        """Discard outward-depth signal before matching."""
        return np.maximum(profile_rows, 0.0)

    def _compress_depths(self, profile_rows: np.ndarray) -> np.ndarray:
        """Apply the configured depth compression to non-negative depths."""
        if self.config.depth_exponent == 1.0:
            return profile_rows
        return profile_rows**self.config.depth_exponent

    def _resample_profiles(self, profile_rows: np.ndarray) -> np.ndarray:
        """Linearly resample profile rows to the matching resolution."""
        source_bins = np.linspace(0.0, profile_rows.shape[1] - 1, self.config.resampled_bins)
        lower_bins = np.floor(source_bins).astype(np.int32)
        upper_bins = np.minimum(lower_bins + 1, profile_rows.shape[1] - 1)
        upper_weight = source_bins - lower_bins
        return (
            profile_rows[:, lower_bins] * (1.0 - upper_weight)
            + profile_rows[:, upper_bins] * upper_weight
        )

    def _stretch_profile(self, resampled_profile: np.ndarray, stretch: float) -> np.ndarray:
        """Sample profiles with a centered stretch transform."""
        output_arc = np.linspace(0.0, 1.0, self.config.resampled_bins)
        source_arc = (output_arc - 0.5) / stretch + 0.5  # keep arc center fixed
        source_bins = source_arc * (self.config.resampled_bins - 1)
        valid_source = (
            (source_bins >= 0.0)
            & (source_bins <= self.config.resampled_bins - 1)
        )
        unclipped_lower_bins = np.floor(source_bins).astype(np.int32)
        lower_bins = np.clip(
            unclipped_lower_bins,
            0,
            self.config.resampled_bins - 1,
        )
        upper_bins = np.minimum(lower_bins + 1, self.config.resampled_bins - 1)
        upper_weight = source_bins - unclipped_lower_bins
        stretched = (
            resampled_profile[:, lower_bins] * (1.0 - upper_weight)
            + resampled_profile[:, upper_bins] * upper_weight
        )
        stretched[:, ~valid_source] = 0.0
        return stretched

    def _shift_profile(self, resampled_profile: np.ndarray, shift: int) -> np.ndarray:
        """Shift 2-D resampled profile rows by whole bins."""
        shifted = np.zeros_like(resampled_profile)
        if shift > 0:
            shifted[:, shift:] = resampled_profile[:, :-shift]
        elif shift < 0:
            shifted[:, :shift] = resampled_profile[:, -shift:]
        else:
            shifted[:, :] = resampled_profile
        return shifted

    def _profile_overlap(
        self,
        query_profile: np.ndarray,
        candidate_profile: np.ndarray,
    ) -> np.ndarray:
        """Return area overlap over area union for each profile row."""
        overlap = np.minimum(query_profile, candidate_profile).sum(axis=1)
        union = np.maximum(query_profile, candidate_profile).sum(axis=1)
        return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)

    def match_gallery(
        self,
        query: np.ndarray,
        gallery: np.ndarray,
    ) -> TearMatchGallery:
        """Score one query profile against every gallery row and return rank order."""
        query_rows = np.asarray(query)
        if query_rows.ndim != 1:
            raise ValueError("query must be a single 1-D profile")
        gallery_rows = np.asarray(gallery)
        if gallery_rows.ndim != 2:
            raise ValueError("gallery must be a 2-D profile array")

        query_batch = np.broadcast_to(query_rows[None, :], gallery_rows.shape)
        result = self.match_row_pairs(query_batch, gallery_rows)
        return TearMatchGallery(
            score=result.score,
            distance=result.distance,
            overlap_score=result.overlap_score,
            shift_bins=result.shift_bins,
            shift_fraction=result.shift_fraction,
            stretch=result.stretch,
            penalty=result.penalty,
            order=np.argsort(result.score)[::-1],
        )
