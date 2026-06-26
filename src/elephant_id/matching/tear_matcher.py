"""Minimal tear-profile matcher for elephant ear re-identification.

A tear profile is a 1-D depth signal along one ear arc. Larger positive values
mean deeper inward tears; negative values are clipped away before matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TearMatcherConfig:
    """Parameters controlling resampling, shift penalties, and stretch search."""

    resampled_bins: int = 120
    max_shift_fraction: float = 0.15  # fraction of the resampled profile length
    shift_penalty_scale: float = 0.08
    shift_penalty_power: float = 8.0
    stretches: tuple[float, ...] = field(
        default=(0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)
    )


class TearMatcher:
    """Score sparse tear-depth profiles with centered stretch and penalized shift."""

    def __init__(self, config: TearMatcherConfig | None = None) -> None:
        """Create a matcher with the default configuration."""
        self.config = config or TearMatcherConfig()
        if self.config.resampled_bins <= 0:
            raise ValueError("resampled_bins must be positive")
        if self.config.max_shift_fraction < 0:
            raise ValueError("max_shift_fraction must be non-negative")
        if self.config.shift_penalty_scale <= 0:
            raise ValueError("shift_penalty_scale must be positive")
        if self.config.shift_penalty_power <= 0:
            raise ValueError("shift_penalty_power must be positive")
        if not self.config.stretches:
            raise ValueError("stretches must not be empty")
        if any(stretch <= 0 for stretch in self.config.stretches):
            raise ValueError("stretches must all be positive")

    def match_pair(
        self,
        query: np.ndarray,
        candidate: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Match one query/candidate pair and return length-1 result arrays."""
        query_rows = self._as_profile_batch(query, "query")
        candidate_rows = self._as_profile_batch(candidate, "candidate")
        if len(query_rows) != 1 or len(candidate_rows) != 1:
            raise ValueError("match_pair expects one query profile and one candidate profile")
        return self.match_row_pairs(query_rows, candidate_rows)

    def match_row_pairs(
        self,
        queries: np.ndarray,
        candidates: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Match query rows to candidate rows.

        Returns row-aligned arrays for score, distance, IoU, shift, stretch,
        and shift penalty. A single query row may broadcast to many candidates.
        """
        query_rows = self._as_profile_batch(queries, "queries")
        candidate_rows = self._as_profile_batch(candidates, "candidates")
        if query_rows.shape[1] != candidate_rows.shape[1]:
            raise ValueError("queries and candidates must have the same profile length")
        if len(query_rows) == 1 and len(candidate_rows) > 1:
            query_rows = np.broadcast_to(query_rows, candidate_rows.shape)
        elif query_rows.shape != candidate_rows.shape:
            raise ValueError("queries and candidates must have the same shape")

        query_resampled_profile = self._resample_profiles(query_rows)
        candidate_resampled_profile = self._resample_profiles(candidate_rows)

        profile_count = len(query_resampled_profile)
        best_score = np.zeros(profile_count, dtype=np.float64)
        best_iou = np.zeros(profile_count, dtype=np.float64)
        best_shift = np.zeros(profile_count, dtype=np.int32)
        best_stretch = np.ones(profile_count, dtype=np.float64)
        best_penalty = np.ones(profile_count, dtype=np.float64)

        max_shift = round(self.config.max_shift_fraction * self.config.resampled_bins)
        shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
        shift_penalties = np.exp(-(np.abs(shifts / self.config.resampled_bins) / self.config.shift_penalty_scale) ** self.config.shift_penalty_power)  # super-Gaussian shift penalty

        for stretch in self.config.stretches:
            stretched = self._stretch_profile(query_resampled_profile, stretch)

            for shift, penalty in zip(shifts, shift_penalties, strict=True):
                shifted = self._shift_profile(stretched, int(shift))
                overlap = np.minimum(shifted, candidate_resampled_profile).sum(axis=1)
                union = np.maximum(shifted, candidate_resampled_profile).sum(axis=1)
                iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
                score = iou * penalty

                update = score > best_score
                best_score[update] = score[update]
                best_iou[update] = iou[update]
                best_shift[update] = shift
                best_stretch[update] = stretch
                best_penalty[update] = penalty

        return {
            "score": best_score,
            "distance": 1.0 - best_score,
            "iou": best_iou,
            "shift_bins": best_shift,
            "shift_fraction": best_shift / self.config.resampled_bins,
            "stretch": best_stretch,
            "penalty": best_penalty,
        }

    def _as_profile_batch(self, profile: np.ndarray, name: str) -> np.ndarray:
        """Return one or more profiles as a 2-D float array."""
        rows = np.asarray(profile, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty 1-D or 2-D array")
        return rows

    def _resample_profiles(self, profile_rows: np.ndarray) -> np.ndarray:
        """Clip negative depths and average raw bins into matching bins."""
        if profile_rows.shape[1] % self.config.resampled_bins:
            raise ValueError(
                "profile length must be divisible by resampled_bins for even averaging"
            )
        raw_bins_per_resampled_bin = profile_rows.shape[1] // self.config.resampled_bins
        return np.maximum(profile_rows, 0.0).reshape(
            len(profile_rows),
            self.config.resampled_bins,
            raw_bins_per_resampled_bin,
        ).mean(axis=2)

    def _stretch_profile(self, resampled_profile: np.ndarray, stretch: float) -> np.ndarray:
        """Stretch 2-D resampled profile rows around the middle of the ear arc."""
        profile_arc = np.linspace(0.0, 1.0, self.config.resampled_bins)
        source_arc = stretch * (profile_arc - 0.5) + 0.5  # keep arc center fixed
        source_bin = source_arc * (self.config.resampled_bins - 1)
        valid_source = (
            (source_bin >= 0.0)
            & (source_bin <= self.config.resampled_bins - 1)
        )
        lower_bin = np.clip(
            np.floor(source_bin).astype(np.int32),
            0,
            self.config.resampled_bins - 1,
        )
        upper_bin = np.minimum(lower_bin + 1, self.config.resampled_bins - 1)
        upper_weight = source_bin - lower_bin
        stretched = (
            resampled_profile[:, lower_bin] * (1.0 - upper_weight)
            + resampled_profile[:, upper_bin] * upper_weight
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

    def match_gallery(
        self,
        query: np.ndarray,
        gallery: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Score one query profile against every gallery row and return rank order."""
        query_rows = np.asarray(query)
        if query_rows.ndim != 1:
            raise ValueError("query must be a single 1-D profile")
        gallery_rows = np.asarray(gallery)
        if gallery_rows.ndim != 2:
            raise ValueError("gallery must be a 2-D profile array")
        result = self.match_row_pairs(query_rows, gallery_rows)
        result["order"] = np.argsort(result["score"])[::-1]
        return result
