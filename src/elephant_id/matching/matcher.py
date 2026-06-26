"""Minimal tear-profile matcher for elephant ear re-identification."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MatcherConfig:
    """Parameters controlling resampling, shift penalties, and stretch search."""

    resampled_bins: int = 120
    max_shift_fraction: float = 0.15
    shift_penalty_scale: float = 0.08
    shift_penalty_power: float = 8.0
    stretches: tuple[float, ...] = field(
        default=(0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15)
    )


class TearMatcher:
    """Score sparse tear-depth profiles with centered stretch and penalized shift."""

    def __init__(self, config: MatcherConfig | None = None) -> None:
        """Create a matcher with the settled configuration."""
        self.config = config or MatcherConfig()
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
        """Return the best shifted/stretched score for each query-candidate row pair."""
        query_rows = np.asarray(query, dtype=np.float64)
        candidate_rows = np.asarray(candidate, dtype=np.float64)
        if query_rows.ndim == 1:
            query_rows = query_rows[None, :]  # one profile -> one-row batch
        if candidate_rows.ndim == 1:
            candidate_rows = candidate_rows[None, :]  # one profile -> one-row batch
        if query_rows.ndim != 2 or query_rows.shape[0] == 0 or query_rows.shape[1] == 0:
            raise ValueError("query must be a non-empty 1-D or 2-D array")
        if (
            candidate_rows.ndim != 2
            or candidate_rows.shape[0] == 0
            or candidate_rows.shape[1] == 0
        ):
            raise ValueError("candidate must be a non-empty 1-D or 2-D array")

        if query_rows.shape != candidate_rows.shape:
            raise ValueError("query and candidate must have the same shape")

        if query_rows.shape[1] % self.config.resampled_bins:
            raise ValueError("profile length must be divisible by resampled_bins")
        raw_bins_per_resampled_bin = query_rows.shape[1] // self.config.resampled_bins

        # Negative depths are not useful tear evidence; average raw bins into
        # the smaller profile used for matching.
        query_resampled_profile = np.maximum(query_rows, 0.0).reshape(
            len(query_rows),
            self.config.resampled_bins,
            raw_bins_per_resampled_bin,
        ).mean(axis=2)
        candidate_resampled_profile = np.maximum(candidate_rows, 0.0).reshape(
            len(candidate_rows),
            self.config.resampled_bins,
            raw_bins_per_resampled_bin,
        ).mean(axis=2)

        profile_count = len(query_resampled_profile)
        best_score = np.zeros(profile_count, dtype=np.float64)  # initializes best scores to zero
        best_iou = np.zeros(profile_count, dtype=np.float64)
        best_shift = np.zeros(profile_count, dtype=np.int32)
        best_stretch = np.ones(profile_count, dtype=np.float64)  # neutral stretch
        best_penalty = np.ones(profile_count, dtype=np.float64)  # neutral penalty

        max_shift = round(self.config.max_shift_fraction * self.config.resampled_bins)  # maximum shift in bins
        shifts = np.arange(-max_shift, max_shift + 1, dtype=np.int32)
        shift_penalties = np.exp(-(np.abs(shifts / self.config.resampled_bins) / self.config.shift_penalty_scale) ** self.config.shift_penalty_power)  # super-Gaussian shift penalty

        for stretch in self.config.stretches:
            stretched = self.stretch(query_resampled_profile, stretch)

            for shift, penalty in zip(shifts, shift_penalties, strict=True):
                shifted = self.shift(stretched, int(shift))
                overlap = np.minimum(shifted, candidate_resampled_profile).sum(axis=1)
                union = np.maximum(shifted, candidate_resampled_profile).sum(axis=1)
                profile_iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
                score = profile_iou * penalty  # penalize large shifts

                update = score > best_score  # keep best transform per profile
                best_score[update] = score[update]
                best_iou[update] = profile_iou[update]
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

    def stretch(self, resampled_profile: np.ndarray, stretch: float) -> np.ndarray:
        """Stretch resampled profiles around the middle of the ear arc."""
        profile_arc = np.linspace(0.0, 1.0, self.config.resampled_bins)  # 0..180 deg as 0..1
        source_arc = stretch * (profile_arc - 0.5) + 0.5  # keep arc center fixed
        source_bin = source_arc * (self.config.resampled_bins - 1)  # arc position -> bin
        valid_source = (
            (source_bin >= 0.0)
            & (source_bin <= self.config.resampled_bins - 1)
        )  # off-arc samples become zero
        lower_bin = np.clip(
            np.floor(source_bin).astype(np.int32),
            0,
            self.config.resampled_bins - 1,
        )  # bin below source
        upper_bin = np.minimum(lower_bin + 1, self.config.resampled_bins - 1)
        upper_weight = source_bin - lower_bin  # fraction from lower to upper
        stretched = (
            resampled_profile[:, lower_bin] * (1.0 - upper_weight)
            + resampled_profile[:, upper_bin] * upper_weight
        )  # linearly sampled profile
        stretched[:, ~valid_source] = 0.0
        return stretched

    def shift(self, resampled_profile: np.ndarray, shift: int) -> np.ndarray:
        """Shift profiles by whole bins; positive moves toward larger bin indices."""
        shifted = np.zeros_like(resampled_profile)
        if shift > 0:
            shifted[:, shift:] = resampled_profile[:, :-shift]  # larger angles
        elif shift < 0:
            shifted[:, :shift] = resampled_profile[:, -shift:]  # smaller angles
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
        query_batch = np.repeat(query_rows[None, :], len(gallery_rows), axis=0)
        result = self.match_pair(query_batch, gallery_rows)
        result["order"] = np.argsort(result["score"])[::-1]
        return result
