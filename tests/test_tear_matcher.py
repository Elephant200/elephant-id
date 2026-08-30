"""Tests for tear-profile matching."""

import numpy as np
import pytest

from elephant_id.matching.tear_matcher import TearMatcher, TearMatcherConfig


def make_profile(peaks: list[tuple[int, float]], bins: int = 720) -> np.ndarray:
    """Return a profile with triangular peaks at (bin, depth) positions."""
    profile = np.zeros(bins)
    half_width = 10
    for center, depth in peaks:
        for offset in range(-half_width, half_width + 1):
            index = center + offset
            if 0 <= index < bins:
                weight = 1.0 - abs(offset) / half_width
                profile[index] = max(profile[index], depth * weight)
    return profile


class TestTearMatcher:
    def test_identical_profiles_score_one(self) -> None:
        profile = make_profile([(300, 0.05)])
        match = TearMatcher().match(profile, profile)
        assert match.score == pytest.approx(1.0)
        assert match.shift_bins == 0

    def test_disjoint_profiles_score_zero(self) -> None:
        query = make_profile([(120, 0.05)])
        candidate = make_profile([(600, 0.05)])
        assert TearMatcher().match(query, candidate).score == pytest.approx(0.0)

    def test_depth_compression_tolerates_depth_mismatch(self) -> None:
        query = make_profile([(300, 0.02)])
        candidate = make_profile([(300, 0.08)])
        plain = TearMatcher(
            TearMatcherConfig(depth_exponent=1.0)
        ).match(query, candidate).score
        compressed = TearMatcher(
            TearMatcherConfig(depth_exponent=0.5)
        ).match(query, candidate).score
        assert compressed > plain

    def test_depth_compression_keeps_identical_profiles_at_one(self) -> None:
        profile = make_profile([(300, 0.05), (500, 0.02)])
        matcher = TearMatcher(TearMatcherConfig(depth_exponent=0.5))
        assert matcher.match(profile, profile).score == pytest.approx(1.0)

    def test_default_config_is_validated_configuration(self) -> None:
        config = TearMatcherConfig()
        assert config.resampled_bins == 240
        assert config.depth_exponent == 0.5
        assert len(config.stretches) == 17
        assert config.stretches[0] == 0.8
        assert config.stretches[-1] == 1.2

    def test_symmetrizes_score_and_retains_only_query_to_catalog_alignment(
        self,
    ) -> None:
        """The score averages directions while provenance keeps the forward one."""
        broad = np.zeros(720)
        broad[280:321] = np.linspace(0.0, 0.12, 41)
        broad[321:361] = np.linspace(0.12, 0.0, 40)
        narrow = np.zeros(720)
        narrow[310:331] = np.linspace(0.0, 0.08, 21)
        narrow[331:351] = np.linspace(0.08, 0.0, 20)

        result = TearMatcher().match(broad, narrow)

        assert result.score == pytest.approx(0.5035144176874071)
        assert result.stretch == pytest.approx(0.8)
        assert result.shift_bins == 0
        assert not hasattr(result, "catalog_to_query")
