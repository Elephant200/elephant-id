"""Tests for tear-profile matching."""

import numpy as np
import pytest

from elephant_id.matching import TearMatcher, TearMatcherConfig


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


class TestTearMatcherConfig:
    def test_rejects_non_positive_depth_exponent(self) -> None:
        with pytest.raises(ValueError, match="depth_exponent"):
            TearMatcherConfig(depth_exponent=0.0)
        with pytest.raises(ValueError, match="depth_exponent"):
            TearMatcherConfig(depth_exponent=-0.5)

    def test_accepts_depth_exponent(self) -> None:
        assert TearMatcherConfig(depth_exponent=0.5).depth_exponent == 0.5


class TestTearMatcher:
    def test_identical_profiles_score_one(self) -> None:
        profile = make_profile([(300, 0.05)])
        match = TearMatcher().match_pair(profile, profile)
        assert match.score == pytest.approx(1.0)
        assert match.shift_bins == 0

    def test_disjoint_profiles_score_zero(self) -> None:
        query = make_profile([(120, 0.05)])
        candidate = make_profile([(600, 0.05)])
        assert TearMatcher().match_pair(query, candidate).score == pytest.approx(0.0)

    def test_depth_compression_tolerates_depth_mismatch(self) -> None:
        query = make_profile([(300, 0.02)])
        candidate = make_profile([(300, 0.08)])
        plain = TearMatcher(
            TearMatcherConfig(depth_exponent=1.0)
        ).match_pair(query, candidate).score
        compressed = TearMatcher(
            TearMatcherConfig(depth_exponent=0.5)
        ).match_pair(query, candidate).score
        assert compressed > plain

    def test_depth_compression_keeps_identical_profiles_at_one(self) -> None:
        profile = make_profile([(300, 0.05), (500, 0.02)])
        matcher = TearMatcher(TearMatcherConfig(depth_exponent=0.5))
        assert matcher.match_pair(profile, profile).score == pytest.approx(1.0)

    def test_default_config_is_validated_configuration(self) -> None:
        config = TearMatcherConfig()
        assert config.resampled_bins == 240
        assert config.depth_exponent == 0.5
        assert len(config.stretches) == 17
        assert config.stretches[0] == 0.8
        assert config.stretches[-1] == 1.2
