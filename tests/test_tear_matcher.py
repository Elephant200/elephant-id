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
        plain = (
            TearMatcher(TearMatcherConfig(depth_exponent=1.0))
            .match(query, candidate)
            .score
        )
        compressed = (
            TearMatcher(TearMatcherConfig(depth_exponent=0.5))
            .match(query, candidate)
            .score
        )
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

    def test_uses_only_query_to_catalog_score_and_alignment(
        self,
    ) -> None:
        """The forward score replaces the previously averaged directional scores."""
        broad = np.zeros(720)
        broad[280:321] = np.linspace(0.0, 0.12, 41)
        broad[321:361] = np.linspace(0.12, 0.0, 40)
        narrow = np.zeros(720)
        narrow[310:331] = np.linspace(0.0, 0.08, 21)
        narrow[331:351] = np.linspace(0.08, 0.0, 20)

        result = TearMatcher().match(broad, narrow)

        assert result.score == pytest.approx(0.5130298508958523)
        assert result.stretch == pytest.approx(0.8)
        assert result.shift_bins == 0
        assert not hasattr(result, "catalog_to_query")


def _forward_reference(
    query: np.ndarray, candidate: np.ndarray, config: TearMatcherConfig
) -> tuple[float, float, int]:
    """Characterize the original forward search with explicit shift padding."""

    def prepare(profile: np.ndarray) -> np.ndarray:
        """Compress and interpolate an input on the configured grid."""
        depths = (
            np.maximum(np.asarray(profile, dtype=np.float64), 0.0)
            ** config.depth_exponent
        )
        return np.interp(
            np.linspace(0, len(depths) - 1, config.resampled_bins),
            np.arange(len(depths)),
            depths,
        )

    query, candidate = prepare(query), prepare(candidate)
    max_shift = round(config.max_shift_fraction * config.resampled_bins)
    best = (0.0, 1.0, 0)
    for stretch in config.stretches:
        source = ((np.linspace(0, 1, config.resampled_bins) - 0.5) / stretch + 0.5) * (
            config.resampled_bins - 1
        )
        stretched = np.interp(source, np.arange(len(query)), query, left=0, right=0)
        for shift in range(-max_shift, max_shift + 1):
            shifted = np.zeros_like(stretched)
            if abs(shift) < len(shifted):
                if shift >= 0:
                    shifted[shift:] = stretched[: len(stretched) - shift]
                else:
                    shifted[:shift] = stretched[-shift:]
            union = np.maximum(shifted, candidate).sum()
            penalty = np.exp(
                -(
                    (abs(shift / config.resampled_bins) / config.shift_penalty_scale)
                    ** config.shift_penalty_power
                )
            )
            score = (
                np.minimum(shifted, candidate).sum() / union * penalty if union else 0.0
            )
            if score > best[0]:
                best = (float(score), stretch, shift)
    return best


@pytest.mark.parametrize(
    "config",
    [
        TearMatcherConfig(),
        TearMatcherConfig(
            resampled_bins=31, stretches=(1.2, 0.8, 1.0), depth_exponent=1.0
        ),
        TearMatcherConfig(
            resampled_bins=20, max_shift_fraction=1.2, stretches=(0.6, 1.4)
        ),
        TearMatcherConfig(resampled_bins=17, max_shift_fraction=0, stretches=(1.0,)),
    ],
)
def test_bulk_matches_original_forward_search(config: TearMatcherConfig) -> None:
    """Ragged profiles and custom grids retain directional numerical behavior."""
    rng = np.random.default_rng(42)
    query = rng.uniform(-0.02, 0.1, 73)
    candidates = [rng.uniform(-0.03, 0.1, n) for n in (1, 15, 73, 720)]
    candidates.append(np.zeros(29))
    actual = TearMatcher(config).match_many(query, candidates)
    for candidate, match in zip(candidates, actual, strict=True):
        score, stretch, shift = _forward_reference(query, candidate, config)
        assert match.score == pytest.approx(score, abs=2e-15)
        assert (match.stretch, match.shift_bins) == (stretch, shift)


def test_bulk_chunking_preserves_order_and_does_not_mutate_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunk boundaries leave scores, ordering, and caller-owned arrays unchanged."""
    import elephant_id.matching.tear_matcher as module

    rng = np.random.default_rng(12)
    query = rng.random(720)
    candidates = rng.random((15, 720))
    original = candidates.copy()
    query.setflags(write=False)
    candidates.setflags(write=False)
    matcher = TearMatcher()
    expected = matcher.match_many(query, candidates)
    monkeypatch.setattr(module, "_OVERLAP_WORKSPACE_BYTES", 1)
    assert matcher.match_many(query, candidates) == expected
    np.testing.assert_array_equal(candidates, original)
    assert matcher.match(query, candidates[0]) == expected[0]


def test_bulk_empty_and_zero_profiles() -> None:
    """Empty catalogs and zero-overlap profiles have explicit neutral results."""
    matcher = TearMatcher(TearMatcherConfig(stretches=(0.8, 1.2)))
    assert matcher.match_many(np.ones(9), ()) == ()
    matches = matcher.match_many(np.zeros(12), [np.zeros(9), np.ones(7)])
    assert [(m.score, m.stretch, m.shift_bins) for m in matches] == [(0.0, 1.0, 0)] * 2


def test_positive_alignment_ties_preserve_configured_search_order() -> None:
    """Equivalent positive alignments retain the first configured stretch."""
    config = TearMatcherConfig(
        resampled_bins=9, stretches=(1.2, 1.1), max_shift_fraction=0
    )
    match = TearMatcher(config).match(np.ones(9), np.ones(9))
    assert (match.score, match.stretch, match.shift_bins) == (1.0, 1.2, 0)


def test_scale_stack_cannot_choose_incompatible_alignments() -> None:
    """Two scales cannot each claim a different perfect tear correspondence."""
    config = TearMatcherConfig(
        resampled_bins=81,
        depth_exponent=1.0,
        stretches=(1.0,),
        max_shift_fraction=0.25,
        shift_penalty_scale=100.0,
    )
    profiles = np.zeros((4, 81))
    profiles[np.arange(4), [20, 50, 25, 45]] = 1.0
    matcher = TearMatcher(config)
    independent = (
        matcher.match(profiles[0], profiles[2]).score
        + matcher.match(profiles[1], profiles[3]).score
    ) / 2
    shared = matcher.match_stack(tuple(profiles[:2]), tuple(profiles[2:])).score
    assert independent == pytest.approx(1.0)
    assert shared == pytest.approx(0.5)


def test_signed_depth_change_distinguishes_rising_from_falling() -> None:
    """Equal slope magnitudes with opposite directions are not matching boundaries."""
    from dataclasses import replace

    config = TearMatcherConfig(
        resampled_bins=41,
        depth_exponent=1.0,
        stretches=(1.0,),
        max_shift_fraction=0.0,
        channel="depth_change",
    )
    rising = np.linspace(0.0, 1.0, 41)
    falling = 1.0 - rising
    magnitude = TearMatcher(config).match(rising, falling)
    signed = TearMatcher(replace(config, channel="signed_depth_change")).match(
        rising, falling
    )
    assert magnitude.score == pytest.approx(1.0)
    assert signed.score == 0.0
    assert (signed.stretch, signed.shift_bins) == (1.0, 0)
