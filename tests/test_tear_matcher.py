"""Tests for tear-profile matching."""

import numpy as np
import pytest

from elephant_id.matching.alphaphant.similarity import (
    EarProfileSimilarity,
    EarProfileSimilarityConfig,
)


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


class TestEarProfileSimilarity:
    def test_identical_profiles_score_one(self) -> None:
        profile = make_profile([(300, 0.05)])
        match = EarProfileSimilarity().compare((profile,), (profile,))
        assert match.score == pytest.approx(1.0)
        assert match.shift_bins == 0

    def test_disjoint_profiles_score_zero(self) -> None:
        query = make_profile([(120, 0.05)])
        candidate = make_profile([(600, 0.05)])
        assert EarProfileSimilarity().compare((query,), (candidate,)).score == pytest.approx(0.0)

    def test_depth_compression_tolerates_depth_mismatch(self) -> None:
        query = make_profile([(300, 0.02)])
        candidate = make_profile([(300, 0.08)])
        plain = (
            EarProfileSimilarity(EarProfileSimilarityConfig(depth_exponent=1.0))
            .compare((query,), (candidate,))
            .score
        )
        compressed = (
            EarProfileSimilarity(EarProfileSimilarityConfig(depth_exponent=0.5))
            .compare((query,), (candidate,))
            .score
        )
        assert compressed > plain

    def test_depth_compression_keeps_identical_profiles_at_one(self) -> None:
        profile = make_profile([(300, 0.05), (500, 0.02)])
        matcher = EarProfileSimilarity(EarProfileSimilarityConfig(depth_exponent=0.5))
        assert matcher.compare((profile,), (profile,)).score == pytest.approx(1.0)

    def test_default_config_is_validated_configuration(self) -> None:
        config = EarProfileSimilarityConfig()
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

        result = EarProfileSimilarity().compare((broad,), (narrow,))

        assert result.score == pytest.approx(0.5130298508958523)
        assert result.stretch == pytest.approx(0.8)
        assert result.shift_bins == 0
        assert not hasattr(result, "catalog_to_query")


def _on_grid(profile: np.ndarray, config: EarProfileSimilarityConfig) -> np.ndarray:
    """Compress one profile and interpolate it on the configured grid."""
    depths = np.maximum(np.asarray(profile, dtype=np.float64), 0.0) ** config.depth_exponent
    return np.interp(
        np.linspace(0, len(depths) - 1, config.resampled_bins), np.arange(len(depths)), depths
    )


def _aligned(
    profile: np.ndarray, stretch: float, shift: int, config: EarProfileSimilarityConfig
) -> np.ndarray:
    """Put one profile on the grid, then stretch and shift it with explicit padding."""
    grid = _on_grid(profile, config)
    source = ((np.linspace(0, 1, config.resampled_bins) - 0.5) / stretch + 0.5) * (
        config.resampled_bins - 1
    )
    stretched = np.interp(source, np.arange(len(grid)), grid, left=0, right=0)
    shifted = np.zeros_like(stretched)
    if abs(shift) < len(shifted):
        if shift >= 0:
            shifted[shift:] = stretched[: len(stretched) - shift]
        else:
            shifted[:shift] = stretched[-shift:]
    return shifted


def _overlap(query: np.ndarray, reference: np.ndarray) -> float:
    """Return `sum(min) / sum(max)`, or zero for two empty rows."""
    union = np.maximum(query, reference).sum()
    return float(np.minimum(query, reference).sum() / union) if union else 0.0


def _forward_reference(
    query: tuple[np.ndarray, ...],
    candidate: tuple[np.ndarray, ...],
    config: EarProfileSimilarityConfig,
) -> tuple[float, float, int]:
    """Search every alignment; average the alpha-scale overlaps, then apply the penalty."""
    max_shift = round(config.max_shift_fraction * config.resampled_bins)
    best = (0.0, 1.0, 0)
    for stretch in config.stretches:
        for shift in range(-max_shift, max_shift + 1):
            overlaps = [
                _overlap(_aligned(q, stretch, shift, config), _on_grid(r, config))
                for q, r in zip(query, candidate, strict=True)
            ]
            penalty = np.exp(
                -(
                    (abs(shift / config.resampled_bins) / config.shift_penalty_scale)
                    ** config.shift_penalty_power
                )
            )
            score = float(np.mean(overlaps) * penalty)
            if score > best[0]:
                best = (score, stretch, shift)
    return best


@pytest.mark.parametrize(
    "config",
    [
        EarProfileSimilarityConfig(),
        EarProfileSimilarityConfig(
            resampled_bins=31, stretches=(1.2, 0.8, 1.0), depth_exponent=1.0
        ),
        EarProfileSimilarityConfig(
            resampled_bins=20, max_shift_fraction=1.2, stretches=(0.6, 1.4)
        ),
        EarProfileSimilarityConfig(resampled_bins=17, max_shift_fraction=0, stretches=(1.0,)),
    ],
)
def test_bulk_matches_original_forward_search(config: EarProfileSimilarityConfig) -> None:
    """Custom grids retain directional numerical behavior."""
    rng = np.random.default_rng(42)
    query = rng.uniform(-0.02, 0.1, 73)
    candidates = [rng.uniform(-0.03, 0.1, 73) for _ in range(4)]
    candidates.append(np.zeros(73))
    actual = EarProfileSimilarity(config).compare_many(
        (query,), [(candidate,) for candidate in candidates]
    )
    for candidate, match in zip(candidates, actual, strict=True):
        score, stretch, shift = _forward_reference((query,), (candidate,), config)
        assert match.score == pytest.approx(score, abs=2e-15)
        assert (match.stretch, match.shift_bins) == (stretch, shift)


def test_bulk_empty_and_zero_profiles() -> None:
    """Empty catalogs and zero-overlap profiles have explicit neutral results."""
    matcher = EarProfileSimilarity(EarProfileSimilarityConfig(stretches=(0.8, 1.2)))
    assert matcher.compare_many((np.ones(9),), ()) == ()
    matches = matcher.compare_many((np.zeros(9),), [(np.zeros(9),), (np.ones(9),)])
    assert [(m.score, m.stretch, m.shift_bins) for m in matches] == [(0.0, 1.0, 0)] * 2


@pytest.mark.parametrize(
    "query,references",
    [((np.ones(9),), [(np.ones(9),), (np.ones(7),)]), ((np.ones(12),), [(np.ones(9),)])],
)
def test_profiles_of_unequal_length_are_rejected(
    query: tuple[np.ndarray, ...], references: list[tuple[np.ndarray, ...]]
) -> None:
    """Every compared tear profile must have the same angle samples."""
    with pytest.raises(ValueError, match="length"):
        EarProfileSimilarity().compare_many(query, references)


def _triangle(center: int, half_width: int, bins: int = 24) -> np.ndarray:
    """Return a unit-height triangular tear centered on one bin."""
    return np.maximum(0.0, 1.0 - np.abs(np.arange(bins) - center) / half_width)


def test_alpha_scales_share_the_alignment_that_maximizes_their_average() -> None:
    """Scales that disagree on alignment get one shared stretch and shift."""
    config = EarProfileSimilarityConfig(
        resampled_bins=24,
        max_shift_fraction=4 / 24,
        shift_penalty_scale=1.0,
        depth_exponent=1.0,
        stretches=(1.0,),
    )
    query = (_triangle(10, 5), _triangle(10, 5))
    reference = (_triangle(12, 5), _triangle(10, 3))
    small_scale_best = _forward_reference(query[:1], reference[:1], config)
    large_scale_best = _forward_reference(query[1:], reference[1:], config)

    result = EarProfileSimilarity(config).compare(query, reference)

    assert (small_scale_best[2], large_scale_best[2]) == (2, 0)
    assert (result.stretch, result.shift_bins) == (1.0, 2)
    assert result.score == pytest.approx(
        _forward_reference(query, reference, config)[0], abs=2e-15
    )
    assert result.score < (small_scale_best[0] + large_scale_best[0]) / 2


@pytest.mark.parametrize(
    "query,references",
    [
        ((), [()]),
        ((np.ones(9), np.ones(9)), [(np.ones(9),)]),
        ((np.ones(9),), [(np.ones(9),), (np.ones(9), np.ones(9))]),
    ],
)
def test_ears_with_different_alpha_scale_counts_are_rejected(
    query: tuple[np.ndarray, ...], references: list[tuple[np.ndarray, ...]]
) -> None:
    """Every ear needs one tear profile for each alpha scale."""
    with pytest.raises(ValueError, match="each alpha scale"):
        EarProfileSimilarity().compare_many(query, references)


def test_positive_alignment_ties_preserve_configured_search_order() -> None:
    """Equivalent positive alignments retain the first configured stretch."""
    config = EarProfileSimilarityConfig(
        resampled_bins=9, stretches=(1.2, 1.1), max_shift_fraction=0
    )
    match = EarProfileSimilarity(config).compare((np.ones(9),), (np.ones(9),))
    assert (match.score, match.stretch, match.shift_bins) == (1.0, 1.2, 0)
