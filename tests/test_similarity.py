"""Tests for ear similarity."""

from uuid import UUID

import numpy as np
import pytest

from elephant_id.domain import Photo
from elephant_id.image.boxes import BoundingBox
from elephant_id.matching.alphaphant.extraction import AnalyzedEar, TearProfile
from elephant_id.matching.alphaphant.similarity import (
    EarComparator,
    EarComparatorConfig,
    EarSimilarity,
)

_PHOTO = Photo(
    photo_id=UUID("00000000-0000-4000-8000-000000000001"),
    sighting_id=UUID("00000000-0000-4000-8000-000000000002"),
)


def _ear(*depths: np.ndarray) -> AnalyzedEar:
    """Return one left ear with a tear profile for each depth array."""
    return AnalyzedEar(
        _PHOTO,
        "left",
        BoundingBox(0, 0, 4, 4),
        tuple(TearProfile(row, start_degrees=5.0, end_degrees=175.0) for row in depths),
    )


def _compare_one(
    query: AnalyzedEar,
    reference: AnalyzedEar,
    config: EarComparatorConfig | None = None,
) -> EarSimilarity:
    """Return the ear similarity of one query ear with one reference ear."""
    (similarity,) = EarComparator(config).compare_ears(query, (reference,))
    return similarity


def _make_profile(peaks: list[tuple[int, float]], bins: int = 720) -> np.ndarray:
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


def test_identical_profiles_score_one() -> None:
    """An ear compared with itself has ear similarity 1 at shift 0."""
    profile = _make_profile([(300, 0.05), (500, 0.02)])
    similarity = _compare_one(_ear(profile), _ear(profile))
    assert similarity.score == pytest.approx(1.0)
    assert similarity.shift_bins == 0


def test_disjoint_profiles_score_exactly_zero_with_neutral_alignment() -> None:
    """Tears that never meet give ear similarity 0 with stretch 1 and shift 0."""
    query = _make_profile([(120, 0.05)])
    reference = _make_profile([(600, 0.05)])
    similarity = _compare_one(_ear(query), _ear(reference))
    assert (similarity.score, similarity.stretch, similarity.shift_bins) == (0.0, 1.0, 0)


def test_rounding_noise_below_tolerance_scores_zero() -> None:
    """Nonuniform disjoint tears leave L1 rounding noise; the ear similarity is still zero."""
    rng = np.random.default_rng(72)
    query = np.zeros(240)
    reference = np.zeros(240)
    query[10:35] = rng.random(25)
    reference[200:230] = rng.random(30)
    similarity = _compare_one(_ear(query), _ear(reference))
    assert (similarity.score, similarity.stretch, similarity.shift_bins) == (0.0, 1.0, 0)


def test_square_root_halves_a_fourfold_depth_mismatch() -> None:
    """At the identity alignment, tears of one shape at 1/4 the depth score sqrt(1/4)."""
    query = _make_profile([(300, 0.02)])
    reference = _make_profile([(300, 0.08)])
    config = EarComparatorConfig(stretches=(1.0,), max_shift_fraction=0)
    assert _compare_one(_ear(query), _ear(reference), config).score == pytest.approx(0.5)


def test_default_config_values() -> None:
    """The default grid has 240 bins and 17 stretches from 0.8 to 1.2."""
    config = EarComparatorConfig()
    assert config.resampled_bins == 240
    assert len(config.stretches) == 17
    assert config.stretches[0] == 0.8
    assert config.stretches[-1] == 1.2


def test_broad_query_stretches_onto_narrow_reference() -> None:
    """Only the query is aligned; the reference stays fixed."""
    broad = np.zeros(720)
    broad[280:321] = np.linspace(0.0, 0.12, 41)
    broad[321:361] = np.linspace(0.12, 0.0, 40)
    narrow = np.zeros(720)
    narrow[310:331] = np.linspace(0.0, 0.08, 21)
    narrow[331:351] = np.linspace(0.08, 0.0, 20)

    similarity = _compare_one(_ear(broad), _ear(narrow))

    assert similarity.score == pytest.approx(0.5130298508958523)
    assert similarity.stretch == pytest.approx(0.8)
    assert similarity.shift_bins == 0


def _depth_row(profile: np.ndarray, config: EarComparatorConfig) -> np.ndarray:
    """Return the depth row of one profile."""
    depths = np.sqrt(np.maximum(np.asarray(profile, dtype=np.float64), 0.0))
    return np.interp(
        np.linspace(0, len(depths) - 1, config.resampled_bins), np.arange(len(depths)), depths
    )


def _aligned(
    profile: np.ndarray, stretch: float, shift: int, config: EarComparatorConfig
) -> np.ndarray:
    """Return the depth row of one profile, stretched and shifted with explicit padding."""
    grid = _depth_row(profile, config)
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


def _jaccard(query: np.ndarray, reference: np.ndarray) -> float:
    """Return `sum(min) / sum(max)`, or zero for two empty rows."""
    union = np.maximum(query, reference).sum()
    return float(np.minimum(query, reference).sum() / union) if union else 0.0


def _brute_force_similarity(
    query: tuple[np.ndarray, ...],
    reference: tuple[np.ndarray, ...],
    config: EarComparatorConfig,
) -> tuple[float, float, int]:
    """Search every alignment; average the Jaccard over scales, then apply the penalty."""
    max_shift = round(config.max_shift_fraction * config.resampled_bins)
    best = (0.0, 1.0, 0)
    for stretch in config.stretches:
        for shift in range(-max_shift, max_shift + 1):
            jaccards = [
                _jaccard(_aligned(q, stretch, shift, config), _depth_row(r, config))
                for q, r in zip(query, reference, strict=True)
            ]
            penalty = np.exp(
                -(
                    (abs(shift / config.resampled_bins) / config.shift_penalty_scale)
                    ** config.shift_penalty_power
                )
            )
            score = float(np.mean(jaccards) * penalty)
            if score > best[0]:
                best = (score, stretch, shift)
    return best


@pytest.mark.parametrize(
    "config",
    [
        EarComparatorConfig(),
        EarComparatorConfig(resampled_bins=31, stretches=(1.2, 0.8, 1.0)),
        EarComparatorConfig(resampled_bins=20, max_shift_fraction=1.2, stretches=(0.6, 1.4)),
        EarComparatorConfig(resampled_bins=17, max_shift_fraction=0, stretches=(1.0,)),
    ],
)
def test_ear_similarities_match_brute_force_search(config: EarComparatorConfig) -> None:
    """Custom grids give the score and alignment of an exhaustive search."""
    rng = np.random.default_rng(42)
    query = rng.uniform(-0.02, 0.1, 73)
    references = [rng.uniform(-0.03, 0.1, 73) for _ in range(4)]
    references.append(np.zeros(73))
    actual = EarComparator(config).compare_ears(
        _ear(query), [_ear(reference) for reference in references]
    )
    for reference, similarity in zip(references, actual, strict=True):
        score, stretch, shift = _brute_force_similarity((query,), (reference,), config)
        assert similarity.score == pytest.approx(score, abs=2e-15)
        assert (similarity.stretch, similarity.shift_bins) == (stretch, shift)


def test_no_references_and_zero_scores_have_neutral_results() -> None:
    """An empty catalog gives no results, and a zero score gets stretch 1 and shift 0."""
    comparator = EarComparator(EarComparatorConfig(stretches=(0.8, 1.2)))
    assert comparator.compare_ears(_ear(np.ones(9)), ()) == ()
    similarities = comparator.compare_ears(_ear(np.zeros(9)), [_ear(np.zeros(9)), _ear(np.ones(9))])
    assert [(s.score, s.stretch, s.shift_bins) for s in similarities] == [(0.0, 1.0, 0)] * 2


@pytest.mark.parametrize(
    "query,references",
    [
        (_ear(np.ones(9)), [_ear(np.ones(9)), _ear(np.ones(7))]),
        (_ear(np.ones(12)), [_ear(np.ones(9))]),
    ],
)
def test_profiles_of_unequal_length_are_rejected(
    query: AnalyzedEar, references: list[AnalyzedEar]
) -> None:
    """Every compared tear profile must have the same angle samples."""
    with pytest.raises(ValueError, match="length"):
        EarComparator().compare_ears(query, references)


def _triangle(center: int, half_width: int, bins: int = 24) -> np.ndarray:
    """Return a tear whose depth row is a unit-height triangle centered on one bin."""
    return np.maximum(0.0, 1.0 - np.abs(np.arange(bins) - center) / half_width) ** 2


def test_alpha_scales_share_the_alignment_that_maximizes_their_average() -> None:
    """Scales that disagree on alignment get one shared stretch and shift."""
    config = EarComparatorConfig(
        resampled_bins=24,
        max_shift_fraction=4 / 24,
        shift_penalty_scale=1.0,
        stretches=(1.0,),
    )
    query = (_triangle(10, 5), _triangle(10, 5))
    reference = (_triangle(12, 5), _triangle(10, 3))
    small_scale_best = _brute_force_similarity(query[:1], reference[:1], config)
    large_scale_best = _brute_force_similarity(query[1:], reference[1:], config)

    result = _compare_one(_ear(*query), _ear(*reference), config)

    assert (small_scale_best[2], large_scale_best[2]) == (2, 0)
    assert (result.stretch, result.shift_bins) == (1.0, 2)
    assert result.score == pytest.approx(
        _brute_force_similarity(query, reference, config)[0], abs=2e-15
    )
    assert result.score < (small_scale_best[0] + large_scale_best[0]) / 2


@pytest.mark.parametrize(
    "query,references",
    [
        (_ear(), [_ear()]),
        (_ear(np.ones(9), np.ones(9)), [_ear(np.ones(9))]),
        (_ear(np.ones(9)), [_ear(np.ones(9)), _ear(np.ones(9), np.ones(9))]),
    ],
)
def test_ears_with_different_alpha_scale_counts_are_rejected(
    query: AnalyzedEar, references: list[AnalyzedEar]
) -> None:
    """Every ear needs one tear profile for each alpha scale."""
    with pytest.raises(ValueError, match="each alpha scale"):
        EarComparator().compare_ears(query, references)


def test_positive_alignment_ties_preserve_configured_search_order() -> None:
    """Equivalent positive alignments keep the first configured stretch."""
    config = EarComparatorConfig(resampled_bins=9, stretches=(1.2, 1.1), max_shift_fraction=0)
    similarity = _compare_one(_ear(np.ones(9)), _ear(np.ones(9)), config)
    assert (similarity.score, similarity.stretch, similarity.shift_bins) == (1.0, 1.2, 0)
