"""Tests for ear similarity and side scores."""

import math
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
    log_mean_exp,
    weighted_real_valued_jaccard,
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


def test_broad_query_stretches_onto_narrow_reference() -> None:
    """Only the query is aligned; the reference stays fixed."""
    broad = np.zeros(720)
    broad[280:321] = np.linspace(0.0, 0.12, 41)
    broad[321:361] = np.linspace(0.12, 0.0, 40)
    narrow = np.zeros(720)
    narrow[310:331] = np.linspace(0.0, 0.08, 21)
    narrow[331:351] = np.linspace(0.08, 0.0, 20)

    config = EarComparatorConfig(
        gradient_weight=0.0,
        stretches=tuple(round(0.8 + index * 0.025, 3) for index in range(17)),
        stretch_penalty_scale=math.inf,
    )

    similarity = _compare_one(_ear(broad), _ear(narrow), config)

    assert similarity.score == pytest.approx(0.5551403753377685)
    assert similarity.stretch == pytest.approx(0.8)
    assert similarity.shift_bins == -3


def _measured_ramp(slope: float, bins: int = 720) -> np.ndarray:
    """Return depths that rise linearly with `slope` per sample inside 5° to 175°, and zero outside."""
    angles = np.linspace(0.0, 180.0, bins)
    measured = (angles > 5.0) & (angles < 175.0)
    depths = np.zeros(bins)
    depths[measured] = 0.01 + slope * np.arange(measured.sum())
    return depths


def test_gradient_row_of_a_linear_profile_is_constant_inside_the_measured_interval() -> None:
    """The gradient row has no step at 5° or 175°, and it is zero outside the measured interval."""
    profile = _measured_ramp(slope=0.0004)
    comparator = EarComparator(EarComparatorConfig(resampled_bins=720))

    (row,) = comparator._build_gradient_rows((TearProfile(profile, 5.0, 175.0),))

    measured = profile != 0.0
    np.testing.assert_allclose(row[measured], np.sqrt(0.0004), rtol=1e-12)
    assert np.all(row[~measured] == 0.0)


def test_gradient_row_ignores_negative_depths() -> None:
    """The gradient row is the gradient of the clipped depth, so a dip below zero leaves no trace."""
    clipped = np.zeros(720)
    clipped[300:400] = 0.05
    signed = clipped.copy()
    signed[100:200] = -0.05
    comparator = EarComparator()

    (from_signed,) = comparator._build_gradient_rows((TearProfile(signed, 5.0, 175.0),))
    (from_clipped,) = comparator._build_gradient_rows((TearProfile(clipped, 5.0, 175.0),))

    np.testing.assert_array_equal(from_signed, from_clipped)
    assert np.any(from_signed > 0) and np.any(from_signed < 0)


def test_aligned_gradient_row_is_the_gradient_of_the_stretched_profile() -> None:
    """Under stretch `t`, the aligned query gradient row is scaled by `t ** -0.5`.

    The reference is the query profile stretched by 1.25 about 90°, so its
    gradient is 1/1.25 as steep. Without the factor, the gradient rows would
    score about `1.25 ** -0.5`, which is 0.89.
    """
    stretch = 1.25
    query = np.zeros(720)
    query[300:360] = np.linspace(0.0, 0.06, 60)
    query[360:420] = np.linspace(0.06, 0.0, 60)
    source = (np.arange(720) - 359.5) / stretch + 359.5
    reference = np.interp(source, np.arange(720), query)
    config = EarComparatorConfig(
        gradient_weight=1.0, stretches=(stretch,), max_shift_fraction=0, stretch_penalty_scale=math.inf
    )

    similarity = _compare_one(_ear(query), _ear(reference), config)

    assert similarity.score == pytest.approx(1.0, abs=0.02)


def _depth_row(profile: np.ndarray, config: EarComparatorConfig) -> np.ndarray:
    """Return the depth row of one profile."""
    depths = np.sqrt(np.maximum(np.asarray(profile, dtype=np.float64), 0.0))
    return _resample(depths, config)


def _gradient_row(profile: np.ndarray, config: EarComparatorConfig) -> np.ndarray:
    """Return the compressed gradient of the clipped depth inside 5° to 175°, resampled to the grid."""
    angles = np.linspace(0.0, 180.0, len(profile))
    measured = (angles > 5.0) & (angles < 175.0)
    gradient = np.zeros(len(profile))
    gradient[measured] = np.gradient(np.maximum(profile[measured], 0.0))
    return _resample(np.sign(gradient) * np.sqrt(np.abs(gradient)), config)


def _resample(values: np.ndarray, config: EarComparatorConfig) -> np.ndarray:
    """Return the values linearly interpolated to the comparison grid."""
    return np.interp(
        np.linspace(0, len(values) - 1, config.resampled_bins), np.arange(len(values)), values
    )


def _aligned(row: np.ndarray, stretch: float, shift: int) -> np.ndarray:
    """Return the row stretched about its center and shifted, with explicit zero padding."""
    bins = len(row)
    source = ((np.linspace(0, 1, bins) - 0.5) / stretch + 0.5) * (bins - 1)
    stretched = np.interp(source, np.arange(bins), row, left=0, right=0)
    shifted = np.zeros_like(stretched)
    if abs(shift) < len(shifted):
        if shift >= 0:
            shifted[shift:] = stretched[: len(stretched) - shift]
        else:
            shifted[:shift] = stretched[-shift:]
    return shifted


def _angle_weights(config: EarComparatorConfig) -> np.ndarray:
    """Return the Gaussian angle weight at each angle of the comparison grid."""
    angles = np.linspace(0.0, 180.0, config.resampled_bins)
    return np.exp(
        -0.5
        * ((angles - config.angle_weight_center_degrees) / config.angle_weight_width_degrees) ** 2
    )


def _jaccard(query: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted real-valued Jaccard of two rows from their positive and negative parts.

    The value is zero for two empty rows.
    """
    shared = np.minimum(np.maximum(query, 0), np.maximum(reference, 0)) + np.minimum(
        np.maximum(-query, 0), np.maximum(-reference, 0)
    )
    combined = np.maximum(np.maximum(query, 0), np.maximum(reference, 0)) + np.maximum(
        np.maximum(-query, 0), np.maximum(-reference, 0)
    )
    weighted_combined = float(weights @ combined)
    return float(weights @ shared) / weighted_combined if weighted_combined else 0.0


def _brute_force_similarity(
    query: tuple[np.ndarray, ...],
    reference: tuple[np.ndarray, ...],
    config: EarComparatorConfig,
) -> tuple[float, float, int]:
    """Search every alignment; combine the channel means of the weighted Jaccards, then apply the penalty."""
    max_shift = round(config.max_shift_fraction * config.resampled_bins)
    weights = _angle_weights(config)
    best = (0.0, 1.0, 0)
    for stretch in config.stretches:
        for shift in range(-max_shift, max_shift + 1):
            depth_jaccards = [
                _jaccard(
                    _aligned(_depth_row(q, config), stretch, shift), _depth_row(r, config), weights
                )
                for q, r in zip(query, reference, strict=True)
            ]
            gradient_jaccards = [
                _jaccard(
                    stretch**-0.5 * _aligned(_gradient_row(q, config), stretch, shift),
                    _gradient_row(r, config),
                    weights,
                )
                for q, r in zip(query, reference, strict=True)
            ]
            penalty = np.exp(
                -(
                    (abs(shift / config.resampled_bins) / config.shift_penalty_scale)
                    ** config.shift_penalty_power
                )
            ) * np.exp(
                -((abs(np.log(stretch)) / config.stretch_penalty_scale) ** config.stretch_penalty_power)
            )
            channel_score = (1 - config.gradient_weight) * np.mean(
                depth_jaccards
            ) + config.gradient_weight * np.mean(gradient_jaccards)
            score = float(channel_score * penalty)
            if score > best[0]:
                best = (score, stretch, shift)
    return best


@pytest.mark.parametrize(
    "config",
    [
        EarComparatorConfig(),
        EarComparatorConfig(gradient_weight=0.0),
        EarComparatorConfig(gradient_weight=1.0),
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
        gradient_weight=0.0,
    )
    query = (_triangle(10, 5), _triangle(10, 5))
    reference = (_triangle(12, 5), _triangle(10, 3))
    small_scale_best = _brute_force_similarity(query[:1], reference[:1], config)
    large_scale_best = _brute_force_similarity(query[1:], reference[1:], config)

    result = _compare_one(_ear(*query), _ear(*reference), config)

    assert (small_scale_best[2], large_scale_best[2]) == (2, -1)
    assert (result.stretch, result.shift_bins) == (1.0, 2)
    assert result.score == pytest.approx(
        _brute_force_similarity(query, reference, config)[0], abs=2e-15
    )
    assert result.score < (small_scale_best[0] + large_scale_best[0]) / 2


def _random_tears(rng: np.random.Generator, bins: int = 240) -> np.ndarray:
    """Return a profile of three triangular tears with random places, widths, and depths."""
    profile = np.zeros(bins)
    for _ in range(3):
        center, half_width = rng.integers(40, 200), rng.integers(4, 15)
        offsets = np.arange(-half_width, half_width)
        profile[center + offsets] += rng.uniform(0.02, 0.1) * (1 - np.abs(offsets) / half_width)
    return profile


def test_depth_and_gradient_rows_share_the_alignment_that_maximizes_their_sum() -> None:
    """Channels that disagree on alignment get one shared stretch and shift."""
    rng = np.random.default_rng(50)
    query, reference = _ear(_random_tears(rng)), _ear(_random_tears(rng))

    def config(gradient_weight: float) -> EarComparatorConfig:
        """Return a shift-only grid with a flat shift penalty."""
        return EarComparatorConfig(
            stretches=(1.0,), shift_penalty_scale=1.0, gradient_weight=gradient_weight
        )

    depth_best = _compare_one(query, reference, config(0.0))
    gradient_best = _compare_one(query, reference, config(1.0))

    result = _compare_one(query, reference, config(0.5))

    assert depth_best.shift_bins != gradient_best.shift_bins
    expected = _brute_force_similarity(
        (query.profiles_by_scale[0].depths,), (reference.profiles_by_scale[0].depths,), config(0.5)
    )
    assert (result.score, result.stretch, result.shift_bins) == pytest.approx(expected, abs=2e-15)
    assert result.score < (depth_best.score + gradient_best.score) / 2


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


_UNEVEN_WEIGHTS = np.linspace(0.2, 1.0, 12)


def test_weighted_jaccard_is_one_for_identical_rows_and_zero_for_rows_that_share_nothing() -> None:
    """Identical nonzero rows give 1. Rows with no common nonzero bin of the same sign give 0."""
    row = np.array([0.0, 0.3, -0.2, 0.5, 0.0, 0.0, 0.1, 0.0, 0.0, -0.4, 0.0, 0.0])
    disjoint = np.zeros(12)
    disjoint[[0, 4, 5]] = [0.2, 0.7, -0.1]

    values = weighted_real_valued_jaccard(
        row[None, :], np.array([row, disjoint, -row]), _UNEVEN_WEIGHTS
    )

    np.testing.assert_allclose(values, [[1.0, 0.0, 0.0]], atol=1e-15)


def test_weighted_jaccard_is_unchanged_when_both_rows_are_scaled_by_one_constant() -> None:
    """The weighted real-valued Jaccard depends on the shape of the rows, not on their size."""
    rng = np.random.default_rng(7)
    row_a, row_b = rng.uniform(-0.1, 0.1, (2, 12))

    original = weighted_real_valued_jaccard(row_a[None, :], row_b[None, :], _UNEVEN_WEIGHTS)
    scaled = weighted_real_valued_jaccard(3.7 * row_a[None, :], 3.7 * row_b[None, :], _UNEVEN_WEIGHTS)

    assert 0.0 < original[0, 0] < 1.0
    assert scaled[0, 0] == pytest.approx(original[0, 0], rel=1e-12)


def test_a_rise_against_a_fall_shares_nothing_and_counts_both_magnitudes() -> None:
    """A bin with opposite signs adds both magnitudes to the denominator and nothing to the numerator."""
    rise = np.zeros(12)
    rise[3] = 0.4
    fall = np.zeros(12)
    fall[3] = -0.1
    rise_and_shared, fall_and_shared = rise.copy(), fall.copy()
    rise_and_shared[8] = fall_and_shared[8] = 0.2
    weights = _UNEVEN_WEIGHTS

    values = weighted_real_valued_jaccard(
        np.array([fall, fall_and_shared]), np.array([rise, rise_and_shared]), weights
    )

    assert values[0, 0] == 0.0
    expected = weights[8] * 0.2 / (weights[8] * 0.2 + weights[3] * (0.4 + 0.1))
    assert values[1, 1] == pytest.approx(expected, rel=1e-12)


def test_angle_weight_stays_at_fixed_angles_when_the_query_shifts() -> None:
    """The angle weight applies after alignment, at the angles of the comparison grid.

    The reference has the query tear 15 bins later, and an extra tear at
    about 150°. The score at shift 15 uses the weight at the grid angles. A weight
    that moved with the query would give a clearly different score.
    """
    config = EarComparatorConfig(
        stretches=(1.0,), max_shift_fraction=0.1, shift_penalty_scale=1.0, gradient_weight=0.0
    )
    comparator = EarComparator(config)
    shift_bins = 15
    query = (TearProfile(_make_profile([(100, 0.05)], bins=240), 5.0, 175.0),)
    reference = (TearProfile(_make_profile([(100 + shift_bins, 0.05), (200, 0.05)], bins=240), 5.0, 175.0),)
    query_depth_rows = comparator._build_depth_rows(query)
    reference_depth_rows = comparator._build_depth_rows(reference)

    scores = comparator._score_alignments(
        query_depth_rows,
        comparator._build_gradient_rows(query),
        reference_depth_rows[None],
        comparator._build_gradient_rows(reference)[None],
    )

    score = scores[0, comparator._alignments.index((1.0, shift_bins))]
    aligned_query_row = _aligned(query_depth_rows[0], 1.0, shift_bins)
    penalty = comparator._compute_shift_penalty(shift_bins)
    weights = _angle_weights(config)
    fixed = penalty * _jaccard(aligned_query_row, reference_depth_rows[0], weights)
    moved = penalty * _jaccard(
        aligned_query_row, reference_depth_rows[0], _aligned(weights, 1.0, shift_bins)
    )
    assert score == pytest.approx(fixed, rel=1e-12)
    assert abs(fixed - moved) > 0.1


def test_a_disagreement_near_120_degrees_costs_more_than_at_a_low_weight_angle() -> None:
    """An extra query tear at 120° lowers the ear similarity more than the same tear at 30°."""
    config = EarComparatorConfig(stretches=(1.0,), max_shift_fraction=0)
    shared = _make_profile([(300, 0.05), (600, 0.03)])
    extra_near_120 = shared + _make_profile([(480, 0.04)])
    extra_near_30 = shared + _make_profile([(120, 0.04)])

    near_120 = _compare_one(_ear(extra_near_120), _ear(shared), config)
    near_30 = _compare_one(_ear(extra_near_30), _ear(shared), config)

    assert near_120.score < near_30.score - 0.05


@pytest.mark.parametrize("count", [1, 2, 7])
def test_log_mean_exp_of_equal_values_is_their_value(count: int) -> None:
    """Equal values give their shared value, for any number of values."""
    assert log_mean_exp(np.full(count, 0.6), temperature=0.05) == pytest.approx(0.6)


@pytest.mark.parametrize("raised_index", [0, 1, 2])
def test_log_mean_exp_rises_when_any_value_rises(raised_index: int) -> None:
    """A rise in any one value raises the log-mean-exp."""
    values = np.array([0.2, 0.5, 0.8])
    raised = values.copy()
    raised[raised_index] += 0.05

    assert log_mean_exp(raised, temperature=0.05) > log_mean_exp(values, temperature=0.05)


def test_log_mean_exp_is_the_temperature_times_the_log_of_the_mean_exponential() -> None:
    """The log-mean-exp is `τ * log(mean(exp(s / τ)))`."""
    values = np.array([0.2, 0.5, 0.8])

    assert log_mean_exp(values, temperature=0.05) == pytest.approx(
        0.05 * np.log(np.mean(np.exp(values / 0.05)))
    )


def test_log_mean_exp_lies_between_the_mean_and_the_maximum() -> None:
    """For unequal values, the log-mean-exp is above the mean and below the maximum."""
    values = np.array([0.2, 0.5, 0.8, 0.75])

    assert np.mean(values) < log_mean_exp(values, temperature=0.05) < np.max(values)


def test_side_score_is_the_log_mean_exp_of_the_ear_similarities() -> None:
    """The side score uses the configured temperature over the candidate's same-side reference ears."""
    config = EarComparatorConfig(temperature=0.2)
    comparator = EarComparator(config)
    query = _ear(_make_profile([(300, 0.05), (500, 0.02)]))
    references = [
        _ear(_make_profile([(300, 0.05), (500, 0.02)])),
        _ear(_make_profile([(310, 0.04)])),
        _ear(np.zeros(720)),
    ]
    ear_similarities = np.array([similarity.score for similarity in comparator.compare_ears(query, references)])

    assert comparator.score_side(query, references) == pytest.approx(log_mean_exp(ear_similarities, temperature=0.2))


def test_positive_alignment_ties_preserve_configured_search_order() -> None:
    """Equivalent positive alignments keep the first configured stretch."""
    config = EarComparatorConfig(
        resampled_bins=9,
        stretches=(1.2, 1.1),
        max_shift_fraction=0,
        gradient_weight=0.0,
        stretch_penalty_scale=math.inf,
    )
    similarity = _compare_one(_ear(np.ones(9)), _ear(np.ones(9)), config)
    assert (similarity.score, similarity.stretch, similarity.shift_bins) == (1.0, 1.2, 0)
