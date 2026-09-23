"""Numerical characterization of fixed AlphaTear stack similarity."""

import numpy as np
import pytest

from elephant_id.matching.alphaphant.similarity import TearMatcher


def make_profile(peaks: list[tuple[int, float]], bins: int = 240) -> np.ndarray:
    """Return a profile with triangular peaks at `(bin, depth)` positions."""
    profile = np.zeros(bins)
    half_width = 8
    for center, depth in peaks:
        for offset in range(-half_width, half_width + 1):
            index = center + offset
            if 0 <= index < bins:
                profile[index] = max(
                    profile[index], depth * (1 - abs(offset) / half_width)
                )
    return profile


def test_identical_stack_has_expected_two_channel_score() -> None:
    """An ordinary profile matches itself perfectly in both channels."""
    profile = make_profile([(120, 0.05)])

    scores = TearMatcher().match_many((profile,), ((profile,),))

    np.testing.assert_allclose(scores, [1.0], rtol=0, atol=1e-12)


def test_all_channels_and_scales_share_one_alignment() -> None:
    """Conflicting channel optima cannot select different transformations."""
    query = (
        np.asarray([0.3658017467639605, 0.39651648937307304, -0.12531048092590164]),
        np.asarray(
            [
                -0.1159222795104357,
                0.09419447939636129,
                0.3025783232114532,
                0.2633859408106408,
                0.45766879154693557,
            ]
        ),
    )
    candidate = (
        np.asarray(
            [
                -0.24577177841555098,
                0.09287617312558763,
                -0.029285893523125023,
                0.18153351422376843,
            ]
        ),
        np.asarray(
            [
                -0.26617144755736344,
                -0.02800164028671978,
                -0.23464010866503002,
                0.47699112398384197,
                0.1710237149109526,
                0.10614167419279162,
            ]
        ),
    )
    matcher = TearMatcher()

    score = matcher.match_many(query, (candidate,))[0]
    channels = []
    for signed, weight in ((False, 0.4), (True, 0.6)):
        query_rows = matcher._prepare_stack(query, signed=signed)
        candidate_rows = matcher._prepare_stack(candidate, signed=signed)
        channel = np.zeros(len(matcher._variant_penalties))
        for source, target in zip(query_rows, candidate_rows, strict=True):
            channel += matcher._overlap_scores(
                target[None, :], matcher._build_query_variants(source)
            )[0] / len(query_rows)
        channels.append(weight * channel)
    shared_best = np.max(sum(channels) * matcher._variant_penalties)
    separate_best = sum(
        np.max(channel * matcher._variant_penalties) for channel in channels
    )

    assert score <= shared_best + 1e-12
    assert shared_best < separate_best


def test_shared_scale_alignment_scores_below_independent_scale_average() -> None:
    """One alignment must serve every scale in a profile stack."""
    query = (make_profile([(70, 1.0)]), make_profile([(150, 1.0)]))
    candidate = (make_profile([(94, 1.0)]), make_profile([(126, 1.0)]))
    matcher = TearMatcher()

    independent = np.mean(
        [
            matcher.match_many((source,), ((target,),))[0]
            for source, target in zip(query, candidate, strict=True)
        ]
    )
    shared = matcher.match_many(query, (candidate,))[0]

    assert shared < independent


def test_signed_change_is_taken_before_depth_clipping() -> None:
    """A negative-depth valley still contributes its signed derivatives."""
    profile = np.asarray([-1.0, -0.5, -1.0])

    score = TearMatcher().match_many((profile,), ((profile,),))[0]

    # Depth is empty after clipping; the signed channel remains a perfect match.
    assert score == pytest.approx(0.60, abs=1e-12)


def test_ragged_stacks_and_empty_or_zero_candidates() -> None:
    """Ragged source lengths, empty catalogs, and zero mass have clear results."""
    query = (np.linspace(0.0, 1.0, 7), np.linspace(1.0, 0.0, 31))
    candidates = (
        (np.linspace(0.0, 1.0, 3), np.linspace(1.0, 0.0, 73)),
        (np.zeros(11), np.zeros(5)),
    )
    matcher = TearMatcher()

    scores = matcher.match_many(query, candidates)

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert scores[1] == 0.0
    assert matcher.match_many(query, ()).shape == (0,)


def test_shared_scores_match_direct_min_max_overlap() -> None:
    """Batched L1 scores match direct min/max overlap with one alignment."""
    rng = np.random.default_rng(123)
    query = tuple(rng.uniform(-0.02, 0.1, n) for n in (1, 2, 7))
    candidates = tuple(
        tuple(rng.uniform(-0.03, 0.1, n) for n in (1, 2, 7)) for _ in range(3)
    )

    matcher = TearMatcher()
    scores = matcher.match_many(query, candidates)
    expected = []
    for candidate in candidates:
        combined = np.zeros(len(matcher._variant_penalties))
        for signed, weight in ((False, 0.4), (True, 0.6)):
            query_rows = matcher._prepare_stack(query, signed=signed)
            candidate_rows = matcher._prepare_stack(candidate, signed=signed)
            for source, target in zip(query_rows, candidate_rows, strict=True):
                variants = matcher._build_query_variants(source)
                numerator = np.minimum(variants, target).sum(axis=1)
                denominator = np.maximum(variants, target).sum(axis=1)
                overlap = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator),
                    where=denominator > 0,
                )
                combined += weight * overlap / len(query_rows)
        expected.append(np.max(combined * matcher._variant_penalties))

    np.testing.assert_allclose(
        scores,
        expected,
        rtol=0,
        atol=1e-12,
    )


def test_rejects_empty_or_mismatched_stacks() -> None:
    """A candidate stack must contain one profile for every query scale."""
    profile = np.ones(7)
    matcher = TearMatcher()

    with pytest.raises(ValueError, match="nonempty"):
        matcher.match_many((), ((profile,),))
    with pytest.raises(ValueError, match="one-to-one"):
        matcher.match_many((profile,), ((),))


def test_chunking_preserves_scores_and_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing the bounded batch size does not alter scores or source arrays."""
    import elephant_id.matching.alphaphant.similarity as module

    rng = np.random.default_rng(12)
    query = (rng.random(720), rng.random(73))
    candidates = tuple((rng.random(720), rng.random(73)) for _ in range(15))
    query_copy = tuple(profile.copy() for profile in query)
    candidates_copy = tuple(
        tuple(profile.copy() for profile in stack) for stack in candidates
    )
    matcher = TearMatcher()
    expected = matcher.match_many(query, candidates)

    monkeypatch.setattr(module, "_OVERLAP_WORKSPACE_BYTES", 1)
    actual = matcher.match_many(query, candidates)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    for actual_profile, original_profile in zip(query, query_copy, strict=True):
        np.testing.assert_array_equal(actual_profile, original_profile)
    for actual_stack, original_stack in zip(candidates, candidates_copy, strict=True):
        for actual_profile, original_profile in zip(
            actual_stack, original_stack, strict=True
        ):
            np.testing.assert_array_equal(actual_profile, original_profile)
