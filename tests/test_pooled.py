"""Tests for pool-size-matched retrieval metrics."""

from uuid import UUID

import pytest

from elephant_id.evaluation.pooled import (
    distractor_counts,
    paired_delta,
    pool_hits,
    pool_metrics,
)


def _uuid(value: int) -> UUID:
    """Create a deterministic synthetic query identifier."""
    return UUID(int=value)


def _scores(
    rows: dict[str, list[dict[str, float]]],
) -> dict[str, dict[UUID, dict[str, float]]]:
    """Build a scores mapping from one list of candidate rows per name."""
    return {
        name: {_uuid(index): row for index, row in enumerate(queries)}
        for name, queries in rows.items()
    }


def test_distractor_counts_exclude_the_target() -> None:
    """Distractor counts exclude the target."""
    scores = _scores({"a": [{"a": 0.9, "b": 0.5, "c": 0.1}]})
    beaten_by, available = distractor_counts(scores)
    assert list(beaten_by) == [0]
    assert list(available) == [2]


def test_a_target_beaten_by_one_candidate_is_counted() -> None:
    """A target beaten by one candidate is counted."""
    scores = _scores({"a": [{"a": 0.5, "b": 0.9, "c": 0.1}]})
    beaten_by, _ = distractor_counts(scores)
    assert list(beaten_by) == [1]


def test_an_unbeaten_target_always_takes_the_top_rank() -> None:
    """An unbeaten target always takes the top rank."""
    scores = _scores({"a": [{"a": 1.0, "b": 0.5, "c": 0.4}]})
    assert pool_hits(scores, cutoff=1, pool_size=3) == pytest.approx([1.0])


def test_a_pool_smaller_than_the_candidates_dilutes_the_distractors() -> None:
    """One beating distractor among three cannot always be drawn."""
    scores = _scores({"a": [{"a": 0.5, "b": 0.9, "c": 0.1, "d": 0.2}]})
    both_drawn = pool_hits(scores, cutoff=1, pool_size=4)
    one_drawn = pool_hits(scores, cutoff=1, pool_size=2)
    assert both_drawn == pytest.approx([0.0])
    assert one_drawn == pytest.approx([2.0 / 3.0])


def test_a_wider_cutoff_never_scores_below_a_narrower_one() -> None:
    """A wider cutoff never scores below a narrower one."""
    scores = _scores({"a": [{"a": 0.5, "b": 0.9, "c": 0.8, "d": 0.2}]})
    assert pool_hits(scores, 1, 4)[0] <= pool_hits(scores, 3, 4)[0]


def test_pool_metrics_average_over_every_query() -> None:
    """Pool metrics average over every query."""
    scores = _scores(
        {
            "a": [{"a": 1.0, "b": 0.1}, {"a": 0.0, "b": 0.9}],
        }
    )
    assert pool_metrics(scores, cutoffs=(1,), pool_size=2) == pytest.approx(
        {"top_1": 0.5}
    )


def test_paired_delta_is_zero_against_itself() -> None:
    """Paired delta is zero against itself."""
    scores = _scores({"a": [{"a": 1.0, "b": 0.1}], "b": [{"b": 0.2, "a": 0.9}]})
    delta, (low, high) = paired_delta(scores, scores, cutoff=1, pool_size=2)
    assert delta == pytest.approx(0.0)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0)


def test_paired_delta_reports_the_sign_of_a_real_improvement() -> None:
    """Paired delta reports the sign of a real improvement."""
    worse = _scores({"a": [{"a": 0.1, "b": 0.9}], "b": [{"b": 0.1, "a": 0.9}]})
    better = _scores({"a": [{"a": 0.9, "b": 0.1}], "b": [{"b": 0.9, "a": 0.1}]})
    delta, _ = paired_delta(better, worse, cutoff=1, pool_size=2)
    assert delta == pytest.approx(1.0)


def test_pool_cannot_extrapolate_unobserved_distractors() -> None:
    """An undersized catalog must not masquerade as the benchmark pool."""
    scores = _scores({'a': [{'a': 1.0, 'b': 0.0}]})
    with pytest.raises(ValueError, match='Pool size'):
        pool_hits(scores, pool_size=89)


def test_paired_delta_joins_keys_independent_of_mapping_order() -> None:
    """Reordered repeated observations retain exactly paired intervals."""
    scores = _scores({
        'a': [{'a': 1.0, 'b': 0.0}, {'a': 0.0, 'b': 1.0}],
        'b': [{'b': 1.0, 'a': 0.0}],
    })
    reordered = {name: dict(reversed(list(queries.items())))
                 for name, queries in reversed(list(scores.items()))}
    assert paired_delta(scores, reordered, pool_size=2) == (0.0, (0.0, 0.0))


def test_paired_delta_rejects_different_query_sets() -> None:
    """A missing observation cannot silently change the pairing."""
    scores = _scores({'a': [{'a': 1.0, 'b': 0.0}]})
    reference = {'a': {_uuid(99): {'a': 1.0, 'b': 0.0}}}
    with pytest.raises(ValueError, match='query sightings'):
        paired_delta(scores, reference, pool_size=2)


def test_paired_delta_rejects_different_candidate_sets() -> None:
    """Systems must be measured against the same candidate catalogs."""
    scores = _scores({'a': [{'a': 1.0, 'b': 0.0}]})
    reference = _scores({'a': [{'a': 1.0, 'c': 0.0}]})
    with pytest.raises(ValueError, match='identical candidates'):
        paired_delta(scores, reference, pool_size=2)


def test_nonzero_paired_interval_is_independent_of_mapping_order() -> None:
    """Canonical elephant ordering keeps seeded uncertainty reproducible."""
    scores = _scores({
        'a': [{'a': 1.0, 'b': 0.0}, {'a': 0.0, 'b': 1.0}],
        'b': [{'b': 1.0, 'a': 0.0}],
    })
    reference = _scores({
        'a': [{'a': 0.0, 'b': 1.0}, {'a': 0.0, 'b': 1.0}],
        'b': [{'b': 0.0, 'a': 1.0}],
    })
    expected = paired_delta(scores, reference, pool_size=2, resamples=1000)
    reordered = {name: dict(reversed(list(queries.items())))
                 for name, queries in reversed(list(scores.items()))}
    assert paired_delta(reordered, reference, pool_size=2, resamples=1000) == expected
