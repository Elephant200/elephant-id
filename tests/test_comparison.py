"""Tests for paired actual-catalog retrieval comparisons."""

from uuid import UUID

import pytest

from elephant_id.evaluation import EvaluationResult
from elephant_id.evaluation.comparison import paired_delta, top_hits


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


def test_paired_delta_is_zero_against_itself() -> None:
    """Paired delta is zero against itself."""
    scores = _scores({"a": [{"a": 1.0, "b": 0.1}], "b": [{"b": 0.2, "a": 0.9}]})
    delta, (low, high) = paired_delta(scores, scores, cutoff=1)
    assert delta == pytest.approx(0.0)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0)


def test_paired_delta_reports_the_sign_of_a_real_improvement() -> None:
    """Paired delta reports the sign of a real improvement."""
    worse = _scores({"a": [{"a": 0.1, "b": 0.9}], "b": [{"b": 0.1, "a": 0.9}]})
    better = _scores({"a": [{"a": 0.9, "b": 0.1}], "b": [{"b": 0.9, "a": 0.1}]})
    delta, _ = paired_delta(better, worse, cutoff=1)
    assert delta == pytest.approx(1.0)


def test_paired_delta_joins_keys_independent_of_mapping_order() -> None:
    """Reordered repeated observations retain exactly paired intervals."""
    scores = _scores(
        {
            "a": [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}],
            "b": [{"b": 1.0, "a": 0.0}],
        }
    )
    reordered = {
        name: dict(reversed(list(queries.items())))
        for name, queries in reversed(list(scores.items()))
    }
    assert paired_delta(scores, reordered) == (0.0, (0.0, 0.0))


def test_paired_delta_rejects_different_query_sets() -> None:
    """A missing observation cannot silently change the pairing."""
    scores = _scores({"a": [{"a": 1.0, "b": 0.0}]})
    reference = {"a": {_uuid(99): {"a": 1.0, "b": 0.0}}}
    with pytest.raises(ValueError, match="query sightings"):
        paired_delta(scores, reference)


def test_paired_delta_rejects_different_candidate_sets() -> None:
    """Systems must be measured against the same candidate catalogs."""
    scores = _scores({"a": [{"a": 1.0, "b": 0.0}]})
    reference = _scores({"a": [{"a": 1.0, "c": 0.0}]})
    with pytest.raises(ValueError, match="identical candidates"):
        paired_delta(scores, reference)


def test_nonzero_paired_interval_is_independent_of_mapping_order() -> None:
    """Canonical elephant ordering keeps seeded uncertainty reproducible."""
    scores = _scores(
        {
            "a": [{"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}],
            "b": [{"b": 1.0, "a": 0.0}],
        }
    )
    reference = _scores(
        {
            "a": [{"a": 0.0, "b": 1.0}, {"a": 0.0, "b": 1.0}],
            "b": [{"b": 0.0, "a": 1.0}],
        }
    )
    expected = paired_delta(scores, reference, resamples=1000)
    reordered = {
        name: dict(reversed(list(queries.items())))
        for name, queries in reversed(list(scores.items()))
    }
    assert paired_delta(reordered, reference, resamples=1000) == expected


def test_actual_catalog_hits_share_evaluator_ranks_and_tie_convention() -> None:
    """Larger catalogs retain every distractor; equal target scores share rank."""
    scores = _scores(
        {
            "a": [
                {"a": 0.5, "b": 0.9, "c": 0.1, "d": 0.5},
                {"a": 0.5, "b": 0.5, "c": 0.1, "d": 0.2},
            ]
        }
    )
    assert list(EvaluationResult(scores).ranks) == [2, 1]
    assert list(top_hits(scores)) == [0.0, 1.0]
    assert top_hits(scores).mean() == EvaluationResult(scores).metrics["top_1"]
    assert list(top_hits(scores, 2)) == [1.0, 1.0]
