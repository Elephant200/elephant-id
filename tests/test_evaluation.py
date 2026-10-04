"""Tests for the retrieval-evaluation package interface."""

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pytest

from elephant_id.dataset import Dataset
from elephant_id.domain import SightingEarPair
from elephant_id.evaluation import Benchmark, evaluate, load_benchmark
from elephant_id.matching import CandidateKey, CandidateScores
from elephant_id.matching.protocol import MatchingError


def _uuid(value: int) -> UUID:
    """Return a deterministic UUIDv4-shaped value."""
    return UUID(f"00000000-0000-4000-8000-{value:012x}")


def _dataset(tmp_path: Path) -> Dataset:
    """Build a small identity-aware Dataset."""
    root = tmp_path / "coded"
    root.mkdir()
    metadata = tmp_path / "images.csv"
    names = ("Ada", "Ada", "Bea", "Bea", "Cia")
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("photo_id", "sighting_id", "date", "name", "image_path"))
        for sighting_offset, name in enumerate(names, start=1):
            sighting_id = _uuid(100 + sighting_offset)
            for side_offset, side in enumerate(("left", "right")):
                photo_id = _uuid(2 * sighting_offset - 1 + side_offset)
                writer.writerow((photo_id, sighting_id, "2020-01-01", name, f"{name}/{sighting_id}-{side}.jpg"))
    return Dataset(root, metadata)


def _write_benchmark(path: Path, rows: tuple[tuple[object, ...], ...]) -> None:
    """Write a benchmark file."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("known_elephant_name", "sighting_id", "left_photo_id", "right_photo_id"))
        writer.writerows(rows)


def _valid_rows() -> tuple[tuple[object, ...], ...]:
    """Return two eligible elephants and one ineligible distractor."""
    return (
        ("Ada", _uuid(101), _uuid(1), _uuid(2)),
        ("Ada", _uuid(102), _uuid(3), _uuid(4)),
        ("Bea", _uuid(103), _uuid(5), _uuid(6)),
        ("Bea", _uuid(104), _uuid(7), _uuid(8)),
        ("Cia", _uuid(105), _uuid(9), _uuid(10)),
    )


def _benchmark(tmp_path: Path, rows: tuple[tuple[object, ...], ...] = _valid_rows()) -> Benchmark:
    """Write and load a benchmark against the small Dataset."""
    path = tmp_path / "benchmark.csv"
    _write_benchmark(path, rows)
    return load_benchmark(path, _dataset(tmp_path))


def test_load_benchmark_groups_resolved_pairs_by_elephant(tmp_path: Path) -> None:
    """Each row becomes a sighting ear pair under its known elephant."""
    benchmark = _benchmark(tmp_path)

    assert list(benchmark) == ["Ada", "Bea", "Cia"]
    first = benchmark["Ada"][0]
    assert (first.sighting_id, first.left_photo.photo_id, first.right_photo.photo_id) == (
        _uuid(101),
        _uuid(1),
        _uuid(2),
    )


@pytest.mark.parametrize(
    ("row", "error", "message"),
    [
        (("Bea", _uuid(101), _uuid(1), _uuid(2)), ValueError, "belongs to 'Ada'"),
        (("Ada", _uuid(101), _uuid(999), _uuid(2)), KeyError, "Unknown photo_id"),
        (("Ada", _uuid(101), _uuid(5), _uuid(2)), ValueError, "does not belong to Sighting"),
        (("Ada", _uuid(999), _uuid(1), _uuid(2)), KeyError, "Unknown sighting_id"),
    ],
)
def test_load_benchmark_rejects_rows_that_disagree_with_the_dataset(
    tmp_path: Path, row: tuple[object, ...], error: type[Exception], message: str
) -> None:
    """A bad row fails with its row number."""
    with pytest.raises(error, match=message) as caught:
        _benchmark(tmp_path, (row,))

    assert caught.value.__notes__ == [f"{tmp_path / 'benchmark.csv'}, row 2"]


def test_load_benchmark_rejects_duplicate_sightings(tmp_path: Path) -> None:
    """A sighting appears at most once."""
    with pytest.raises(ValueError, match="Duplicate sighting_id"):
        _benchmark(tmp_path, (_valid_rows()[0], _valid_rows()[0]))


@dataclass
class RecordingMatcher:
    """Record queries and return complete deterministic scores."""

    calls: list[tuple[SightingEarPair, Mapping[CandidateKey, tuple[SightingEarPair, ...]]]] = field(
        default_factory=list
    )

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Record one catalog and score every candidate."""
        self.calls.append((query, catalog))
        return {key: 1.0 if query.sighting_id < evidence[0].sighting_id else 0.0 for key, evidence in catalog.items()}


def test_evaluate_uses_neutral_leave_one_sighting_out_queries(tmp_path: Path) -> None:
    """Only eligible queries run, while all non-query evidence stays in the catalog."""
    matcher = RecordingMatcher()

    result = evaluate(_benchmark(tmp_path), lambda: matcher)

    assert set(result.scores) == {"Ada", "Bea"}
    assert set(result.scores["Ada"][_uuid(101)]) == {"Ada", "Bea", "Cia"}
    assert [query.sighting_id for query, _ in matcher.calls] == [_uuid(101), _uuid(102), _uuid(103), _uuid(104)]
    issued_keys = tuple(matcher.calls[0][1])
    assert all(tuple(catalog) == issued_keys for _, catalog in matcher.calls)
    assert all(type(key) is UUID and key.version == 4 for key in issued_keys)
    for query, catalog in matcher.calls:
        pairs = tuple(pair for evidence in catalog.values() for pair in evidence)
        assert all(pair.sighting_id != query.sighting_id for pair in pairs)
        assert _uuid(105) in {pair.sighting_id for pair in pairs}


@dataclass(frozen=True)
class RankedMatcher:
    """Give four eligible queries the ranks 1, 3, 2, and 2."""

    ranks: ClassVar = {_uuid(101): 1, _uuid(102): 3, _uuid(103): 2, _uuid(104): 2}
    sightings: ClassVar = {
        _uuid(101): {_uuid(101), _uuid(102)},
        _uuid(102): {_uuid(101), _uuid(102)},
        _uuid(103): {_uuid(103), _uuid(104)},
        _uuid(104): {_uuid(103), _uuid(104)},
    }

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Place the true candidate at the configured competition rank."""
        target = next(
            key for key, evidence in catalog.items() if evidence[0].sighting_id in self.sightings[query.sighting_id]
        )
        scores = dict.fromkeys(catalog, 0.4)
        scores[target] = 0.5
        distractors = [key for key in catalog if key != target]
        for key in distractors[: self.ranks[query.sighting_id] - 1]:
            scores[key] = 0.6
        return scores


def test_result_has_query_weighted_metrics_and_elephant_bootstrap_intervals(tmp_path: Path) -> None:
    """Point metrics weight queries; bootstrap resampling keeps all queries of each elephant."""
    result = evaluate(_benchmark(tmp_path), RankedMatcher)

    assert result.metrics == {
        "eligible_queries": 4,
        "top_1": pytest.approx(0.25),
        "top_3": pytest.approx(1.0),
        "top_5": pytest.approx(1.0),
        "top_10": pytest.approx(1.0),
        "top_15": pytest.approx(1.0),
        "mrr": pytest.approx(7 / 12),
        "median_rank": pytest.approx(2.0),
    }
    assert result.intervals["top_1"] == pytest.approx((0.0, 0.5))
    assert result.intervals["mrr"] == pytest.approx((0.5, 2 / 3))
    assert result.intervals["median_rank"] == pytest.approx((2.0, 2.0))


def test_median_rank_averages_the_two_middle_ranks(tmp_path: Path) -> None:
    """An even query count takes the mean of the two middle ranks."""

    @dataclass(frozen=True)
    class SplitMatcher(RankedMatcher):
        ranks: ClassVar = {_uuid(101): 1, _uuid(102): 1, _uuid(103): 2, _uuid(104): 3}

    assert evaluate(_benchmark(tmp_path), SplitMatcher).metrics["median_rank"] == pytest.approx(1.5)


@dataclass(frozen=True)
class FixedMatcher:
    """Return a fixed score for each candidate, plus any extra scores."""

    score: float
    extra: Mapping[CandidateKey, float] = field(default_factory=dict)

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Score every candidate with the fixed score."""
        return dict.fromkeys(catalog, self.score) | dict(self.extra)


@pytest.mark.parametrize(
    "matcher",
    [FixedMatcher(math.nan), FixedMatcher(math.inf), FixedMatcher(0.5, {CandidateKey(_uuid(999)): 0.5})],
)
def test_evaluate_rejects_invalid_matcher_scores(tmp_path: Path, matcher: FixedMatcher) -> None:
    """Scores must be finite and cover exactly the candidate keys."""
    with pytest.raises(ValueError, match="one finite score") as caught:
        evaluate(_benchmark(tmp_path), lambda: matcher)

    assert caught.value.__notes__ == [f"Query sighting {_uuid(101)}"]


class FailingMatcher:
    """Raise an analysis failure for catalog evidence."""

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Fail on the first catalog pair."""
        pair = next(iter(catalog.values()))[0]
        raise MatchingError(photo=pair.left_photo, side="left", stage="tear-profile extraction", message="failure")


def test_evaluate_reraises_matcher_failures_with_the_query(tmp_path: Path) -> None:
    """A matcher failure keeps its photo and side and gains the query sighting ID."""
    with pytest.raises(MatchingError) as caught:
        evaluate(_benchmark(tmp_path), FailingMatcher)

    assert caught.value.photo_id == _uuid(3)
    assert caught.value.side == "left"
    assert caught.value.__notes__ == [f"Query sighting {_uuid(101)}"]


def test_evaluate_rejects_a_benchmark_without_eligible_queries(tmp_path: Path) -> None:
    """Metrics need at least one elephant with another sighting in the catalog."""
    with pytest.raises(ValueError, match="no eligible queries"):
        evaluate(_benchmark(tmp_path, (_valid_rows()[0], _valid_rows()[-1])), RecordingMatcher)


def test_worker_processes_match_a_single_process(tmp_path: Path) -> None:
    """Worker processes give the same scores, metrics, and intervals as one process."""
    benchmark = _benchmark(tmp_path)

    assert evaluate(benchmark, RankedMatcher, workers=2) == evaluate(benchmark, RankedMatcher)


def test_worker_failures_keep_their_type_and_query(tmp_path: Path) -> None:
    """A matcher failure in a worker process reaches the caller unchanged."""
    with pytest.raises(MatchingError) as caught:
        evaluate(_benchmark(tmp_path), FailingMatcher, workers=2)

    assert caught.value.photo_id == _uuid(3)
    assert caught.value.__notes__ == [f"Query sighting {_uuid(101)}"]
