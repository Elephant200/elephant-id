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
from elephant_id.evaluation import (
    BenchmarkValidationError,
    EvaluationError,
    EvaluationStage,
    evaluate,
    load_benchmark,
)
from elephant_id.matching import CandidateKey, CandidateScores, MatchingError


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
                writer.writerow(
                    (
                        photo_id,
                        sighting_id,
                        "2020-01-01",
                        name,
                        f"{name}/{sighting_id}-{side}.jpg",
                    )
                )
    return Dataset(root, metadata)


def _write_manifest(path: Path, rows: tuple[tuple[object, ...], ...]) -> None:
    """Write a benchmark manifest."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("known_elephant_name", "sighting_id", "left_photo_id", "right_photo_id")
        )
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


def test_load_benchmark_returns_only_manifest_ids(tmp_path: Path) -> None:
    """Loading does not resolve or retain Dataset objects or paths."""
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, _valid_rows())

    benchmark = load_benchmark(manifest)

    assert benchmark.sightings["Ada"][_uuid(101)] == (_uuid(1), _uuid(2))


def test_load_benchmark_collects_manifest_errors(tmp_path: Path) -> None:
    """Malformed cells and duplicate declarations are reported together."""
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        (
            (
                "Ada",
                _uuid(101),
                "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
                "not-a-uuid",
                "extra",
            ),
            ("", _uuid(101), "", _uuid(2)),
        ),
    )

    with pytest.raises(BenchmarkValidationError) as caught:
        load_benchmark(manifest)

    assert len(caught.value.errors) == 6
    assert any("expected 4 values" in error for error in caught.value.errors)
    assert any("canonical UUIDv4" in error for error in caught.value.errors)
    assert any("duplicate sighting_id" in error for error in caught.value.errors)
    assert any("missing known_elephant_name" in error for error in caught.value.errors)


@dataclass
class RecordingMatcher:
    """Record folds and return complete deterministic scores."""

    calls: list[
        tuple[SightingEarPair, Mapping[CandidateKey, tuple[SightingEarPair, ...]]]
    ] = field(default_factory=list)

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Record one catalog and score every candidate."""
        self.calls.append((query, catalog))
        return {
            key: 1.0 if query.sighting_id < evidence[0].sighting_id else 0.0
            for key, evidence in catalog.items()
        }


def test_evaluate_preflights_every_dataset_reference(tmp_path: Path) -> None:
    """All Dataset disagreements are found before the matcher is called."""
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        (
            ("Ada", _uuid(101), _uuid(999), _uuid(5)),
            ("Wrong", _uuid(999), _uuid(1), _uuid(2)),
        ),
    )
    matcher = RecordingMatcher()

    with pytest.raises(BenchmarkValidationError) as caught:
        evaluate(load_benchmark(manifest), _dataset(tmp_path), matcher)

    assert len(caught.value.errors) == 5
    assert matcher.calls == []


def test_evaluate_uses_neutral_leakage_free_folds(tmp_path: Path) -> None:
    """Only eligible queries run, while all non-query evidence remains cataloged."""
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, _valid_rows())
    matcher = RecordingMatcher()

    result = evaluate(load_benchmark(manifest), _dataset(tmp_path), matcher)

    assert set(result.scores) == {"Ada", "Bea"}
    assert set(result.scores["Ada"][_uuid(101)]) == {"Ada", "Bea", "Cia"}
    assert [query.sighting_id for query, _ in matcher.calls] == [
        _uuid(101),
        _uuid(102),
        _uuid(103),
        _uuid(104),
    ]
    issued_keys = tuple(matcher.calls[0][1])
    assert all(tuple(catalog) == issued_keys for _, catalog in matcher.calls)
    assert all(type(key) is UUID and key.version == 4 for key in issued_keys)
    for query, catalog in matcher.calls:
        pairs = tuple(pair for evidence in catalog.values() for pair in evidence)
        assert all(pair.sighting_id != query.sighting_id for pair in pairs)
        assert _uuid(105) in {pair.sighting_id for pair in pairs}
    assert not hasattr(result, "queries")
    assert not hasattr(result, "ineligible_queries")


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
            key
            for key, evidence in catalog.items()
            if evidence[0].sighting_id in self.sightings[query.sighting_id]
        )
        scores = dict.fromkeys(catalog, 0.4)
        scores[target] = 0.5
        distractors = [key for key in catalog if key != target]
        for key in distractors[: self.ranks[query.sighting_id] - 1]:
            scores[key] = 0.6
        return scores


def test_result_derives_query_metrics_and_elephant_cluster_intervals(
    tmp_path: Path,
) -> None:
    """Point metrics weight queries; bootstrap resampling keeps elephant clusters."""
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, _valid_rows())

    result = evaluate(load_benchmark(manifest), _dataset(tmp_path), RankedMatcher())

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
    assert result.intervals == result.intervals


@dataclass(frozen=True)
class InvalidMatcher:
    """Return a selected invalid matcher result."""

    result: object

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Return the invalid value without adjustment."""
        return self.result  # type: ignore[return-value]


@dataclass(frozen=True)
class InvalidScoreMatcher:
    """Return one invalid score for every valid candidate key."""

    score: object

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Return a complete mapping containing the invalid score."""
        return dict.fromkeys(catalog, self.score)  # type: ignore[arg-type,return-value]


@pytest.mark.parametrize("result", [None, {}, {CandidateKey(_uuid(999)): math.nan}])
def test_evaluate_structures_matcher_result_errors(
    tmp_path: Path,
    result: object,
) -> None:
    """Malformed matcher output fails at the result-validation stage."""
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, _valid_rows())

    with pytest.raises(EvaluationError) as caught:
        evaluate(load_benchmark(manifest), _dataset(tmp_path), InvalidMatcher(result))

    assert caught.value.stage is EvaluationStage.MATCHER_RESULT_VALIDATION
    assert caught.value.query_sighting_id == _uuid(101)
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))


@pytest.mark.parametrize("score", [math.nan, math.inf, True])
def test_evaluate_requires_finite_real_scores(tmp_path: Path, score: object) -> None:
    """Complete matcher results still require usable similarity floats."""
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, _valid_rows())

    with pytest.raises(EvaluationError) as caught:
        evaluate(load_benchmark(manifest), _dataset(tmp_path), InvalidScoreMatcher(score))

    assert caught.value.stage is EvaluationStage.MATCHER_RESULT_VALIDATION


class FailingMatcher:
    """Raise a structured analysis failure for catalog evidence."""

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Fail on the first catalog pair."""
        pair = next(iter(catalog.values()))[0]
        raise MatchingError(
            photo=pair.left_photo,
            side="left",
            stage="tear-profile extraction",
            message="controlled failure",
        )


def test_evaluate_preserves_matching_failure_and_evidence_context(
    tmp_path: Path,
) -> None:
    """Matcher failures retain their cause and recoverable catalog location."""
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, _valid_rows())

    with pytest.raises(EvaluationError) as caught:
        evaluate(load_benchmark(manifest), _dataset(tmp_path), FailingMatcher())

    error = caught.value
    assert error.stage is EvaluationStage.CATALOG_MATCHING
    assert error.query_sighting_id == _uuid(101)
    assert error.role == "catalog"
    assert error.photo_id == _uuid(3)
    assert error.side == "left"
    assert isinstance(error.__cause__, MatchingError)


def test_evaluate_rejects_a_benchmark_without_eligible_queries(tmp_path: Path) -> None:
    """Metrics require at least one elephant with held-out catalog evidence."""
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, ((_valid_rows()[0]), (_valid_rows()[-1])))

    with pytest.raises(BenchmarkValidationError, match="no eligible queries"):
        evaluate(load_benchmark(manifest), _dataset(tmp_path), RecordingMatcher())
