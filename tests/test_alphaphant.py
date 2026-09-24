"""Tests for the public AlphaPhant catalog-matching seam."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

import numpy as np
import pytest

import elephant_id.matching as public_matching
from elephant_id.domain import Photo, SightingEarPair
from elephant_id.image.boxes import BoundingBox
from elephant_id.matching import (
    CandidateKey,
    CandidateScores,
    CatalogMatcher,
)
from elephant_id.matching.alphaphant import AlphaPhant
from elephant_id.matching.alphaphant.extraction import (
    AnalyzedEar,
    AnalyzedSightingEarPair,
    TearProfile,
)
from elephant_id.matching.alphaphant.matcher import _subtract_background
from elephant_id.matching.alphaphant.similarity import EarProfileSimilarity
from elephant_id.preparation import EarSide, PreparedEar


def _uuid(value: int) -> UUID:
    """Return a deterministic UUIDv4-shaped value."""
    return UUID(f"00000000-0000-4000-8000-{value:012x}")


def _pair(value: int) -> SightingEarPair:
    """Return one neutral sighting ear pair."""
    sighting_id = _uuid(value)
    return SightingEarPair(
        sighting_id=sighting_id,
        left_photo=Photo(photo_id=_uuid(value + 1), sighting_id=sighting_id),
        right_photo=Photo(photo_id=_uuid(value + 2), sighting_id=sighting_id),
    )


def _ear(value: int, side: EarSide, depths: list[float]) -> AnalyzedEar:
    """Return one analyzed ear with deterministic neutral provenance."""
    return AnalyzedEar(
        source_photo=Photo(
            photo_id=_uuid(value),
            sighting_id=_uuid(value + 1000),
        ),
        side=side,
        source_box=BoundingBox(0, 0, 4, 4),
        tear_profiles=(TearProfile(np.asarray(depths)),),
    )


def _analysis(
    value: int, left: list[float], right: list[float]
) -> AnalyzedSightingEarPair:
    """Return one analyzed two-sided sighting."""
    return AnalyzedSightingEarPair(
        sighting_id=_uuid(value + 2000),
        left=_ear(value, "left", left),
        right=_ear(value + 1, "right", right),
    )


class RecordingPreparation:
    """Return controlled analyses while recording neutral pair inputs."""

    def __init__(
        self,
        analyses: Mapping[SightingEarPair, AnalyzedSightingEarPair],
    ) -> None:
        """Initialize the analyzer with one result per neutral pair."""
        self._analyses = analyses
        self.calls: list[SightingEarPair] = []

    def prepare(self, pair: SightingEarPair) -> tuple[PreparedEar, PreparedEar]:
        """Record and return the analysis for `pair`."""
        self.calls.append(pair)
        profiles = self._analyses[pair]
        return tuple(
            PreparedEar(
                source_photo=ear.source_photo,
                source_box=ear.source_box,
                contour=np.asarray([[0, 0], [3, 2], [0, 3]], dtype=float),
                original_landmarks=((0, 0), (0, 3)),
                contour_anchors=((0, 0), (0, 3)),
                inferred_side=ear.side,
                cleaned_area=4.5,
            )
            for ear in (profiles.left, profiles.right)
        )

    producer_slug = "synthetic-profiles"

    def extract(self, ear: PreparedEar) -> tuple[TearProfile, ...]:
        """Return the controlled profile bundle for the prepared source photo."""
        return next(
            item.tear_profiles
            for profiles in self._analyses.values()
            for item in (profiles.left, profiles.right)
            if item.source_photo == ear.source_photo and item.side == ear.inferred_side
        )


def _alphaphant(
    analyses: Mapping[SightingEarPair, AnalyzedSightingEarPair],
) -> tuple[AlphaPhant, RecordingPreparation]:
    """Return AlphaPhant with a controlled recording analyzer."""
    analyzer = RecordingPreparation(analyses)
    return (
        AlphaPhant(
            prepare_ears=analyzer.prepare,
            profile_extractor=analyzer,
        ),
        analyzer,
    )


@dataclass(frozen=True)
class ScriptedCatalogMatcher:
    """Return scripted candidate scores through the public matcher seam."""

    scores: CandidateScores

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Return the scripted scores."""
        return self.scores


def test_scripted_matcher_satisfies_public_catalog_matcher_seam() -> None:
    """Evaluation-style consumers need no AlphaPhant implementation details."""
    query = _pair(10)
    candidate_key = CandidateKey(_uuid(100))
    catalog = {candidate_key: (_pair(20),)}
    matcher: CatalogMatcher = ScriptedCatalogMatcher({candidate_key: 0.75})

    assert matcher.match(query, catalog) == {candidate_key: 0.75}


def test_alphaphant_analyzes_neutral_catalog_and_returns_candidate_scores() -> None:
    """AlphaPhant hides analysis and projects authoritative catalog scores."""
    query = _pair(10)
    strong_evidence = _pair(20)
    alternate_evidence = _pair(30)
    weak_evidence = _pair(40)
    query_analysis = _analysis(100, [0.0, 0.1, 0.0], [0.0, 0.2, 0.0])
    strong_analysis = _analysis(200, [0.0, 0.1, 0.0], [0.0, 0.2, 0.0])
    weak_analysis = _analysis(300, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    alphaphant, analyzer = _alphaphant(
        {
            query: query_analysis,
            strong_evidence: strong_analysis,
            alternate_evidence: weak_analysis,
            weak_evidence: weak_analysis,
        }
    )
    strong_key = CandidateKey(_uuid(500))
    weak_key = CandidateKey(_uuid(501))
    catalog = {
        weak_key: (weak_evidence,),
        strong_key: (alternate_evidence, strong_evidence),
    }
    scores = alphaphant.match(query, catalog)

    assert isinstance(scores, dict)
    assert scores.keys() == catalog.keys()
    assert scores[strong_key] == pytest.approx(
        2 / (1 + math.exp(-3 / (2 * math.sqrt(2))))
    )
    assert scores[weak_key] == pytest.approx(0.0)
    assert scores[strong_key] > scores[weak_key]
    assert all(math.isfinite(score) for score in scores.values())
    assert analyzer.calls == [
        query,
        weak_evidence,
        alternate_evidence,
        strong_evidence,
    ]


def test_alphaphant_returns_empty_scores_for_empty_catalog() -> None:
    """An empty catalog produces a complete empty score mapping."""
    query = _pair(10)
    alphaphant, analyzer = _alphaphant(
        {query: _analysis(100, [0.0, 0.1, 0.0], [0.0, 0.2, 0.0])}
    )

    assert alphaphant.match(query, {}) == {}
    assert analyzer.calls == [query]


def test_alphaphant_propagates_analysis_failure_without_partial_scores() -> None:
    """A failed catalog analysis aborts before candidate matching."""
    query = _pair(10)
    available_evidence = _pair(20)
    missing_evidence = _pair(30)
    alphaphant, _ = _alphaphant(
        {
            query: _analysis(100, [0.0, 0.1, 0.0], [0.0, 0.2, 0.0]),
            available_evidence: _analysis(
                200,
                [0.0, 0.1, 0.0],
                [0.0, 0.2, 0.0],
            ),
        }
    )

    with pytest.raises(KeyError) as error:
        alphaphant.match(
            query,
            {
                CandidateKey(_uuid(500)): (
                    available_evidence,
                    missing_evidence,
                )
            },
        )

    assert error.value.args == (missing_evidence,)


def test_alphaphant_scores_do_not_depend_on_call_history() -> None:
    """Intervening matches cannot change a repeated call's scores."""
    query = _pair(10)
    strong_evidence = _pair(20)
    weak_evidence = _pair(30)
    alphaphant, analyzer = _alphaphant(
        {
            query: _analysis(100, [0.0, 0.1, 0.0], [0.0, 0.2, 0.0]),
            strong_evidence: _analysis(
                200,
                [0.0, 0.1, 0.0],
                [0.0, 0.2, 0.0],
            ),
            weak_evidence: _analysis(
                300,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ),
        }
    )
    candidate_key = CandidateKey(_uuid(500))
    catalog = {candidate_key: (strong_evidence,)}

    first = alphaphant.match(query, catalog)
    alphaphant.match(query, {candidate_key: (weak_evidence,)})
    repeated = alphaphant.match(query, catalog)

    assert repeated == first
    assert repeated is not first
    assert analyzer.calls == [query, strong_evidence, weak_evidence]


def test_alphaphant_rejects_empty_candidate_evidence() -> None:
    """A listed candidate must contain at least one sighting ear pair."""
    query = _pair(10)
    alphaphant, _ = _alphaphant(
        {query: _analysis(100, [0.0, 0.1, 0.0], [0.0, 0.2, 0.0])}
    )
    candidate_key = CandidateKey(_uuid(500))

    with pytest.raises(
        RuntimeError,
        match=f"{candidate_key} has no catalog evidence",
    ):
        alphaphant.match(query, {candidate_key: ()})


def test_alphaphant_aggregates_each_side_before_averaging() -> None:
    """Complementary sightings support each side before the two sides are averaged."""
    query = _pair(10)
    right_match = _pair(20)
    left_match = _pair(30)
    candidate_key = CandidateKey(_uuid(500))
    alphaphant, _ = _alphaphant(
        {
            query: _analysis(100, [0.0, 0.1, 0.0], [0.0, 0.2, 0.0]),
            right_match: _analysis(200, [0.0, 0.0, 0.0], [0.0, 0.2, 0.0]),
            left_match: _analysis(300, [0.0, 0.1, 0.0], [0.0, 0.0, 0.0]),
        }
    )

    scores = alphaphant.match(query, {candidate_key: (right_match, left_match)})

    assert scores[candidate_key] == pytest.approx(2 / (1 + math.exp(-1)))


def test_alphaphant_compares_only_corresponding_sides() -> None:
    """Strong opposite-side profiles cannot support a candidate score."""
    query = _pair(10)
    cross_side_decoys = _pair(20)
    candidate_key = CandidateKey(_uuid(500))
    left_depths = np.zeros(720)
    left_depths[100] = 0.1
    right_depths = np.zeros(720)
    right_depths[600] = 0.2
    alphaphant, _ = _alphaphant(
        {
            query: _analysis(100, left_depths.tolist(), right_depths.tolist()),
            cross_side_decoys: _analysis(
                200,
                right_depths.tolist(),
                left_depths.tolist(),
            ),
        }
    )

    scores = alphaphant.match(query, {candidate_key: (cross_side_decoys,)})

    assert scores[candidate_key] == pytest.approx(0.0)


def test_matching_package_exports_only_public_catalog_interface() -> None:
    """Matching internals remain importable only from their implementation modules."""
    assert public_matching.__all__ == [
        "CandidateKey",
        "CandidateScores",
        "CatalogMatcher",
        "MatchingError",
    ]


def test_alphaphant_batches_all_candidates_by_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each directional row is batched once and reused after catalog reordering."""
    query, first, second = _pair(10), _pair(20), _pair(30)
    query_analysis = _analysis(100, [0, 0.1, 0], [0, 0.2, 0])
    first_analysis = _analysis(200, [0, 0.1, 0], [0, 0, 0])
    second_analysis = _analysis(300, [0, 0, 0], [0, 0.2, 0])
    alphaphant, _ = _alphaphant(
        {query: query_analysis, first: first_analysis, second: second_analysis}
    )
    calls: list[tuple[Sequence[np.ndarray], Sequence[Sequence[np.ndarray]]]] = []
    original = EarProfileSimilarity.compare_many

    def record(
        self: EarProfileSimilarity,
        profile: Sequence[np.ndarray],
        candidates: Sequence[Sequence[np.ndarray]],
    ) -> np.ndarray:
        """Record each bulk call while executing the real matcher."""
        calls.append((profile, candidates))
        return original(self, profile, candidates)

    monkeypatch.setattr(EarProfileSimilarity, "compare_many", record)
    first_key, second_key = CandidateKey(_uuid(500)), CandidateKey(_uuid(501))
    scores = alphaphant.match(query, {first_key: (first,), second_key: (second,)})
    assert len(calls) == 6
    assert calls[0][0][0] is query_analysis.left.tear_profiles[0].depths
    assert calls[3][0][0] is query_analysis.right.tear_profiles[0].depths
    assert all(len(candidates) == 2 for _, candidates in calls)
    assert all(
        (side, alphaphant._slots[source], alphaphant._slots[query])
        not in alphaphant._similarities
        for source in (query, first, second)
        for side in ("left", "right")
    )
    assert scores == pytest.approx({first_key: 1.0, second_key: 1.0})
    assert (
        alphaphant.match(query, {second_key: (second,), first_key: (first,)}) == scores
    )
    assert len(calls) == 6


def test_new_query_compares_only_to_warm_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new query needs one forward batch per side without reverse comparisons."""
    pairs = tuple(_pair(value) for value in (10, 20, 30, 40))
    analyses = {
        pair: _analysis(100 + i, [0, i + 1, 0], [0, i + 2, 0])
        for i, pair in enumerate(pairs)
    }
    matcher, _ = _alphaphant(analyses)
    catalog = {CandidateKey(_uuid(500)): pairs[2:]}
    matcher.match(pairs[0], catalog)
    counts = []
    original = EarProfileSimilarity.compare_many

    def record(
        self: EarProfileSimilarity,
        query: Sequence[np.ndarray],
        candidates: Sequence[Sequence[np.ndarray]],
    ) -> np.ndarray:
        """Count actual comparisons while retaining real similarity arithmetic."""
        counts.append(len(candidates))
        return original(self, query, candidates)

    monkeypatch.setattr(EarProfileSimilarity, "compare_many", record)
    actual = matcher.match(pairs[1], catalog)
    assert counts == [2, 2]
    fresh, _ = _alphaphant(analyses)
    assert actual == fresh.match(pairs[1], catalog)


def test_query_in_catalog_retains_explicit_self_evidence() -> None:
    """Rectangular comparison preserves caller-supplied self evidence."""
    query = _pair(10)
    matcher, _ = _alphaphant(
        {query: _analysis(100, [0, 1, 0], [0, 2, 0])}
    )
    key = CandidateKey(_uuid(500))

    assert matcher.match(query, {key: (query,)})[key] == pytest.approx(2.0)


def test_catalog_neighborhood_excludes_held_out_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached similarities cannot leak a removed sighting into a later neighborhood."""
    pairs = tuple(_pair(100 + 10 * i) for i in range(4))
    values = np.asarray(
        [
            [1.0, 0.9, 0.6, 0.8],
            [0.9, 1.0, 0.9, 0.1],
            [0.6, 0.9, 1.0, 0.2],
            [0.8, 0.1, 0.2, 1.0],
        ]
    )

    def controlled_similarity(
        self: EarProfileSimilarity,
        query: Sequence[np.ndarray],
        candidates: Sequence[Sequence[np.ndarray]],
    ) -> np.ndarray:
        """Return scripted similarities indexed by neutral synthetic profiles."""
        return np.asarray(
            [values[int(query[0][0]), int(stack[0][0])] for stack in candidates]
        )

    monkeypatch.setattr(EarProfileSimilarity, "compare_many", controlled_similarity)

    analyzer = RecordingPreparation(
        {pair: _analysis(500 + i, [i], [i]) for i, pair in enumerate(pairs)}
    )
    matcher = AlphaPhant(
        prepare_ears=analyzer.prepare,
        profile_extractor=analyzer,
    )
    first, second = CandidateKey(_uuid(900)), CandidateKey(_uuid(901))
    full = {first: (pairs[1], pairs[2]), second: (pairs[3],)}
    scores = matcher.match(pairs[0], full)
    assert scores[second] == pytest.approx(1.45)
    assert scores[second] > scores[first]

    reduced = matcher.match(pairs[0], {first: (pairs[1],), second: (pairs[3],)})
    assert reduced == pytest.approx({first: 1.7, second: 1.5})
    assert matcher.match(pairs[0], full) == scores


def test_one_extraction_call_returns_ordered_multiscale_profiles() -> None:
    """One prepared ear extraction supplies all scales in stable order."""
    pair = _pair(10)
    analysis = _analysis(100, [0.0, 0.2, 0.0], [0.0, 0.1, 0.0])
    prepared = RecordingPreparation({pair: analysis})

    class RecordingExtractor:
        """Record per-ear calls for one synthetic multiscale extraction."""

        producer_slug = "synthetic-multiscale"

        def __init__(self) -> None:
            """Start without extracted ears."""
            self.calls: list[PreparedEar] = []

        def extract(self, ear: PreparedEar) -> tuple[TearProfile, ...]:
            """Retain the prepared ear and return two ordered scales."""
            self.calls.append(ear)
            base = prepared.extract(ear)[0]
            return (base, TearProfile(base.depths * 0.5))

    extractor = RecordingExtractor()
    matcher = AlphaPhant(
        prepare_ears=prepared.prepare,
        profile_extractor=extractor,
    )
    result = matcher.analyze(pair)
    assert matcher.match(pair, {}) == {}
    assert matcher.analyze(pair) is result
    assert prepared.calls == [pair]
    assert len(extractor.calls) == 2
    assert len(result.left.tear_profiles) == 2
    assert result.sighting_id == pair.sighting_id
    assert result.left.source_photo == analysis.left.source_photo
    assert result.right.source_photo == analysis.right.source_photo
    with pytest.raises(AttributeError):
        result.left.side = "right"  # type: ignore[misc]
    np.testing.assert_array_equal(
        result.left.tear_profiles[0].depths,
        [0.0, 0.2, 0.0],
    )
    np.testing.assert_array_equal(
        result.left.tear_profiles[1].depths,
        [0.0, 0.1, 0.0],
    )


def test_alphaphant_package_exports_matcher() -> None:
    """Algorithm imports expose AlphaPhant without widening the matching package."""
    import elephant_id.matching.alphaphant as public_alphaphant

    assert public_alphaphant.__all__ == ["AlphaPhant"]
    assert public_alphaphant.AlphaPhant is AlphaPhant


def test_catalog_correction_uses_five_other_ears() -> None:
    """Limit each outgoing background mean to five ears and exclude itself."""
    internal = np.full((12, 12), 0.1)
    internal[:, :5] = 0.9
    np.fill_diagonal(internal, 1.0)

    corrected = _subtract_background(np.ones(12), internal)

    np.testing.assert_allclose(corrected, [1.26] * 5 + [1.10] * 7)
