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
from elephant_id.matching.alphaphant.profile import (
    EarProfile,
    SightingProfiles,
    TearProfile,
)
from elephant_id.matching.alphaphant.similarity import TearMatch, TearMatcher
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


def _ear(value: int, side: EarSide, depths: list[float]) -> EarProfile:
    """Return one analyzed ear with deterministic neutral provenance."""
    return EarProfile(
        source_photo=Photo(
            photo_id=_uuid(value),
            sighting_id=_uuid(value + 1000),
        ),
        side=side,
        source_box=BoundingBox(0, 0, 4, 4),
        tear_profile=TearProfile(np.asarray(depths)),
    )


def _analysis(value: int, left: list[float], right: list[float]) -> SightingProfiles:
    """Return one analyzed two-sided sighting."""
    return SightingProfiles(
        left=_ear(value, "left", left),
        right=_ear(value + 1, "right", right),
    )


class RecordingPreparation:
    """Return controlled analyses while recording neutral pair inputs."""

    def __init__(
        self,
        analyses: Mapping[SightingEarPair, SightingProfiles],
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

    def extract(self, ear: PreparedEar) -> TearProfile:
        """Return controlled numerical profiles for the prepared source photo."""
        return next(
            item.tear_profile
            for profiles in self._analyses.values()
            for item in (profiles.left, profiles.right)
            if item.source_photo == ear.source_photo and item.side == ear.inferred_side
        )


def _alphaphant(
    analyses: Mapping[SightingEarPair, SightingProfiles],
) -> tuple[AlphaPhant, RecordingPreparation]:
    """Return AlphaPhant with a controlled recording analyzer."""
    analyzer = RecordingPreparation(analyses)
    return (
        AlphaPhant(
            prepare_ears=analyzer.prepare,
            profile_extractors=(analyzer,),
            channel_matchers=(TearMatcher(),),
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
    assert scores[strong_key] == pytest.approx(2 / (1 + math.exp(-3 / math.sqrt(2))))
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
            query: _analysis(100, [0.0, 0.1, 0.0], [0.0, 0.0, 0.2]),
            right_match: _analysis(200, [0.0, 0.0, 0.0], [0.0, 0.0, 0.2]),
            left_match: _analysis(300, [0.0, 0.1, 0.0], [0.0, 0.0, 0.0]),
        }
    )

    scores = alphaphant.match(query, {candidate_key: (right_match, left_match)})

    assert scores[candidate_key] == pytest.approx(2 / (1 + math.exp(-2)))


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
    original = TearMatcher.match_stack_many

    def record(
        self: TearMatcher,
        profile: Sequence[np.ndarray],
        candidates: Sequence[Sequence[np.ndarray]],
    ) -> tuple[TearMatch, ...]:
        """Record each bulk call while executing the real matcher."""
        calls.append((profile, candidates))
        return original(self, profile, candidates)

    monkeypatch.setattr(TearMatcher, "match_stack_many", record)
    first_key, second_key = CandidateKey(_uuid(500)), CandidateKey(_uuid(501))
    scores = alphaphant.match(query, {first_key: (first,), second_key: (second,)})
    assert len(calls) == 6
    assert calls[0][0][0] is query_analysis.left.tear_profile.depths
    assert calls[3][0][0] is query_analysis.right.tear_profile.depths
    assert all(len(candidates) == 3 for _, candidates in calls)
    assert scores == pytest.approx({first_key: 1.0, second_key: 1.0})
    assert (
        alphaphant.match(query, {second_key: (second,), first_key: (first,)}) == scores
    )


def test_catalog_neighborhood_excludes_held_out_evidence() -> None:
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

    class ControlledTearMatcher(TearMatcher):
        """Provide known ear similarities so catalog arithmetic is observable."""

        def match_stack_many(
            self,
            query: Sequence[np.ndarray],
            candidates: Sequence[Sequence[np.ndarray]],
        ) -> tuple[TearMatch, ...]:
            """Read an identity-neutral synthetic profile index."""
            return tuple(
                TearMatch(float(values[int(query[0][0]), int(stack[0][0])]), 1.0, 0)
                for stack in candidates
            )

    analyzer = RecordingPreparation(
        {pair: _analysis(500 + i, [i], [i]) for i, pair in enumerate(pairs)}
    )
    matcher = AlphaPhant(
        prepare_ears=analyzer.prepare,
        profile_extractors=(analyzer,),
        channel_matchers=(ControlledTearMatcher(),),
    )
    first, second = CandidateKey(_uuid(900)), CandidateKey(_uuid(901))
    full = {first: (pairs[1], pairs[2]), second: (pairs[3],)}
    scores = matcher.match(pairs[0], full)
    assert scores[second] == pytest.approx(1.45)
    assert scores[second] > scores[first]

    reduced = matcher.match(pairs[0], {first: (pairs[1],), second: (pairs[3],)})
    assert reduced == pytest.approx({first: 1.7, second: 1.5})
    assert matcher.match(pairs[0], full) == scores


@pytest.mark.parametrize("magnitude", [1e308, 5e-324])
def test_channel_weight_magnitude_does_not_change_scores(magnitude: float) -> None:
    """Finite extreme weights preserve relative influence without overflow."""
    query, evidence = _pair(10), _pair(20)
    analysis = _analysis(100, [0.0, 0.2, 0.0], [0.0, 0.1, 0.0])
    analyzer = RecordingPreparation({query: analysis, evidence: analysis})
    catalog = {CandidateKey(_uuid(100)): (evidence,)}
    common = {
        "prepare_ears": analyzer.prepare,
        "profile_extractors": (analyzer,),
        "channel_matchers": (TearMatcher(), TearMatcher()),
    }
    reference = AlphaPhant(**common, channel_weights=(1.0, 1.0)).match(query, catalog)
    result = AlphaPhant(**common, channel_weights=(magnitude, magnitude)).match(
        query, catalog
    )
    assert next(iter(reference.values())) > 0.0
    assert result == pytest.approx(reference, abs=1e-12)


def test_profiles_share_preparation_across_scales_and_matching() -> None:
    """Inspection and repeated matching reuse the same multiscale extraction."""
    pair = _pair(10)
    prepared = RecordingPreparation(
        {pair: _analysis(100, [0.0, 0.2, 0.0], [0.0, 0.1, 0.0])}
    )

    class RecordingExtractor:
        """Record per-ear calls for one synthetic extraction scale."""

        producer_slug = "synthetic-scale"

        def __init__(self) -> None:
            """Start without extracted ears."""
            self.calls: list[PreparedEar] = []

        def extract(self, ear: PreparedEar) -> TearProfile:
            """Retain the prepared ear and return its synthetic profile."""
            self.calls.append(ear)
            return prepared.extract(ear)

    fine, coarse = RecordingExtractor(), RecordingExtractor()
    matcher = AlphaPhant(
        prepare_ears=prepared.prepare,
        profile_extractors=(fine, coarse),
        channel_matchers=(TearMatcher(),),
    )
    profiles = matcher.profiles(pair)
    assert len(profiles) == 2
    assert matcher.match(pair, {}) == {}
    assert matcher.profiles(pair) is profiles
    assert prepared.calls == [pair]
    assert len(fine.calls) == len(coarse.calls) == 2
    assert all(
        first is second for first, second in zip(fine.calls, coarse.calls, strict=True)
    )


def test_alphaphant_package_exports_matcher() -> None:
    """Algorithm imports expose AlphaPhant without widening the matching package."""
    import elephant_id.matching.alphaphant as public_alphaphant

    assert public_alphaphant.__all__ == ["AlphaPhant"]
    assert public_alphaphant.AlphaPhant is AlphaPhant
