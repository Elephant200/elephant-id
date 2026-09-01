"""Concrete AlphaPhant catalog matcher."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache

from elephant_id.analysis import EarAnalysis, SightingAnalysis, SightingAnalyzer
from elephant_id.domain import SightingEarPair
from elephant_id.matching.protocol import CandidateKey, CandidateScores
from elephant_id.matching.tear_matcher import TearMatch, TearMatcher


@dataclass(frozen=True, slots=True)
class _SideMatch:
    """One winning catalog ear and its tear-profile match."""

    catalog_evidence: EarAnalysis
    tear_match: TearMatch

    @property
    def score(self) -> float:
        """Return the score derived from the tear-profile match."""
        return self.tear_match.score


@dataclass(frozen=True, slots=True)
class _CandidateMatch:
    """The independently winning left- and right-ear evidence for a candidate."""

    left: _SideMatch
    right: _SideMatch

    @property
    def score(self) -> float:
        """Return the arithmetic mean of the two winning side scores."""
        return (self.left.score + self.right.score) / 2.0


class AlphaPhant:
    """Analyze sighting ear pairs and score every catalog candidate."""

    def __init__(
        self,
        *,
        analyzer: SightingAnalyzer,
        tear_matcher: TearMatcher,
    ) -> None:
        """Initialize AlphaPhant with sighting analysis and tear matching."""
        self._analyze = cache(analyzer.analyze) # Memoize sighting analysis for performance
        self._tear_matcher = tear_matcher

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Return one similarity score per catalog candidate."""
        query_analysis = self._analyze(query)
        scores: dict[CandidateKey, float] = {}
        for candidate_key, evidence in catalog.items():
            candidate_analyses = tuple(self._analyze(pair) for pair in evidence)
            candidate_match = self._match_candidate(
                candidate_key,
                query_analysis,
                candidate_analyses,
            )
            scores[candidate_key] = candidate_match.score
        return scores

    def _match_candidate(
        self,
        candidate_key: CandidateKey,
        query: SightingAnalysis,
        candidate_analyses: tuple[SightingAnalysis, ...],
    ) -> _CandidateMatch:
        """Return the independently strongest match for each ear side.

        Raises:
            RuntimeError: If the candidate has no analyzed sightings.
        """
        if not candidate_analyses:
            raise RuntimeError(f"{candidate_key} has no catalog evidence")
        return _CandidateMatch(
            left=self._match_candidate_side(
                query.left,
                (pair_analysis.left for pair_analysis in candidate_analyses),
            ),
            right=self._match_candidate_side(
                query.right,
                (pair_analysis.right for pair_analysis in candidate_analyses),
            ),
        )

    def _match_candidate_side(
        self,
        query: EarAnalysis,
        catalog: Iterable[EarAnalysis],
    ) -> _SideMatch:
        """Return the strongest match for one ear side."""

        def side_match_key(match: _SideMatch) -> tuple[float, int, int]:
            return (
                match.score,
                -match.catalog_evidence.source_photo.sighting_id.int,
                -match.catalog_evidence.source_photo.photo_id.int,
            )

        return max( # TODO: tune this aggregation function (could be avg or median)
            (self._compute_side_match(query, ear) for ear in catalog),
            key=side_match_key, # Order matches by highest score, then stable evidence keys
        )

    def _compute_side_match(self, query: EarAnalysis, candidate: EarAnalysis) -> _SideMatch:
        """Return symmetric similarity and provenance for one ear side."""
        return _SideMatch(
            catalog_evidence=candidate,
            tear_match=self._tear_matcher.match(
                query.tear_profile.depths,
                candidate.tear_profile.depths,
            ),
        )
