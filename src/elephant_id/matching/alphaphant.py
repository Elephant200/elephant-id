"""Concrete AlphaPhant catalog matcher."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from itertools import islice

from elephant_id.analysis import EarAnalysis, SightingAnalysis, SightingAnalyzer
from elephant_id.domain import SightingEarPair
from elephant_id.matching.protocol import CandidateKey, CandidateScores
from elephant_id.matching.tear_matcher import TearMatch, TearMatcher


@dataclass(frozen=True, slots=True)
class _CatalogEarMatch:
    """One analyzed catalog ear and its forward profile match."""

    catalog_ear: EarAnalysis
    profile_match: TearMatch

    @property
    def score(self) -> float:
        """Return the profile's similarity score."""
        return self.profile_match.score


class AlphaPhant:
    """Analyze sighting ear pairs and score every catalog candidate."""

    def __init__(
        self,
        *,
        analyzer: SightingAnalyzer,
        tear_matcher: TearMatcher,
    ) -> None:
        """Initialize AlphaPhant with sighting analysis and tear matching."""
        self._analyze = cache(analyzer.analyze)
        self._tear_matcher = tear_matcher

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Return one similarity score per catalog candidate.

        Raises:
            RuntimeError: If a candidate has no catalog evidence.
        """
        query_analysis = self._analyze(query)
        catalog_analyses = self._analyze_catalog(catalog)

        left_matches = self._match_side(query_analysis.left, catalog_analyses)
        right_matches = self._match_side(query_analysis.right, catalog_analyses)
        scores: dict[CandidateKey, float] = {}

        for candidate_key in catalog:
            best_left = self._select_best_match(left_matches[candidate_key])
            best_right = self._select_best_match(right_matches[candidate_key])
            scores[candidate_key] = (best_left.score + best_right.score) / 2.0

        return scores

    def _analyze_catalog(
        self,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> dict[CandidateKey, tuple[SightingAnalysis, ...]]:
        """Analyze catalog evidence without changing its candidate grouping.

        Raises:
            RuntimeError: If a candidate has no catalog evidence.
        """
        analyses = {}

        for candidate_key, evidence_pairs in catalog.items():
            if not evidence_pairs:
                raise RuntimeError(f"{candidate_key} has no catalog evidence")

            analyses[candidate_key] = tuple(
                self._analyze(pair) for pair in evidence_pairs
            )

        return analyses

    def _match_side(
        self,
        query_ear: EarAnalysis,
        catalog: Mapping[CandidateKey, tuple[SightingAnalysis, ...]],
    ) -> dict[CandidateKey, tuple[_CatalogEarMatch, ...]]:
        """Bulk-match one ear side and retain each candidate's evidence grouping."""
        catalog_ears = tuple(
            analysis.left if query_ear.side == "left" else analysis.right
            for evidence in catalog.values()
            for analysis in evidence
        )
        profile_matches = self._tear_matcher.match_many(
            query_ear.tear_profile.depths,
            tuple(ear.tear_profile.depths for ear in catalog_ears),
        )

        ear_matches = (
            _CatalogEarMatch(ear, match)
            for ear, match in zip(catalog_ears, profile_matches, strict=True)
        )
        return {
            key: tuple(islice(ear_matches, len(evidence)))
            for key, evidence in catalog.items()
        }

    @staticmethod
    def _select_best_match(matches: Iterable[_CatalogEarMatch]) -> _CatalogEarMatch:
        """Select the highest score, then the smallest sighting UUID and photo UUID."""
        return max(
            matches,
            key=lambda match: (
                match.score,
                -match.catalog_ear.source_photo.sighting_id.int,
                -match.catalog_ear.source_photo.photo_id.int,
            ),
        )
