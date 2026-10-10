"""AlphaPhant analysis and same-side catalog matching."""

from collections.abc import Callable, Mapping
from functools import cache

from elephant_id.domain import SightingEarPair
from elephant_id.matching.alphaphant.extraction import (
    AnalyzedEar,
    AnalyzedSightingEarPair,
    TearProfileExtractor,
)
from elephant_id.matching.alphaphant.similarity import EarComparator
from elephant_id.matching.protocol import CandidateKey, CandidateScores, MatchingError
from elephant_id.preparation import EarSide, PreparedEar


class AlphaPhant:
    """Analyze sighting ear pairs and calculate a candidate score for each candidate."""

    def __init__(
        self,
        *,
        prepare_ears: Callable[[SightingEarPair], tuple[PreparedEar, PreparedEar]],
        profile_extractor: TearProfileExtractor,
        ear_comparator: EarComparator,
    ) -> None:
        """Compose left/right preparation, profile extraction, and comparison.

        Args:
            prepare_ears: Returns the left and right prepared ears of a sighting
                ear pair. Raises `MatchingError` when preparation fails.
            profile_extractor: Extracts one tear profile for each alpha scale.
            ear_comparator: Computes the left score and the right score of each candidate.
        """
        self._prepare_ears = prepare_ears
        self._profile_extractor = profile_extractor
        self._ear_comparator = ear_comparator
        self._analysis_for = cache(self._analyze_pair)

    def analyze(self, pair: SightingEarPair) -> AnalyzedSightingEarPair:
        """Return the analyzed left ear and right ear of the pair.

        This instance stores the analysis and returns it for the same pair.

        Raises:
            MatchingError: If preparation or extraction fails.
        """
        return self._analysis_for(pair)

    def _analyze_pair(self, pair: SightingEarPair) -> AnalyzedSightingEarPair:
        """Return analyzed ears with the input sighting ID."""
        left, right = self._prepare_ears(pair)
        return AnalyzedSightingEarPair(
            sighting_id=pair.sighting_id,
            left=self._extract_ear(left, "left"),
            right=self._extract_ear(right, "right"),
        )

    def _extract_ear(self, ear: PreparedEar, side: EarSide) -> AnalyzedEar:
        """Return an analyzed ear with its source photo, box, and declared side."""
        try:
            profiles = self._profile_extractor.extract(ear)
        except Exception as error:
            raise MatchingError(
                photo=ear.source_photo,
                side=side,
                stage="tear-profile extraction",
                message=str(error),
            ) from error
        return AnalyzedEar(ear.source_photo, side, ear.source_box, profiles)

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Return the candidate score of each candidate: the mean of its left score and right score.

        Returns:
            One finite candidate score per supplied candidate; larger is stronger.

        Raises:
            RuntimeError: If a candidate has no reference sightings.
            MatchingError: If query or reference analysis fails.
        """
        query_analysis = self.analyze(query)
        reference_analyses = self._analyze_catalog(catalog)
        left_scores = self._score_side(query_analysis.left, reference_analyses)
        right_scores = self._score_side(query_analysis.right, reference_analyses)
        return {
            candidate_key: (left_scores[candidate_key] + right_scores[candidate_key]) / 2.0
            for candidate_key in catalog
        }

    def _analyze_catalog(
        self,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> dict[CandidateKey, tuple[AnalyzedSightingEarPair, ...]]:
        """Return the analyzed reference sightings of each candidate.

        Raises:
            RuntimeError: If a candidate has no reference sightings.
        """
        analyses = {}
        for candidate_key, reference_pairs in catalog.items():
            if not reference_pairs:
                raise RuntimeError(f"{candidate_key} has no reference sightings")
            analyses[candidate_key] = tuple(self.analyze(pair) for pair in reference_pairs)
        return analyses

    def _score_side(
        self,
        query_ear: AnalyzedEar,
        catalog: Mapping[CandidateKey, tuple[AnalyzedSightingEarPair, ...]],
    ) -> dict[CandidateKey, float]:
        """Return the side score of the query ear against each candidate's same-side ears."""
        return {
            candidate_key: self._ear_comparator.score_side(
                query_ear,
                [analysis.left if query_ear.side == "left" else analysis.right for analysis in references],
            )
            for candidate_key, references in catalog.items()
        }
