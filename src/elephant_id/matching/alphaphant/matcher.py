"""AlphaPhant analysis and same-side catalog matching."""

from collections.abc import Callable, Iterable, Mapping
from functools import cache

from elephant_id.domain import SightingEarPair
from elephant_id.matching.alphaphant.extraction import (
    AnalyzedEar,
    AnalyzedSightingEarPair,
    TearProfileExtractor,
)
from elephant_id.matching.alphaphant.similarity import EarComparator, EarSimilarity
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
            ear_comparator: Computes ear similarities.
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
        """Return the candidate score of each candidate by aggregating the ear similarities for each side.

        Returns:
            One finite candidate score per supplied candidate; larger is stronger.

        Raises:
            RuntimeError: If a candidate has no reference sightings.
            MatchingError: If query or reference analysis fails.
        """
        query_analysis = self.analyze(query)
        reference_analyses = self._analyze_catalog(catalog)
        left_similarities = self._compare_side(query_analysis.left, reference_analyses)
        right_similarities = self._compare_side(query_analysis.right, reference_analyses)

        scores: dict[CandidateKey, float] = {}
        for candidate_key in catalog:
            left_score = _aggregate_ear_similarities(left_similarities[candidate_key])
            right_score = _aggregate_ear_similarities(right_similarities[candidate_key])
            scores[candidate_key] = (left_score + right_score) / 2.0
        return scores

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

    def _compare_side(
        self,
        query_ear: AnalyzedEar,
        catalog: Mapping[CandidateKey, tuple[AnalyzedSightingEarPair, ...]],
    ) -> dict[CandidateKey, tuple[EarSimilarity, ...]]:
        """Return the ear similarities of the query ear with each candidate's same-side ears.

        The ear similarities of each candidate are in the order of its reference sightings.
        """
        return {
            candidate_key: self._ear_comparator.compare_ears(
                query_ear,
                [analysis.left if query_ear.side == "left" else analysis.right for analysis in references],
            )
            for candidate_key, references in catalog.items()
        }


def _aggregate_ear_similarities(ear_similarities: Iterable[EarSimilarity]) -> float:
    """Return the left score or right score: the highest of a candidate's ear similarities on that side."""
    return max(similarity.score for similarity in ear_similarities)
