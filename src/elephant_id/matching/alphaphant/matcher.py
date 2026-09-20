"""AlphaPhant analysis and corresponding-side catalog matching."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from itertools import islice

from elephant_id.domain import SightingEarPair
from elephant_id.matching.alphaphant.extraction import (
    AnalyzedEar,
    AnalyzedSightingEarPair,
    TearProfileExtractor,
)
from elephant_id.matching.alphaphant.similarity import (
    EarComparison,
    EarProfileSimilarity,
)
from elephant_id.matching.protocol import CandidateKey, CandidateScores, MatchingError
from elephant_id.preparation import EarSide, PreparedEar


@dataclass(frozen=True, slots=True)
class _ReferenceComparison:
    """A reference ear and the directional comparison it supports."""

    reference_ear: AnalyzedEar
    comparison: EarComparison

    @property
    def score(self) -> float:
        """Return the comparison's similarity score."""
        return self.comparison.score


class AlphaPhant:
    """Analyze sighting ears and score candidates using corresponding sides."""

    def __init__(
        self,
        *,
        prepare_ears: Callable[[SightingEarPair], tuple[PreparedEar, PreparedEar]],
        profile_extractor: TearProfileExtractor,
        ear_similarity: EarProfileSimilarity,
    ) -> None:
        """Compose left/right preparation, profile extraction, and comparison."""
        self._prepare_ears = prepare_ears
        self._profile_extractor = profile_extractor
        self._ear_similarity = ear_similarity
        self._analysis_for = cache(self._analyze_pair)

    def analyze(self, pair: SightingEarPair) -> AnalyzedSightingEarPair:
        """Return source-labelled profiles for both ears. Reuse completed results.

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
            profile = self._profile_extractor.extract(ear)
        except Exception as error:
            raise MatchingError(
                photo=ear.source_photo,
                side=side,
                stage="tear-profile extraction",
                message=str(error),
            ) from error
        return AnalyzedEar(ear.source_photo, side, ear.source_box, profile)

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Average each candidate's strongest left and right reference scores.

        The two winning references may come from different sightings.

        Returns:
            One finite similarity per supplied candidate; larger is stronger.

        Raises:
            RuntimeError: If a candidate has no reference sightings.
            MatchingError: If query or reference analysis fails.
        """
        query_analysis = self.analyze(query)
        reference_analyses = self._analyze_catalog(catalog)
        left_comparisons = self._compare_side(query_analysis.left, reference_analyses)
        right_comparisons = self._compare_side(query_analysis.right, reference_analyses)

        scores: dict[CandidateKey, float] = {}
        for candidate_key in catalog:
            best_left = self._select_best_comparison(left_comparisons[candidate_key])
            best_right = self._select_best_comparison(right_comparisons[candidate_key])
            scores[candidate_key] = (best_left.score + best_right.score) / 2.0
        return scores

    def _analyze_catalog(
        self,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> dict[CandidateKey, tuple[AnalyzedSightingEarPair, ...]]:
        """Analyze references in supplied candidate and sighting order.

        Raises:
            RuntimeError: If a candidate has no reference sightings.
        """
        analyses = {}
        for candidate_key, reference_pairs in catalog.items():
            if not reference_pairs:
                raise RuntimeError(f"{candidate_key} has no catalog evidence")
            analyses[candidate_key] = tuple(self.analyze(pair) for pair in reference_pairs)
        return analyses

    def _compare_side(
        self,
        query_ear: AnalyzedEar,
        catalog: Mapping[CandidateKey, tuple[AnalyzedSightingEarPair, ...]],
    ) -> dict[CandidateKey, tuple[_ReferenceComparison, ...]]:
        """Return same-side comparisons grouped by candidate.

        Keep reference order and repeated references.
        """
        reference_ears = tuple(
            analysis.left if query_ear.side == "left" else analysis.right
            for references in catalog.values()
            for analysis in references
        )
        comparisons = self._ear_similarity.compare_many(
            query_ear.tear_profile.depths,
            tuple(ear.tear_profile.depths for ear in reference_ears),
        )
        labelled = (
            _ReferenceComparison(ear, comparison)
            for ear, comparison in zip(reference_ears, comparisons, strict=True)
        )
        return {
            key: tuple(islice(labelled, len(references)))
            for key, references in catalog.items()
        }

    @staticmethod
    def _select_best_comparison(
        comparisons: Iterable[_ReferenceComparison],
    ) -> _ReferenceComparison:
        """Select the strongest reference, retaining its source and alignment."""
        return max(comparisons, key=lambda comparison: comparison.score)
