"""AlphaPhant catalog matching from same-side tear-profile evidence.

For each side, AlphaPhant:

1. compares the query ear with every reference ear;
2. subtracts each reference ear's background similarity to its catalog neighbors;
3. combines each candidate's reference scores into a similarity-weighted mean.

The candidate score is the mean of its left and right scores.
"""

from collections.abc import Callable, Mapping, Sequence
from functools import cache

import numpy as np

from elephant_id.domain import SightingEarPair
from elephant_id.matching.alphaphant.extraction import (
    AnalyzedEar,
    AnalyzedSightingEarPair,
    TearProfileExtractor,
)
from elephant_id.matching.alphaphant.similarity import EarProfileSimilarity
from elephant_id.matching.protocol import CandidateKey, CandidateScores, MatchingError
from elephant_id.preparation import EarSide, PreparedEar

_BACKGROUND_NEIGHBORS = 5
_EVIDENCE_TEMPERATURE_FACTOR = 2.0


class AlphaPhant:
    """Score catalog candidates by same-side tear-profile similarity."""

    def __init__(
        self,
        *,
        prepare_ears: Callable[[SightingEarPair], tuple[PreparedEar, PreparedEar]],
        profile_extractor: TearProfileExtractor,
    ) -> None:
        """Compose shared ear preparation with one tear-profile extractor."""
        self._prepare_ears = prepare_ears
        self._extractor = profile_extractor
        self._similarity = EarProfileSimilarity()
        self._analyses = cache(self._analyze)
        self._slots: dict[SightingEarPair, int] = {}
        self._similarities: dict[tuple[EarSide, int, int], float] = {}

    def analyze(self, pair: SightingEarPair) -> AnalyzedSightingEarPair:
        """Return source-labelled tear profiles for both ears, reusing earlier results.

        Raises:
            MatchingError: If preparation or extraction fails.
        """
        return self._analyses(pair)

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Return one score per candidate, using only the supplied catalog.

        Raises:
            RuntimeError: If a candidate has no reference sightings.
            MatchingError: If query or reference analysis fails.
        """
        self.analyze(query)
        for key, references in catalog.items():
            if not references:
                raise RuntimeError(f"{key} has no catalog evidence")
        if not catalog:
            return {}
        references = tuple(
            dict.fromkeys(pair for pairs in catalog.values() for pair in pairs)
        )
        for pair in references:
            self.analyze(pair)

        left = self._score_side("left", query, references, catalog)
        right = self._score_side("right", query, references, catalog)
        return {key: (left[key] + right[key]) / 2.0 for key in catalog}

    def _analyze(self, pair: SightingEarPair) -> AnalyzedSightingEarPair:
        """Prepare both ears and extract their tear profiles."""
        left, right = self._prepare_ears(pair)
        return AnalyzedSightingEarPair(
            sighting_id=pair.sighting_id,
            left=self._extract_ear(left, "left"),
            right=self._extract_ear(right, "right"),
        )

    def _extract_ear(self, ear: PreparedEar, side: EarSide) -> AnalyzedEar:
        """Extract one ear's tear profiles and label them with their source."""
        try:
            profiles = self._extractor.extract(ear)
        except Exception as error:
            raise MatchingError(
                photo=ear.source_photo,
                side=side,
                stage="tear-profile extraction",
                message=str(error),
            ) from error
        return AnalyzedEar(ear.source_photo, side, ear.source_box, profiles)

    def _score_side(
        self,
        side: EarSide,
        query: SightingEarPair,
        references: Sequence[SightingEarPair],
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Return one score per candidate from this side's ears."""
        similarities = self._similarity_matrix(side, (query, *references), references)
        corrected = _subtract_background(similarities[0], similarities[1:])
        spread = float(np.std(corrected))
        by_reference = dict(zip(references, corrected, strict=True))
        return {
            key: _weighted_evidence_mean([by_reference[pair] for pair in pairs], spread)
            for key, pairs in catalog.items()
        }

    def _similarity_matrix(
        self,
        side: EarSide,
        sources: Sequence[SightingEarPair],
        targets: Sequence[SightingEarPair],
    ) -> np.ndarray:
        """Return directional similarities shaped `(sources, targets)`.

        Comparisons are remembered across calls, so a catalog is compared with
        itself once and each new query adds one row.
        """
        for pair in (*sources, *targets):
            self._slots.setdefault(pair, len(self._slots))
        target_slots = [self._slots[pair] for pair in targets]
        for source in sources:
            source_slot = self._slots[source]
            missing = [
                target
                for target, target_slot in zip(targets, target_slots, strict=True)
                if (side, source_slot, target_slot) not in self._similarities
            ]
            if not missing:
                continue
            scores = self._similarity.compare_many(
                self._depths(source, side),
                [self._depths(target, side) for target in missing],
            )
            for target, score in zip(missing, scores, strict=True):
                self._similarities[side, source_slot, self._slots[target]] = float(score)
        return np.asarray(
            [
                [self._similarities[side, self._slots[source], slot] for slot in target_slots]
                for source in sources
            ]
        )

    def _depths(self, pair: SightingEarPair, side: EarSide) -> tuple[np.ndarray, ...]:
        """Return one side's depth arrays in alpha-scale order."""
        analysis = self.analyze(pair)
        ear = analysis.left if side == "left" else analysis.right
        return tuple(profile.depths for profile in ear.tear_profiles)


def _subtract_background(raw: np.ndarray, internal: np.ndarray) -> np.ndarray:
    """Return `2 * raw - background` for each reference ear.

    A reference ear's background is its mean similarity to its five most similar
    other reference ears. A lone reference ear has zero background.

    Args:
        raw: Query-to-reference similarities, shaped `(references,)`.
        internal: Reference-to-reference similarities, shaped
            `(references, references)`; the diagonal is ignored.
    """
    if len(raw) < 2:
        return 2.0 * raw
    neighbors = internal.copy()
    np.fill_diagonal(neighbors, -np.inf)
    count = min(_BACKGROUND_NEIGHBORS, len(raw) - 1)
    background = np.partition(neighbors, -count, axis=1)[:, -count:].mean(axis=1)
    return 2.0 * raw - background


def _weighted_evidence_mean(scores: Sequence[float], spread: float) -> float:
    """Return a similarity-weighted mean of one candidate's reference scores.

    Weights are `exp((score - best) / (2 * spread))`, where `spread` is the
    standard deviation over all reference scores on this side. Zero spread
    falls back to the best score.
    """
    values = np.asarray(scores, dtype=np.float64)
    if spread <= 0.0:
        return float(values.max())
    weights = np.exp((values - values.max()) / (_EVIDENCE_TEMPERATURE_FACTOR * spread))
    return float(np.dot(values, weights) / weights.sum())
