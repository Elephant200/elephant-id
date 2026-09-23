"""AlphaPhant catalog matching from same-side tear-profile evidence."""

from collections.abc import Callable, Mapping, Sequence
from functools import cache

import numpy as np

from elephant_id.domain import SightingEarPair
from elephant_id.matching.alphaphant.profile import (
    SightingProfiles,
    TearProfileExtractor,
    extract_sighting_profiles,
)
from elephant_id.matching.alphaphant.similarity import TearMatcher
from elephant_id.matching.protocol import CandidateKey, CandidateScores
from elephant_id.preparation import EarSide, PreparedEar

_NEIGHBOR_COUNT = 5
_EVIDENCE_TEMPERATURE_FACTOR = 2.0


class AlphaPhant:
    """Compare ear profiles, combine each candidate's sightings, and average sides."""

    def __init__(
        self,
        *,
        prepare_ears: Callable[[SightingEarPair], tuple[PreparedEar, PreparedEar]],
        profile_extractor: TearProfileExtractor,
    ) -> None:
        """Accept shared preparation and one complete ear-representation extractor."""
        self._prepare_ears = prepare_ears
        self._extractor = profile_extractor
        self._profiles = cache(self._extract_profiles)
        self._matcher = TearMatcher()
        self._sighting_slots: dict[SightingEarPair, int] = {}
        self._similarities: dict[tuple[int, int, EarSide], float] = {}

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Return one score per candidate using only the supplied catalog evidence.

        Raises:
            RuntimeError: If a candidate has no catalog evidence.
        """
        query_profiles = self.profiles(query)
        for key, evidence in catalog.items():
            if not evidence:
                raise RuntimeError(f"{key} has no catalog evidence")
        if not catalog:
            return {}
        profiles_by_sighting = {query: query_profiles}
        for sightings in catalog.values():
            for pair in sightings:
                if pair not in profiles_by_sighting:
                    profiles_by_sighting[pair] = self.profiles(pair)

        left_scores = self._score_side(query, catalog, profiles_by_sighting, "left")
        right_scores = self._score_side(query, catalog, profiles_by_sighting, "right")
        return {key: (left_scores[key] + right_scores[key]) / 2.0 for key in catalog}

    def profiles(self, pair: SightingEarPair) -> tuple[SightingProfiles, ...]:
        """Return source-labelled profiles in reference order, reusing matching's extraction."""
        return self._profiles(pair)

    def _extract_profiles(self, pair: SightingEarPair) -> tuple[SightingProfiles, ...]:
        """Prepare the sighting and extract its complete ear representations."""
        return extract_sighting_profiles(self._prepare_ears(pair), self._extractor)

    def _side_matrix(
        self,
        profiles_by_sighting: Mapping[SightingEarPair, tuple[SightingProfiles, ...]],
        side: EarSide,
        evidence: Sequence[SightingEarPair],
    ) -> np.ndarray:
        """Compare each source only to catalog evidence in directional batches."""
        pairs = tuple(profiles_by_sighting)
        for pair in pairs:
            if pair not in self._sighting_slots:
                self._sighting_slots[pair] = len(self._sighting_slots)
        sources = tuple((pair, self._sighting_slots[pair]) for pair in pairs)
        targets = tuple((pair, self._sighting_slots[pair]) for pair in evidence)
        depths_by_sighting = {
            pair: tuple(
                getattr(sighting_profile, side).tear_profile.depths
                for sighting_profile in extracted
            )
            for pair, extracted in profiles_by_sighting.items()
        }
        for first, first_slot in sources:
            missing = tuple(
                (second, second_slot)
                for second, second_slot in targets
                if (first_slot, second_slot, side) not in self._similarities
            )
            if missing:
                scores = self._matcher.match_many(
                    depths_by_sighting[first],
                    tuple(depths_by_sighting[second] for second, _ in missing)
                )
                for (_, second_slot), value in zip(missing, scores, strict=True):
                    self._similarities[first_slot, second_slot, side] = float(value)
        return np.asarray(
            [
                [
                    self._similarities[first_slot, second_slot, side]
                    for _, second_slot in targets
                ]
                for _, first_slot in sources
            ]
        )

    def _score_side(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
        profiles_by_sighting: Mapping[SightingEarPair, tuple[SightingProfiles, ...]],
        side: EarSide,
    ) -> CandidateScores:
        """Correct catalog-ear scores and combine each candidate's sightings."""
        evidence = tuple(
            dict.fromkeys(pair for pairs in catalog.values() for pair in pairs)
        )
        slots = {pair: index for index, pair in enumerate(profiles_by_sighting)}
        catalog_rows = np.asarray([slots[pair] for pair in evidence])
        similarities = self._side_matrix(profiles_by_sighting, side, evidence)
        corrected = _correct_catalog(
            similarities[slots[query]],
            similarities[catalog_rows],
        )
        spread = float(np.std(corrected))
        scores = dict(zip(evidence, corrected, strict=True))
        return {
            key: _aggregate_evidence([scores[pair] for pair in sightings], spread)
            for key, sightings in catalog.items()
        }


def _correct_catalog(raw: np.ndarray, internal: np.ndarray) -> np.ndarray:
    """Subtract each catalog ear's mean similarity to its five strongest neighbors.

    Only the supplied catalog matrix determines the neighborhood; cached scores
    from other calls do not add evidence.
    """
    if len(raw) < 2:
        return 2.0 * raw
    neighbors = internal.copy()
    np.fill_diagonal(neighbors, -np.inf)
    count = min(_NEIGHBOR_COUNT, len(raw) - 1)
    background = np.partition(neighbors, -count, axis=1)[:, -count:].mean(axis=1)
    return 2.0 * raw - background


def _aggregate_evidence(scores: Sequence[float], spread: float) -> float:
    """Return a similarity-weighted mean of one candidate's sighting evidence."""
    values = np.asarray(scores, dtype=np.float64)
    if spread <= 0.0:
        return float(values.max())
    weights = np.exp((values - values.max()) / (_EVIDENCE_TEMPERATURE_FACTOR * spread))
    return float(np.dot(values, weights) / weights.sum())
