"""AlphaPhant catalog matching with directional tear-profile evidence."""

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

_NEIGHBOR_COUNT = 10
CHANNEL_WEIGHTS = (0.55, 0.45)
"""Depth and signed-depth-change shares of directional ear similarity."""


class AlphaPhant:
    """Compare same-side ears, combine sightings, and average both ear scores."""

    def __init__(
        self,
        *,
        prepare_ears: Callable[[SightingEarPair], tuple[PreparedEar, PreparedEar]],
        profile_extractors: Sequence[TearProfileExtractor],
        channel_matchers: Sequence[TearMatcher],
        channel_weights: Sequence[float] | None = None,
    ) -> None:
        """Compose extraction scales and complementary profile channels.

        Raises:
            ValueError: If a component sequence is empty or weights are invalid.
        """
        if not profile_extractors or not channel_matchers:
            raise ValueError(
                "AlphaPhant requires profile extractors and channel matchers"
            )
        weights = np.asarray(
            channel_weights
            if channel_weights is not None
            else [1.0] * len(channel_matchers),
            dtype=np.float64,
        )
        if (
            weights.shape != (len(channel_matchers),)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.any(weights > 0)
        ):
            raise ValueError(
                "Channel weights must be finite, nonnegative, and match the channels"
            )
        weights = weights / weights.max()
        self._prepare_ears = prepare_ears
        self._extractors = tuple(profile_extractors)
        self._profiles = cache(self._extract_profiles)
        self._channels = tuple(
            zip(channel_matchers, weights / weights.sum(), strict=True)
        )
        self._similarities: dict[
            tuple[SightingEarPair, SightingEarPair, EarSide], float
        ] = {}

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Score each supplied candidate without making an identity decision.

        Raises:
            RuntimeError: If a candidate has no catalog evidence.
        """
        self.profiles(query)
        return score_catalog(query, catalog, self._side_matrix)

    def profiles(self, pair: SightingEarPair) -> tuple[SightingProfiles, ...]:
        """Return source-labelled left/right profiles in configured scale order.

        Research inspection uses the same memoized extraction as matching.
        """
        return self._profiles(pair)

    def _extract_profiles(self, pair: SightingEarPair) -> tuple[SightingProfiles, ...]:
        """Prepare once and extract each configured scale."""
        ears = self._prepare_ears(pair)
        return tuple(
            extract_sighting_profiles(ears, extractor) for extractor in self._extractors
        )

    def _side_matrix(
        self, pairs: Sequence[SightingEarPair], side: EarSide
    ) -> np.ndarray:
        """Compute missing directional pair similarities in shared query batches."""
        profiles = {
            pair: tuple(
                getattr(profiles, side).tear_profile.depths
                for profiles in self.profiles(pair)
            )
            for pair in pairs
        }
        for first in pairs:
            missing = tuple(
                second
                for second in pairs
                if (first, second, side) not in self._similarities
            )
            if not missing:
                continue
            scores = np.zeros(len(missing))
            stacks = tuple(profiles[second] for second in missing)
            for matcher, weight in self._channels:
                matches = matcher.match_stack_many(profiles[first], stacks)
                scores += weight * np.asarray([match.score for match in matches])
            for second, value in zip(missing, scores, strict=True):
                self._similarities[first, second, side] = float(value)
        return np.asarray(
            [
                [self._similarities[first, second, side] for second in pairs]
                for first in pairs
            ]
        )


def correct_catalog(raw: np.ndarray, internal: np.ndarray) -> np.ndarray:
    """Discount each catalog ear's mean similarity to its strongest neighbors.

    Only the supplied catalog enters the neighborhood. Pair-score caching
    never puts a held-out query back into the catalog calculation.
    """
    if len(raw) < 2:
        return 2.0 * raw
    neighbors = internal.copy()
    np.fill_diagonal(neighbors, -np.inf)
    count = min(_NEIGHBOR_COUNT, len(raw) - 1)
    background = np.sort(neighbors, axis=1)[:, ::-1][:, :count].mean(axis=1)
    return 2.0 * raw - background


def aggregate_evidence(scores: Sequence[float], spread: float) -> float:
    """Return a similarity-weighted mean of one candidate's sighting evidence."""
    values = np.asarray(scores, dtype=np.float64)
    if spread <= 0.0:
        return float(values.max())
    weights = np.exp((values - values.max()) / spread)
    return float(np.dot(values, weights) / weights.sum())


def score_catalog(
    query: SightingEarPair,
    catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    side_matrix: Callable[[Sequence[SightingEarPair], EarSide], np.ndarray],
    *,
    correction: Callable[[np.ndarray, np.ndarray], np.ndarray] = correct_catalog,
    aggregation: Callable[[Sequence[float], float], float] = aggregate_evidence,
) -> CandidateScores:
    """Score a catalog from directional similarities, with research scoring substitutions.

    The matrix callable returns rows and columns in the supplied pair order.
    Only current catalog evidence enters background correction.

    Raises:
        RuntimeError: If a candidate has no catalog evidence.
    """
    for key, evidence in catalog.items():
        if not evidence:
            raise RuntimeError(f"{key} has no catalog evidence")
    if not catalog:
        return {}
    left = _score_side(query, catalog, "left", side_matrix, correction, aggregation)
    right = _score_side(query, catalog, "right", side_matrix, correction, aggregation)
    return {key: (left[key] + right[key]) / 2.0 for key in catalog}


def _score_side(
    query: SightingEarPair,
    catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    side: EarSide,
    side_matrix: Callable[[Sequence[SightingEarPair], EarSide], np.ndarray],
    correction: Callable[[np.ndarray, np.ndarray], np.ndarray],
    aggregation: Callable[[Sequence[float], float], float],
) -> CandidateScores:
    """Correct ear similarities and combine each candidate's sightings."""
    evidence = tuple(
        dict.fromkeys(pair for pairs in catalog.values() for pair in pairs)
    )
    pairs = tuple(dict.fromkeys((query, *evidence)))
    slots = {pair: index for index, pair in enumerate(pairs)}
    catalog_rows = np.asarray([slots[pair] for pair in evidence])
    evidence_slots = {pair: index for index, pair in enumerate(evidence)}
    similarities = side_matrix(pairs, side)
    query_similarities = similarities[slots[query], catalog_rows]
    catalog_similarities = similarities[np.ix_(catalog_rows, catalog_rows)]
    corrected = correction(query_similarities, catalog_similarities)
    spread = float(np.std(corrected))
    return {
        key: aggregation(
            [corrected[evidence_slots[pair]] for pair in sightings], spread
        )
        for key, sightings in catalog.items()
    }
