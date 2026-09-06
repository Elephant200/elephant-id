"""AlphaPhant catalog matching with directional tear-profile evidence."""

from collections.abc import Mapping, Sequence
from functools import cache

import numpy as np

from elephant_id.analysis import EarSide, SightingAnalyzer
from elephant_id.domain import SightingEarPair
from elephant_id.matching.protocol import CandidateKey, CandidateScores
from elephant_id.matching.tear_matcher import TearMatcher

_NEIGHBOR_COUNT = 10


class AlphaPhant:
    """Compare same-side ears, combine sightings, and average both ear scores."""

    def __init__(
        self,
        *,
        scale_analyzers: Sequence[SightingAnalyzer],
        channel_matchers: Sequence[TearMatcher],
        channel_weights: Sequence[float] | None = None,
    ) -> None:
        """Compose extraction scales and complementary profile channels.

        Raises:
            ValueError: If a component sequence is empty or weights are invalid.
        """
        if not scale_analyzers or not channel_matchers:
            raise ValueError("AlphaPhant requires scale analyzers and channel matchers")
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
        self._analyze = tuple(cache(analyzer.analyze) for analyzer in scale_analyzers)
        self._channels = tuple(zip(channel_matchers, weights / weights.sum(), strict=True))
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
        for analyze in self._analyze:
            analyze(query)
        for key, evidence in catalog.items():
            if not evidence:
                raise RuntimeError(f"{key} has no catalog evidence")
        if not catalog:
            return {}
        left = self._score_side(query, catalog, "left")
        right = self._score_side(query, catalog, "right")
        return {key: (left[key] + right[key]) / 2.0 for key in catalog}

    def _score_side(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
        side: EarSide,
    ) -> CandidateScores:
        """Correct ear similarities and combine each candidate's sightings."""
        evidence = tuple(
            dict.fromkeys(pair for pairs in catalog.values() for pair in pairs)
        )
        pairs = tuple(dict.fromkeys((query, *evidence)))
        slots = {pair: index for index, pair in enumerate(pairs)}
        catalog_rows = np.asarray([slots[pair] for pair in evidence])
        evidence_slots = {pair: index for index, pair in enumerate(evidence)}
        similarities = self._side_matrix(pairs, side)
        query_similarities = similarities[slots[query], catalog_rows]
        catalog_similarities = similarities[np.ix_(catalog_rows, catalog_rows)]
        corrected = self._correct_catalog(query_similarities, catalog_similarities)
        spread = float(np.std(corrected))
        return {
            key: self._aggregate(
                [corrected[evidence_slots[pair]] for pair in sightings], spread
            )
            for key, sightings in catalog.items()
        }

    def _side_matrix(
        self, pairs: Sequence[SightingEarPair], side: EarSide
    ) -> np.ndarray:
        """Compute missing directional pair similarities in shared query batches."""
        profiles = {
            pair: tuple(
                getattr(analyze(pair), side).tear_profile.depths
                for analyze in self._analyze
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

    def _correct_catalog(self, raw: np.ndarray, internal: np.ndarray) -> np.ndarray:
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

    def _aggregate(self, scores: Sequence[float], spread: float) -> float:
        """Return a similarity-weighted mean of one candidate's sighting evidence."""
        values = np.asarray(scores, dtype=np.float64)
        if spread <= 0.0:
            return float(values.max())
        weights = np.exp((values - values.max()) / spread)
        return float(np.dot(values, weights) / weights.sum())
