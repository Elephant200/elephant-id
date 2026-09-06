"""Leave-one-sighting-out retrieval evaluation."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from numbers import Real
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
from tqdm import tqdm

from elephant_id.dataset import Dataset
from elephant_id.domain import SightingEarPair
from elephant_id.matching import CandidateKey, CatalogMatcher, MatchingError

from .benchmark import BenchmarkValidationError, RetrievalBenchmark

_BOOTSTRAP_RESAMPLES = 100_000
_BOOTSTRAP_SEED = 42
_METRIC_KEYS = ("top_1", "top_3", "top_5", "top_10", "top_15", "mrr", "median_rank")

_Scores = dict[str, dict[UUID, dict[str, float]]]


class EvaluationStage(StrEnum):
    """Stages at which a matcher fold can fail."""

    CATALOG_MATCHING = "catalog matching"
    MATCHER_RESULT_VALIDATION = "matcher result validation"


class EvaluationError(RuntimeError):
    """Report a failed query fold with recoverable evidence context."""

    def __init__(
        self,
        *,
        stage: EvaluationStage,
        query_sighting_id: UUID,
        role: Literal["query", "catalog"] | None = None,
        photo_id: UUID | None = None,
        side: Literal["left", "right"] | None = None,
    ) -> None:
        """Initialize one structured evaluation failure."""
        self.stage = stage
        self.query_sighting_id = query_sighting_id
        self.role = role
        self.photo_id = photo_id
        self.side = side
        details = [f"query sighting {query_sighting_id}"]
        if role is not None:
            details.append(role)
        if side is not None:
            details.append(f"{side} side")
        if photo_id is not None:
            details.append(f"photo {photo_id}")
        super().__init__(f"{stage.value.capitalize()} failed ({', '.join(details)})")


@dataclass
class EvaluationResult:
    """Canonical per-query scores with derived retrieval summaries."""

    scores: _Scores

    @property
    def ranks(self) -> np.ndarray:
        """Return competition ranks in score mapping order for saved-score analysis."""
        return np.concatenate(tuple(_ranks_by_elephant(self.scores).values()))

    @cached_property
    def metrics(self) -> dict[str, int | float]:
        """Return point metrics derived from the stored candidate scores."""
        ranks = self.ranks
        return _metrics(ranks)

    @cached_property
    def intervals(self) -> dict[str, tuple[float, float]]:
        """Return seeded elephant-cluster bootstrap intervals."""
        return _bootstrap_intervals(_ranks_by_elephant(self.scores))


def _resolve_benchmark(
    benchmark: RetrievalBenchmark,
    dataset: Dataset,
) -> dict[str, dict[UUID, SightingEarPair]]:
    """Resolve all benchmark IDs, reporting every Dataset disagreement."""
    resolved: dict[str, dict[UUID, SightingEarPair]] = {}
    errors: list[str] = []
    for name, sightings in benchmark.sightings.items():
        for sighting_id, (left_id, right_id) in sightings.items():
            error_count = len(errors)
            prefix = f"Sighting {sighting_id}"
            try:
                dataset.sighting(sighting_id)
            except KeyError:
                errors.append(f"{prefix}: unknown sighting_id")
            else:
                if dataset.known_elephant_name(sighting_id) != name:
                    errors.append(f"{prefix}: known_elephant_name does not match Dataset")

            photos = []
            for side, photo_id in (("left", left_id), ("right", right_id)):
                try:
                    photo = dataset.photo(photo_id)
                except KeyError:
                    errors.append(f"{prefix}: unknown {side}_photo_id {photo_id}")
                    photos.append(None)
                    continue
                if photo.sighting_id != sighting_id:
                    errors.append(f"{prefix}: {side}_photo_id belongs to another sighting")
                photos.append(photo)

            if len(errors) == error_count:
                left_photo, right_photo = photos
                assert left_photo is not None and right_photo is not None
                resolved.setdefault(name, {})[sighting_id] = SightingEarPair(
                    sighting_id=sighting_id,
                    left_photo=left_photo,
                    right_photo=right_photo,
                )
    if errors:
        raise BenchmarkValidationError(tuple(errors))
    if not any(len(sightings) > 1 for sightings in resolved.values()):
        raise BenchmarkValidationError(("Benchmark contains no eligible queries",))
    return resolved


def _failure_context(
    error: Exception,
    query: SightingEarPair,
) -> tuple[
    Literal["query", "catalog"] | None,
    UUID | None,
    Literal["left", "right"] | None,
]:
    """Recover the evidence role of a structured matching failure."""
    if not isinstance(error, MatchingError):
        return None, None, None
    query_photos = {query.left_photo.photo_id, query.right_photo.photo_id}
    role: Literal["query", "catalog"] = (
        "query" if error.photo_id in query_photos else "catalog"
    )
    return role, error.photo_id, error.side


def _name_scores(
    scores: object,
    catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    names_by_key: Mapping[CandidateKey, str],
) -> dict[str, float]:
    """Validate scores and privately replace candidate keys with names."""
    if not isinstance(scores, Mapping):
        raise TypeError("matcher result must be a mapping")
    if set(scores) != set(catalog):
        raise ValueError("matcher result keys must exactly match catalog keys")
    named: dict[str, float] = {}
    for key, score in scores.items():
        if not isinstance(score, Real) or isinstance(score, bool) or not math.isfinite(score):
            raise ValueError("matcher scores must be finite real numbers")
        named[names_by_key[key]] = float(score)
    return named


def evaluate(
    benchmark: RetrievalBenchmark,
    dataset: Dataset,
    matcher: CatalogMatcher,
) -> EvaluationResult:
    """Evaluate a matcher with leakage-free leave-one-sighting-out folds.

    Raises:
        BenchmarkValidationError: If benchmark IDs disagree with the Dataset.
        EvaluationError: If matching or matcher-result validation fails.
    """
    evidence = _resolve_benchmark(benchmark, dataset)
    keys_by_name = {name: CandidateKey(uuid4()) for name in evidence}
    names_by_key = {key: name for name, key in keys_by_name.items()}
    result: _Scores = {}
    queries = [
        (name, sighting_id, pair)
        for name, sightings in evidence.items()
        if len(sightings) > 1
        for sighting_id, pair in sightings.items()
    ]

    for true_name, query_id, query in tqdm(
        queries,
        desc="Evaluating queries",
        unit="query",
        disable=None,
    ):
        catalog = {
            keys_by_name[name]: tuple(
                pair
                for pair in candidate_sightings.values()
                if pair.sighting_id != query_id
            )
            for name, candidate_sightings in evidence.items()
        }
        try:
            scores = matcher.match(query, catalog)
        except Exception as error:
            role, photo_id, side = _failure_context(error, query)
            raise EvaluationError(
                stage=EvaluationStage.CATALOG_MATCHING,
                query_sighting_id=query_id,
                role=role,
                photo_id=photo_id,
                side=side,
            ) from error
        try:
            named = _name_scores(scores, catalog, names_by_key)
        except (TypeError, ValueError) as error:
            raise EvaluationError(
                stage=EvaluationStage.MATCHER_RESULT_VALIDATION,
                query_sighting_id=query_id,
            ) from error
        result.setdefault(true_name, {})[query_id] = named

    return EvaluationResult(result)


def _ranks_by_elephant(scores: _Scores) -> dict[str, np.ndarray]:
    """Derive competition ranks grouped by true elephant."""
    return {
        name: np.asarray(
            [
                1 + sum(score > candidates[name] for score in candidates.values())
                for candidates in queries.values()
            ],
            dtype=np.float64,
        )
        for name, queries in scores.items()
    }


def _metrics(ranks: np.ndarray) -> dict[str, int | float]:
    """Compute ordinary query-weighted retrieval metrics."""
    return {
        "eligible_queries": int(ranks.size),
        **{
            f"top_{cutoff}": float(np.mean(ranks <= cutoff))
            for cutoff in (1, 3, 5, 10, 15)
        },
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
    }


def _bootstrap_intervals(
    ranks_by_elephant: Mapping[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    """Bootstrap elephants while retaining every nested eligible query."""
    clusters = tuple(ranks_by_elephant.values())
    padded = np.full((len(clusters), max(map(len, clusters))), np.nan)
    for index, ranks in enumerate(clusters):
        padded[index, : len(ranks)] = ranks

    samples = {key: np.empty(_BOOTSTRAP_RESAMPLES) for key in _METRIC_KEYS}
    generator = np.random.default_rng(_BOOTSTRAP_SEED)
    batch_size = 1_000
    for start in range(0, _BOOTSTRAP_RESAMPLES, batch_size):
        stop = min(start + batch_size, _BOOTSTRAP_RESAMPLES)
        selected = padded[
            generator.integers(0, len(clusters), size=(stop - start, len(clusters)))
        ].reshape(stop - start, -1)
        counts = np.sum(~np.isnan(selected), axis=1)
        for cutoff in (1, 3, 5, 10, 15):
            samples[f"top_{cutoff}"][start:stop] = np.sum(
                selected <= cutoff, axis=1
            ) / counts
        samples["mrr"][start:stop] = np.nansum(1.0 / selected, axis=1) / counts
        samples["median_rank"][start:stop] = np.nanmedian(selected, axis=1)

    return {
        key: tuple(float(value) for value in np.percentile(values, (2.5, 97.5)))
        for key, values in samples.items()
    }
