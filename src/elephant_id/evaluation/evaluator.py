"""Leave-one-sighting-out retrieval evaluation."""

import math
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from uuid import UUID, uuid4

import numpy as np
from tqdm import tqdm

from elephant_id.domain import SightingEarPair
from elephant_id.matching import CandidateKey, CatalogMatcher

from .benchmark import Benchmark

_BOOTSTRAP_RESAMPLES = 100_000
_BOOTSTRAP_SEED = 42
_CUTOFFS = (1, 3, 5, 10, 15)

Scores = dict[str, dict[UUID, dict[str, float]]]
"""True known elephant to query sighting ID to matching candidate to candidate score."""

Catalog = dict[CandidateKey, tuple[SightingEarPair, ...]]


@dataclass(frozen=True)
class EvaluationResult:
    """Candidate scores for every eligible query, with the metrics and intervals derived from them."""

    scores: Scores
    metrics: dict[str, float]
    intervals: dict[str, tuple[float, float]]


def evaluate(
    benchmark: Benchmark,
    build_matcher: Callable[[], CatalogMatcher],
    workers: int = 1,
) -> EvaluationResult:
    """Evaluate a catalog matcher with leave-one-sighting-out queries.

    Args:
        benchmark: Sighting ear pairs grouped by known elephant.
        build_matcher: Builds the catalog matcher. With more than one worker, each worker process calls it
            once, so it must be picklable.
        workers: Number of processes that run queries.

    Raises:
        ValueError: If the benchmark has no eligible query or the matcher returns invalid scores.
        Exception: Any matcher failure, annotated with the query sighting ID.
    """
    keys = {name: CandidateKey(uuid4()) for name in benchmark}
    names = {key: name for name, key in keys.items()}

    # A query is eligible when its elephant has another sighting left in the catalog.
    eligible = [(name, query) for name, pairs in benchmark.items() if len(pairs) > 1 for query in pairs]
    if not eligible:
        raise ValueError("Benchmark contains no eligible queries")
    queries = [query for _, query in eligible]
    catalogs = [_catalog_without(query, benchmark, keys) for query in queries]

    scores: Scores = {}
    results = _match_all(build_matcher, queries, catalogs, workers)
    progress = tqdm(results, total=len(queries), desc="Evaluating queries", unit="query", disable=None)
    for (true_name, query), candidate_scores in zip(eligible, progress, strict=True):
        if true_name not in scores:
            scores[true_name] = {}
        scores[true_name][query.sighting_id] = {names[key]: score for key, score in candidate_scores.items()}

    histograms = _rank_histograms(scores, candidate_count=len(benchmark))
    pooled = _metrics(histograms.sum(axis=0, keepdims=True))
    metrics = {"eligible_queries": len(queries), **{name: float(values[0]) for name, values in pooled.items()}}
    return EvaluationResult(scores, metrics, _intervals(histograms))


def _catalog_without(query: SightingEarPair, benchmark: Benchmark, keys: dict[str, CandidateKey]) -> Catalog:
    """Return every benchmark sighting except the query's, grouped by candidate key."""
    return {
        keys[name]: tuple(pair for pair in pairs if pair.sighting_id != query.sighting_id)
        for name, pairs in benchmark.items()
    }


def _match_all(
    build_matcher: Callable[[], CatalogMatcher],
    queries: list[SightingEarPair],
    catalogs: list[Catalog],
    workers: int,
) -> Iterator[dict[CandidateKey, float]]:
    """Yield the scores of each query against its catalog, in query order."""
    if workers == 1:
        yield from map(partial(_match, build_matcher()), queries, catalogs)
        return
    pool = ProcessPoolExecutor(workers, initializer=_start_worker, initargs=(build_matcher,))
    try:
        yield from pool.map(_match_in_worker, queries, catalogs)
    finally:
        pool.shutdown(cancel_futures=True)  # After a failure, skip the queries that have not started.


_worker_matcher: CatalogMatcher
"""The matcher of this worker process. Each worker builds it once and reuses it for all its queries."""


def _start_worker(build_matcher: Callable[[], CatalogMatcher]) -> None:
    """Build the matcher of this worker process."""
    global _worker_matcher
    _worker_matcher = build_matcher()


def _match_in_worker(query: SightingEarPair, catalog: Catalog) -> dict[CandidateKey, float]:
    """Match one query with the matcher of this worker process."""
    return _match(_worker_matcher, query, catalog)


def _match(matcher: CatalogMatcher, query: SightingEarPair, catalog: Catalog) -> dict[CandidateKey, float]:
    """Match one query and check that the result has one finite score for each candidate key.

    Raises:
        ValueError: If the scores are invalid.
        Exception: Any matcher failure. Each error gets a note with the query sighting ID.
    """
    try:
        scores = matcher.match(query, catalog)
        if scores.keys() != catalog.keys() or not all(map(math.isfinite, scores.values())):
            raise ValueError("Matcher must return one finite score for each candidate key")
    except Exception as error:
        error.add_note(f"Query sighting {query.sighting_id}")
        raise
    return {key: float(score) for key, score in scores.items()}


def _rank_histograms(scores: Scores, candidate_count: int) -> np.ndarray:
    """Count the queries of each true elephant by rank.

    The rank is one plus the number of candidates with a higher score than the true elephant.

    Returns:
        One row for each true elephant. Column `r - 1` counts its queries at rank `r`.
    """
    histograms = np.zeros((len(scores), candidate_count), dtype=np.int64)
    for row, (true_name, queries) in enumerate(scores.items()):
        for candidates in queries.values():
            higher = sum(score > candidates[true_name] for score in candidates.values())
            histograms[row, higher] += 1
    return histograms


def _intervals(histograms: np.ndarray) -> dict[str, tuple[float, float]]:
    """Return 95% percentile bootstrap intervals that resample elephants.

    Each resample selects elephants with replacement and keeps all queries of each selected elephant.
    Its rank histogram is thus the sum of the selected elephants' histograms.
    """
    elephants = len(histograms)
    generator = np.random.default_rng(_BOOTSTRAP_SEED)
    times_selected = generator.multinomial(elephants, np.full(elephants, 1 / elephants), size=_BOOTSTRAP_RESAMPLES)
    resampled = _metrics(times_selected @ histograms)
    return {name: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) for name, values in resampled.items()}


def _metrics(histograms: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate the metrics of each rank histogram.

    Args:
        histograms: One histogram per row. Column `r - 1` counts the queries at rank `r`.

    Returns:
        One value per histogram for each metric.
    """
    ranks = np.arange(1, histograms.shape[1] + 1)
    queries = histograms.sum(axis=1)
    at_or_above = histograms.cumsum(axis=1)  # Column `r - 1` counts the queries at rank `r` or better.

    def rank_at(position: np.ndarray) -> np.ndarray:
        """Return the rank at a zero-based position in the sorted ranks of each histogram."""
        return 1 + (at_or_above <= position[:, None]).sum(axis=1)

    return {
        **{f"top_{cutoff}": at_or_above[:, min(cutoff, len(ranks)) - 1] / queries for cutoff in _CUTOFFS},
        "mrr": histograms @ (1 / ranks) / queries,
        "median_rank": (rank_at((queries - 1) // 2) + rank_at(queries // 2)) / 2,
    }
