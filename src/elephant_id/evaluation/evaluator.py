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

_BOOTSTRAP_SAMPLES = 100_000
_BOOTSTRAP_SEED = 42

Scores = dict[str, dict[UUID, dict[str, float]]]
"""True known elephant to eligible query sighting ID to known elephant to candidate score."""

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

    Returns:
        The candidate scores of every eligible query, the micro and macro metrics, and their 95% intervals.

    Raises:
        ValueError: If the benchmark has no eligible query or the matcher returns invalid scores.
        Exception: Any matcher failure, annotated with the query sighting ID.
    """
    scores = _score_eligible_queries(benchmark, build_matcher, workers)
    ranks = _true_elephant_ranks(scores)
    return EvaluationResult(scores, _metrics(ranks), _intervals(ranks))


def _score_eligible_queries(
    benchmark: Benchmark,
    build_matcher: Callable[[], CatalogMatcher],
    workers: int,
) -> Scores:
    """Return the candidate scores of each eligible query, grouped by its true known elephant.

    Raises:
        ValueError: If the benchmark has no eligible query or the matcher returns invalid scores.
    """
    queries_by_name = _eligible_queries(benchmark)
    if not queries_by_name:
        raise ValueError("Benchmark contains no eligible queries")

    all_queries = [query for queries in queries_by_name.values() for query in queries]
    candidate_scores_by_sighting_id = _match_queries(all_queries, benchmark, build_matcher, workers)

    scores: Scores = {}
    for true_name, queries in queries_by_name.items():
        scores[true_name] = {query.sighting_id: candidate_scores_by_sighting_id[query.sighting_id] for query in queries}
    return scores


def _eligible_queries(benchmark: Benchmark) -> dict[str, list[SightingEarPair]]:
    """Return the eligible queries of each known elephant that has any, in benchmark order.

    A sighting ear pair is an eligible query when its elephant has another sighting ear pair, so that the elephant
    still has reference sightings after the query is held out.
    """
    return {name: sighting_ear_pairs for name, sighting_ear_pairs in benchmark.items() if len(sighting_ear_pairs) > 1}


def _match_queries(
    queries: list[SightingEarPair],
    benchmark: Benchmark,
    build_matcher: Callable[[], CatalogMatcher],
    workers: int,
) -> dict[UUID, dict[str, float]]:
    """Return the candidate score of each known elephant for each query, by query sighting ID.

    The matcher sees a run-scoped candidate key for each known elephant, never its name.
    """
    key_by_name = {name: CandidateKey(uuid4()) for name in benchmark}
    name_by_key = {key: name for name, key in key_by_name.items()}
    catalogs = [_catalog_without(query, benchmark, key_by_name) for query in queries]
    results = _match_all(build_matcher, queries, catalogs, workers)
    progress = tqdm(results, total=len(queries), desc="Evaluating queries", unit="query", disable=None)

    candidate_scores_by_sighting_id: dict[UUID, dict[str, float]] = {}
    for query, score_by_key in zip(queries, progress, strict=True):
        candidate_scores_by_sighting_id[query.sighting_id] = {name_by_key[key]: score for key, score in score_by_key.items()}
    return candidate_scores_by_sighting_id


def _catalog_without(query: SightingEarPair, benchmark: Benchmark, key_by_name: dict[str, CandidateKey]) -> Catalog:
    """Return the reference sightings of a query: every other benchmark sighting, grouped by candidate key."""
    return {
        key_by_name[name]: tuple(
            sighting_ear_pair
            for sighting_ear_pair in sighting_ear_pairs
            if sighting_ear_pair.sighting_id != query.sighting_id
        )
        for name, sighting_ear_pairs in benchmark.items()
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


def _true_elephant_ranks(scores: Scores) -> dict[str, np.ndarray]:
    """Return the rank of the true known elephant in each of its eligible queries, grouped by that elephant."""
    ranks: dict[str, np.ndarray] = {}
    for true_name, candidate_scores_by_sighting_id in scores.items():
        query_ranks = [_rank(candidate_scores, true_name) for candidate_scores in candidate_scores_by_sighting_id.values()]
        ranks[true_name] = np.array(query_ranks)
    return ranks


def _rank(candidate_scores: dict[str, float], true_name: str) -> int:
    """Return one plus the number of known elephants with a strictly higher candidate score than the true elephant.

    A known elephant that ties with the true elephant does not lower its rank.
    """
    true_score = candidate_scores[true_name]
    return 1 + sum(score > true_score for score in candidate_scores.values())


def _top_k_hits(ranks: np.ndarray, k: int) -> np.ndarray:
    """Return 1 for each rank of `k` or better, and 0 otherwise."""
    return (ranks <= k).astype(float)


def _reciprocal_ranks(ranks: np.ndarray) -> np.ndarray:
    """Return 1 / rank for each rank."""
    return 1 / ranks


_METRICS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "top_1": partial(_top_k_hits, k=1),
    "top_3": partial(_top_k_hits, k=3),
    "top_5": partial(_top_k_hits, k=5),
    "top_10": partial(_top_k_hits, k=10),
    "top_15": partial(_top_k_hits, k=15),
    "mrr": _reciprocal_ranks,
}
"""Metric name to the function that gives one value per query rank. The metric is the mean of these values."""


def _metrics(ranks: dict[str, np.ndarray]) -> dict[str, float]:
    """Return the eligible query count, then the micro average of each metric, then the macro average of each metric.

    - The micro average gives each eligible query equal weight.
    - The macro average, with the prefix `macro_`, gives each known elephant equal weight.
    """
    all_ranks = np.concatenate(list(ranks.values()))
    metrics: dict[str, float] = {"eligible_queries": len(all_ranks)}
    for metric, query_values in _METRICS.items():
        metrics[metric] = float(query_values(all_ranks).mean())
    for metric, query_values in _METRICS.items():
        elephant_means = [query_values(elephant_ranks).mean() for elephant_ranks in ranks.values()]
        metrics[f"macro_{metric}"] = float(np.mean(elephant_means))
    return metrics


def _intervals(ranks: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """Return the 95% bootstrap interval of each micro average, then of each macro average.

    Each bootstrap sample draws known elephants with replacement and keeps all eligible queries of each drawn
    elephant. An elephant drawn twice counts twice.
    """
    ranks_of_each_elephant = list(ranks.values())
    samples = _bootstrap_samples(len(ranks_of_each_elephant))
    query_counts = np.array([len(elephant_ranks) for elephant_ranks in ranks_of_each_elephant])
    sample_query_counts = query_counts[samples].sum(axis=1)

    intervals: dict[str, tuple[float, float]] = {}
    for metric, query_values in _METRICS.items():
        elephant_sums = np.array([query_values(elephant_ranks).sum() for elephant_ranks in ranks_of_each_elephant])
        sample_micro_averages = elephant_sums[samples].sum(axis=1) / sample_query_counts
        intervals[metric] = _percentile_interval(sample_micro_averages)
    for metric, query_values in _METRICS.items():
        elephant_means = np.array([query_values(elephant_ranks).mean() for elephant_ranks in ranks_of_each_elephant])
        sample_macro_averages = elephant_means[samples].mean(axis=1)
        intervals[f"macro_{metric}"] = _percentile_interval(sample_macro_averages)
    return intervals


def _bootstrap_samples(elephant_count: int) -> np.ndarray:
    """Return one bootstrap sample per row: `elephant_count` elephant positions, drawn with replacement."""
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    return rng.integers(elephant_count, size=(_BOOTSTRAP_SAMPLES, elephant_count))


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    """Return the 2.5th and 97.5th percentiles of the values."""
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
