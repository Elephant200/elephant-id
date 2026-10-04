"""Leave-one-sighting-out retrieval evaluation."""

import math
from dataclasses import dataclass
from uuid import UUID, uuid4

import numpy as np
from tqdm import tqdm

from elephant_id.matching import CandidateKey, CatalogMatcher

from .benchmark import Benchmark

_BOOTSTRAP_RESAMPLES = 100_000
_BOOTSTRAP_SEED = 42
_CUTOFFS = (1, 3, 5, 10, 15)

Scores = dict[str, dict[UUID, dict[str, float]]]
"""True known elephant to query sighting ID to matching candidate to similarity score."""


@dataclass(frozen=True)
class EvaluationResult:
    """Candidate scores for every eligible query, with the metrics and intervals derived from them."""

    scores: Scores
    metrics: dict[str, float]
    intervals: dict[str, tuple[float, float]]


def evaluate(benchmark: Benchmark, matcher: CatalogMatcher) -> EvaluationResult:
    """Evaluate a catalog matcher with leave-one-sighting-out queries.

    Raises:
        ValueError: If the benchmark has no eligible query or the matcher returns invalid scores.
        Exception: Any matcher failure, annotated with the query sighting ID.
    """
    queries = [(name, query) for name, pairs in benchmark.items() if len(pairs) > 1 for query in pairs]
    if not queries:
        raise ValueError("Benchmark contains no eligible queries")
    keys = {name: CandidateKey(uuid4()) for name in benchmark}
    names = {key: name for name, key in keys.items()}

    scores: Scores = {}
    for true_name, query in tqdm(queries, desc="Evaluating queries", unit="query", disable=None):
        catalog = {
            keys[name]: tuple(pair for pair in pairs if pair.sighting_id != query.sighting_id)
            for name, pairs in benchmark.items()
        } # Catalog with query sighting removed for leave-one-sighting-out evaluation
        try:
            candidate_scores = matcher.match(query, catalog)
            if candidate_scores.keys() != catalog.keys() or not all(map(math.isfinite, candidate_scores.values())):
                raise ValueError("Matcher must return one finite score for each candidate key")
        except Exception as error:
            error.add_note(f"Query sighting {query.sighting_id}")
            raise
        named_scores = {names[key]: float(score) for key, score in candidate_scores.items()}
        if true_name not in scores:
            scores[true_name] = {}
        scores[true_name][query.sighting_id] = named_scores

    rank_counts = _rank_counts(scores, candidate_count=len(benchmark))
    metrics = {name: float(values[0]) for name, values in _summarize(rank_counts.sum(axis=0, keepdims=True)).items()}
    return EvaluationResult(scores, {"eligible_queries": int(rank_counts.sum()), **metrics}, _intervals(rank_counts))


def _rank_counts(scores: Scores, candidate_count: int) -> np.ndarray:
    """Count each elephant's queries by rank: row = true elephant, column `r` = rank `r + 1`.

    The rank is one plus the number of candidates with a higher score than the true elephant.
    """
    counts = np.zeros((len(scores), candidate_count), dtype=np.int64)
    for row, (true_name, queries) in enumerate(scores.items()):
        for candidates in queries.values():
            counts[row, sum(score > candidates[true_name] for score in candidates.values())] += 1
    return counts


def _intervals(rank_counts: np.ndarray) -> dict[str, tuple[float, float]]:
    """Return 95% percentile intervals from an elephant bootstrap that keeps all queries of each elephant."""
    elephants = len(rank_counts)
    generator = np.random.default_rng(_BOOTSTRAP_SEED)
    selections = generator.multinomial(elephants, np.full(elephants, 1 / elephants), size=_BOOTSTRAP_RESAMPLES)
    return {
        name: tuple(float(bound) for bound in np.percentile(values, (2.5, 97.5)))
        for name, values in _summarize(selections @ rank_counts).items()
    }


def _summarize(rank_counts: np.ndarray) -> dict[str, np.ndarray]:
    """Return query-weighted metrics for each row of rank counts."""
    ranks = np.arange(1, rank_counts.shape[1] + 1)
    totals = rank_counts.sum(axis=1)
    cumulative = rank_counts.cumsum(axis=1)

    def rank_at(index: np.ndarray) -> np.ndarray:
        """Return the rank at a zero-based position in each row's sorted ranks."""
        return 1 + (cumulative <= index[:, None]).sum(axis=1)

    return {
        **{f"top_{cutoff}": cumulative[:, min(cutoff, len(ranks)) - 1] / totals for cutoff in _CUTOFFS},
        "mrr": rank_counts @ (1 / ranks) / totals,
        "median_rank": (rank_at((totals - 1) // 2) + rank_at(totals // 2)) / 2,
    }
