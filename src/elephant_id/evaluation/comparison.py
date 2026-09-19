"""Paired comparisons of retrieval results on identical actual catalogs."""

from collections.abc import Mapping
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from elephant_id.evaluation.evaluator import EvaluationResult

_Scores = Mapping[str, Mapping[UUID, Mapping[str, float]]]


def top_hits(scores: _Scores, cutoff: int = 1) -> NDArray[np.float64]:
    """Return actual-catalog top-k hits in score mapping order, including ties."""
    return (EvaluationResult(scores).ranks <= cutoff).astype(np.float64)


def paired_delta(
    scores: _Scores,
    reference: _Scores,
    cutoff: int = 1,
    resamples: int = 100_000,
    seed: int = 42,
) -> tuple[float, tuple[float, float]]:
    """Return `(delta, (low, high))` for a paired top-`cutoff` comparison.

    Elephants are resampled with replacement, carrying all of their
    queries, because queries of one elephant are not independent. The
    resampled quantity is the same actual-catalog hit the point
    estimate reports.

    Raises:
        ValueError: If observation or candidate sets differ, or no elephants
            are supplied. Mapping order does not affect observation pairing.
    """
    if scores.keys() != reference.keys() or not scores:
        raise ValueError("Paired results must contain the same nonempty elephant set")
    scores = {
        name: {sighting: scores[name][sighting] for sighting in sorted(scores[name])}
        for name in sorted(scores)
    }
    aligned: dict[str, dict[UUID, Mapping[str, float]]] = {}
    for name, queries in scores.items():
        if queries.keys() != reference[name].keys():
            raise ValueError("Paired results must contain identical query sightings")
        aligned[name] = {}
        for sighting_id, candidates in queries.items():
            other = reference[name][sighting_id]
            if candidates.keys() != other.keys():
                raise ValueError("Paired queries must contain identical candidates")
            aligned[name][sighting_id] = other
    difference = top_hits(scores, cutoff) - top_hits(aligned, cutoff)
    sums = []
    counts = []
    offset = 0
    for queries in scores.values():
        count = len(queries)
        if count:
            sums.append(difference[offset : offset + count].sum())
            counts.append(count)
        offset += count
    sums = np.asarray(sums)
    counts = np.asarray(counts)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(sums), size=(resamples, len(sums)))
    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(difference.mean()), (
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )
