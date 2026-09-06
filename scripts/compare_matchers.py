"""Compare AlphaPhant, CurvRank, and zero-shot MiewID on identical sighting folds."""

import argparse
import itertools
import json
import time
from pathlib import Path

from elephant_id.analysis.profile_extraction.alpha_tear import MULTISCALE_VERSIONS
from elephant_id.composition import build_versioned_analyzers, compose_alphaphant
from elephant_id.dataset import Dataset, PhotoStore
from elephant_id.evaluation import evaluate, load_benchmark
from elephant_id.evaluation.pooled import paired_delta, pool_metrics
from elephant_id.log import configure_logging
from elephant_id.matching import CatalogMatcher
from elephant_id.matching.curvrank import CurvRankMatcher
from elephant_id.matching.miewid import MiewIdEmbedder, MiewIdMatcher

MIEWID_MODEL = "conservationxlabs/miewid-msv3"
MIEWID_REVISION = "4f1d7f2b521149e5fe34bb85f377248ce9971a7d"


def build_matchers(photo_store: PhotoStore) -> dict[str, CatalogMatcher]:
    """Share one preparation computation across all three catalog matchers."""
    scale_analyzers = build_versioned_analyzers(photo_store, MULTISCALE_VERSIONS)
    return {
        "alphaphant": compose_alphaphant(scale_analyzers),
        "curvrank": CurvRankMatcher(prepare_ears=scale_analyzers[0].prepare),
        "miewid": MiewIdMatcher(
            prepare_ears=scale_analyzers[0].prepare,
            photo_store=photo_store,
            embedder=MiewIdEmbedder(model_id=MIEWID_MODEL, revision=MIEWID_REVISION),
        ),
    }


def main() -> None:
    """Retain canonical scores and paired elephant-bootstrap comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Comparison output already exists: {args.output}")
    configure_logging()
    dataset = Dataset(
        Path("dataset/elephants-alive/coded"),
        Path("dataset/elephants-alive/images.csv"),
    )
    benchmark = load_benchmark(args.manifest)
    report: dict = {
        "manifest": str(args.manifest),
        "complete": False,
        "miewid_revision": MIEWID_REVISION,
    }
    scored = {}
    for name, matcher in build_matchers(dataset.photo_store).items():
        started = time.perf_counter()
        result = evaluate(benchmark, dataset, matcher)
        scores = {
            elephant: {
                str(sid): dict(candidates) for sid, candidates in queries.items()
            }
            for elephant, queries in result.scores.items()
        }
        scored[name] = result.scores
        tied = sum(
            sum(value == candidates[elephant] for value in candidates.values()) > 1
            for elephant, queries in scores.items()
            for candidates in queries.values()
        )
        report[name] = {
            "scores": scores,
            "metrics": result.metrics,
            "intervals": result.intervals,
            "pool_matched": pool_metrics(result.scores),
            "target_tie_queries": tied,
            "seconds": time.perf_counter() - started,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(name, json.dumps(report[name]["pool_matched"]), flush=True)
    report["paired_deltas"] = {
        f"{first} - {second}": {
            str(cutoff): paired_delta(scored[first], scored[second], cutoff)
            for cutoff in (1, 3, 5, 10, 15)
        }
        for first, second in itertools.combinations(scored, 2)
    }
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["paired_deltas"], indent=2), flush=True)


if __name__ == "__main__":
    main()
