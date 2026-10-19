"""Run the standard retrieval evaluation for one catalog matcher."""

import argparse
import json
import os
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

from elephant_id.composition import build_curvrank, build_standard_alphaphant
from elephant_id.dataset import Dataset
from elephant_id.evaluation import evaluate, load_benchmark
from elephant_id.log import configure_logging
from elephant_id.matching import CatalogMatcher

load_dotenv()

_MANIFEST = Path("dataset/elephants-alive/benchmark/benchmark_final.csv")
_DATASET_ROOT = Path("dataset/elephants-alive/coded")
_METADATA = Path("dataset/elephants-alive/images.csv")
_MATCHERS = {"alphaphant": build_standard_alphaphant, "curvrank": build_curvrank}


def main() -> None:
    """Evaluate one catalog matcher and print its summary as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=_MANIFEST)
    parser.add_argument("--matcher", choices=_MATCHERS, default="alphaphant")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="query processes (default: all cores)")
    args = parser.parse_args()

    configure_logging()
    benchmark = load_benchmark(args.manifest, Dataset(_DATASET_ROOT, _METADATA))
    result = evaluate(benchmark, partial(_build_matcher, args.matcher), args.workers)
    print(json.dumps({"metrics": result.metrics, "intervals": result.intervals}, indent=2))


def _build_matcher(name: str) -> CatalogMatcher:
    """Build the named matcher over the research Dataset, with logging set for this process."""
    configure_logging()
    return _MATCHERS[name](Dataset(_DATASET_ROOT, _METADATA).photo_store)

if __name__ == "__main__":
    main()
