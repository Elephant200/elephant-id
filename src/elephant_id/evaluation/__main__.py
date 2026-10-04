"""Run the standard retrieval evaluation."""

import argparse
import json
import os
from pathlib import Path

from elephant_id.composition import build_standard_alphaphant
from elephant_id.dataset import Dataset
from elephant_id.log import configure_logging
from elephant_id.matching.alphaphant import AlphaPhant

from . import evaluate, load_benchmark

_MANIFEST = Path("dataset/elephants-alive/benchmark/benchmark_final.csv")
_DATASET_ROOT = Path("dataset/elephants-alive/coded")
_METADATA = Path("dataset/elephants-alive/images.csv")


def main() -> None:
    """Evaluate standard AlphaPhant and print its summary as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=_MANIFEST)
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="query processes (default: all cores)")
    args = parser.parse_args()

    configure_logging()
    benchmark = load_benchmark(args.manifest, Dataset(_DATASET_ROOT, _METADATA))
    result = evaluate(benchmark, _build_matcher, args.workers)
    print(json.dumps({"metrics": result.metrics, "intervals": result.intervals}, indent=2))


def _build_matcher() -> AlphaPhant:
    """Build standard AlphaPhant over the research Dataset, with logging set for this process."""
    configure_logging()
    return build_standard_alphaphant(Dataset(_DATASET_ROOT, _METADATA).photo_store)

if __name__ == "__main__":
    main()
