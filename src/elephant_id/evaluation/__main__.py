"""Run the standard retrieval evaluation."""

import argparse
import json
from pathlib import Path

from elephant_id.composition import build_standard_alphaphant
from elephant_id.dataset import Dataset
from elephant_id.log import configure_logging

from . import evaluate, load_benchmark

_MANIFEST = Path("dataset/elephants-alive/benchmark/manifest.csv")
_DATASET_ROOT = Path("dataset/elephants-alive/coded")
_METADATA = Path("dataset/elephants-alive/images.csv")


def main() -> None:
    """Evaluate standard AlphaPhant and print its summary as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=_MANIFEST)
    args = parser.parse_args()

    configure_logging()
    dataset = Dataset(_DATASET_ROOT, _METADATA)
    matcher = build_standard_alphaphant(dataset.photo_store)
    result = evaluate(load_benchmark(args.manifest), dataset, matcher)
    print(json.dumps({"metrics": result.metrics, "intervals": result.intervals}, indent=2))


if __name__ == "__main__":
    main()
