"""Compute exact seven-decision AlphaPhant Shapley attribution on an explicit manifest."""

import argparse
import hashlib
import inspect
import itertools
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from importlib.metadata import version
from math import factorial
from pathlib import Path

import numpy as np

from elephant_id.analysis import EarSide, SightingAnalyzer
from elephant_id.analysis.profile_extraction.alpha_tear import (
    DEFAULT_VERSION,
    MULTISCALE_VERSIONS,
)
from elephant_id.composition import (
    CHANNEL_WEIGHTS,
    SELECTED_PROFILE_SETTINGS,
    build_versioned_analyzers,
)
from elephant_id.dataset import Dataset
from elephant_id.domain import SightingEarPair
from elephant_id.evaluation import evaluate, load_benchmark
from elephant_id.evaluation.evaluator import _resolve_benchmark
from elephant_id.evaluation.pooled import paired_delta, pool_hits, pool_metrics
from elephant_id.log import configure_logging
from elephant_id.matching import AlphaPhant
from elephant_id.matching.tear_matcher import TearMatcher, TearMatcherConfig

DECISIONS = (
    "depth_shift",
    "angular_weights",
    "scale_stack",
    "depth_change",
    "catalog_neighborhood",
    "evidence_mean",
    "signed_change",
)
CUTOFFS = (1, 3, 5, 10, 15)

DECISION_NAMES = {
    "depth_shift": "Depth compression and shift tolerance",
    "angular_weights": "Angular weighting",
    "scale_stack": "Multiple scales with shared alignment",
    "depth_change": "Depth-change comparison",
    "catalog_neighborhood": "Catalog background correction",
    "evidence_mean": "Similarity-weighted evidence mean",
    "signed_change": "Signed depth change",
}


def _source_fingerprints() -> dict[str, str]:
    """Identify this script and active package sources before reusing matrices."""
    package = Path(inspect.getfile(AlphaPhant)).parents[1]
    files = [Path(__file__)]
    for folder, directories, names in os.walk(package):
        directories[:] = sorted(
            name for name in directories if name not in {"api", "__pycache__"}
        )
        files.extend(Path(folder) / name for name in names if name.endswith(".py"))
    return {
        str(path.relative_to(package))
        if path.is_relative_to(package)
        else "scripts/ablation.py": hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(files)
    }


class _MatrixAlphaPhant(AlphaPhant):
    """Reuse computed profile similarities while retaining actual catalog scoring."""

    def __init__(
        self,
        analyzer: SightingAnalyzer,
        pairs: Sequence[SightingEarPair],
        matrix: np.ndarray,
        chosen: frozenset[str],
    ) -> None:
        """Keep neutral row identities and the two catalog-rule interventions."""
        super().__init__(scale_analyzers=(analyzer,), channel_matchers=(TearMatcher(),))
        self._rows = {pair: index for index, pair in enumerate(pairs)}
        self._matrix = matrix
        self._chosen = chosen

    def _side_matrix(
        self, pairs: Sequence[SightingEarPair], side: EarSide
    ) -> np.ndarray:
        """Read only the requested pairs from the precomputed similarity matrix."""
        rows = [self._rows[pair] for pair in pairs]
        return self._matrix[0 if side == "left" else 1][np.ix_(rows, rows)]

    def _correct_catalog(self, raw: np.ndarray, internal: np.ndarray) -> np.ndarray:
        """Remove the background correction only for its declared ablation."""
        if "catalog_neighborhood" not in self._chosen:
            return raw
        return super()._correct_catalog(raw, internal)

    def _aggregate(self, scores: Sequence[float], spread: float) -> float:
        """Replace the similarity-weighted mean with the original maximum."""
        if "evidence_mean" not in self._chosen:
            return float(max(scores))
        return super()._aggregate(scores, spread)


def _shapley(values: Mapping[int, np.ndarray]) -> np.ndarray:
    """Return exact per-query contributions, averaged over all component orders."""
    count = len(DECISIONS)
    contributions = np.zeros((count, *values[0].shape))
    for feature in range(count):
        for mask in range(1 << count):
            if mask & (1 << feature):
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(count - size - 1) / factorial(count)
            contributions[feature] += weight * (
                values[mask | (1 << feature)] - values[mask]
            )
    np.testing.assert_allclose(
        contributions.sum(axis=0),
        values[(1 << count) - 1] - values[0],
        rtol=0,
        atol=1e-12,
    )
    return contributions


def _matrices(profiles: np.ndarray, folder: Path) -> dict[str, np.ndarray]:
    """Compute only eight extraction/settings cases, reusing them across coalitions."""
    matrices = {}
    for settings, angular, scales in itertools.product((False, True), repeat=3):
        label = f"{int(settings)}{int(angular)}{int(scales)}"
        path = folder / f"matrices_{label}.npz"
        if path.exists():
            matrices[label] = np.load(path)["scores"]
            continue
        config = TearMatcherConfig()
        if settings:
            config = replace(
                config,
                depth_exponent=SELECTED_PROFILE_SETTINGS.depth_exponent,
                shift_penalty_scale=SELECTED_PROFILE_SETTINGS.shift_penalty_scale,
            )
        if angular:
            config = replace(config, bin_weights=SELECTED_PROFILE_SETTINGS.bin_weights)
        matchers = tuple(
            TearMatcher(replace(config, channel=channel))
            for channel in ("depth", "depth_change", "signed_depth_change")
        )
        scale_slice = slice(1, None) if scales else slice(0, 1)
        result = np.zeros((2, 3, len(profiles), len(profiles)))
        for side in range(2):
            stacks = tuple(tuple(stack) for stack in profiles[:, side, scale_slice, :])
            for channel, matcher in enumerate(matchers):
                for query, stack in enumerate(stacks):
                    result[side, channel, query] = [
                        match.score for match in matcher.match_stack_many(stack, stacks)
                    ]
        np.savez_compressed(path, scores=result)
        matrices[label] = result
        print(f"Profile comparison case {label} complete", flush=True)
    return matrices


def _coalition_matrix(mask: int, matrices: Mapping[str, np.ndarray]) -> np.ndarray:
    """Compose the exact profile channels for one declared coalition."""
    label = "".join(str(int(bool(mask & (1 << index)))) for index in range(3))
    channels = matrices[label]
    if not mask & (1 << 3):
        return channels[:, 0]
    change_channel = 2 if mask & (1 << 6) else 1
    return (
        CHANNEL_WEIGHTS[0] * channels[:, 0]
        + CHANNEL_WEIGHTS[1] * channels[:, change_channel]
    )


def main() -> None:
    """Retain all coalition scores, exact Shapley values, and paired comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Actual full-pipeline score report for equality verification.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Ablation output already exists: {args.output}")
    folder = args.output.with_suffix("")
    folder.mkdir(parents=True, exist_ok=True)
    provenance = {
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "sources": _source_fingerprints(),
        "numeric_runtime": {name: version(name) for name in ("numpy", "scipy")},
        "profile_settings": repr(SELECTED_PROFILE_SETTINGS),
        "versions": [
            version.slug for version in (DEFAULT_VERSION, *MULTISCALE_VERSIONS)
        ],
        "decisions": list(DECISIONS),
    }
    lineage = folder / "inputs.json"
    if lineage.exists() and json.loads(lineage.read_text()) != provenance:
        raise ValueError(
            "Cached ablation inputs do not match the current manifest and code"
        )
    lineage.write_text(json.dumps(provenance, indent=2) + "\n")
    configure_logging()
    dataset = Dataset(
        Path("dataset/elephants-alive/coded"),
        Path("dataset/elephants-alive/images.csv"),
    )
    benchmark = load_benchmark(args.manifest)
    resolved = _resolve_benchmark(benchmark, dataset)
    pairs = tuple(
        pair for sightings in resolved.values() for pair in sightings.values()
    )
    scale_analyzers = build_versioned_analyzers(
        dataset.photo_store, (DEFAULT_VERSION, *MULTISCALE_VERSIONS)
    )
    profiles = []
    for pair in pairs:
        analyses = tuple(analyzer.analyze(pair) for analyzer in scale_analyzers)
        profiles.append(
            [
                [getattr(analysis, side).tear_profile.depths for analysis in analyses]
                for side in ("left", "right")
            ]
        )
    matrices = _matrices(np.asarray(profiles), folder)
    reference_report = json.loads(args.reference.read_text())
    reference = reference_report.get("alphaphant", reference_report)["scores"]
    full_mask = (1 << len(DECISIONS)) - 1
    rows = {}
    hits = {}
    baseline = None
    source_error = None
    for mask in (full_mask, *range(full_mask)):
        chosen = frozenset(
            name for index, name in enumerate(DECISIONS) if mask & (1 << index)
        )
        matcher = _MatrixAlphaPhant(
            scale_analyzers[0], pairs, _coalition_matrix(mask, matrices), chosen
        )
        result = evaluate(benchmark, dataset, matcher)
        scores = {
            name: {str(sid): dict(candidates) for sid, candidates in queries.items()}
            for name, queries in result.scores.items()
        }
        scores = {
            name: {sid: scores[name][sid] for sid in sorted(scores[name])}
            for name in sorted(scores)
        }
        if mask == full_mask:
            if set(scores) != set(reference):
                raise ValueError(
                    "Full-pipeline reference has different query elephants"
                )
            source_error = max(
                abs(value - reference[name][sid][key])
                for name, queries in scores.items()
                for sid, candidates in queries.items()
                for key, value in candidates.items()
            )
            if source_error > 1e-12:
                raise ValueError(
                    f"Full coalition changed candidate scores: {source_error}"
                )
        if mask == 0:
            baseline = scores
        query_keys = [
            (name, sid) for name, queries in scores.items() for sid in queries
        ]
        candidate_keys = sorted(next(iter(next(iter(scores.values())).values())))
        values = np.asarray(
            [
                [scores[name][sid][key] for key in candidate_keys]
                for name, sid in query_keys
            ]
        )
        np.savez_compressed(folder / f"coalition_{mask:03d}.npz", scores=values)
        if mask == full_mask:
            (folder / "score_keys.json").write_text(
                json.dumps(
                    {"query_keys": query_keys, "candidates": candidate_keys}, indent=2
                )
            )
        rows[str(mask)] = {
            "decisions": sorted(chosen),
            "pool_matched": pool_metrics(scores, CUTOFFS),
        }
        if baseline is not None:
            rows[str(mask)]["paired_delta_vs_baseline"] = {
                str(k): paired_delta(scores, baseline, k) for k in (1, 5)
            }
        hits[mask] = np.asarray([pool_hits(scores, cutoff) for cutoff in CUTOFFS]).T
        print(
            f"Coalition {mask:03d}: {json.dumps(rows[str(mask)]['pool_matched'])}",
            flush=True,
        )
    rows[str(full_mask)]["paired_delta_vs_baseline"] = {
        str(k): paired_delta(reference, baseline, k) for k in (1, 5)
    }
    contributions = _shapley(hits)
    report = {
        "decision_names": DECISION_NAMES,
        "provenance": provenance,
        "maximum_full_score_difference": source_error,
        "rows": rows,
        "shapley": {
            name: {
                str(cutoff): float(contributions[index, :, column].mean())
                for column, cutoff in enumerate(CUTOFFS)
            }
            for index, name in enumerate(DECISIONS)
        },
    }
    np.savez_compressed(
        folder / "shapley_observations.npz", contributions=contributions
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["shapley"], indent=2), flush=True)


if __name__ == "__main__":
    main()
