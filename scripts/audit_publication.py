"""Audit saved publication evidence without running matchers or reading photos."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CUTOFFS = (1, 3, 5, 10, 15)


def read_json(path: Path) -> dict[str, Any]:
    """Read one stored JSON evidence object."""
    return json.loads(path.read_text())


def score_array(scores: dict[str, Any]) -> tuple[list, list, np.ndarray]:
    """Join canonical query and candidate keys and reject incomplete evidence."""
    queries = [(name, sid) for name in sorted(scores) for sid in sorted(scores[name])]
    if not queries:
        raise ValueError("No queries in score evidence")
    candidates = sorted(scores[queries[0][0]][queries[0][1]])
    if len(candidates) != 89:
        raise ValueError("This audit requires the recorded 89-candidate protocol")
    for name, sid in queries:
        if name not in candidates or sorted(scores[name][sid]) != candidates:
            raise ValueError("Candidate coverage differs across queries")
    values = np.asarray(
        [[scores[name][sid][key] for key in candidates] for name, sid in queries]
    )
    if not np.isfinite(values).all():
        raise ValueError("Candidate scores must be finite")
    return queries, candidates, values


def hits(values: np.ndarray, queries: list, candidates: list) -> np.ndarray:
    """Derive retrieval hits independently by counting strictly greater scores."""
    if values.shape != (len(queries), len(candidates)) or not np.isfinite(values).all():
        raise ValueError("Score matrix shape or values are invalid")
    targets = np.asarray([candidates.index(name) for name, _ in queries])
    target_values = values[np.arange(len(queries)), targets]
    if np.any((values == target_values[:, None]).sum(axis=1) != 1):
        raise ValueError("Target ties require a declared tie rule before this audit")
    ranks = 1 + (values > target_values[:, None]).sum(axis=1)
    return ranks[:, None] <= np.asarray(CUTOFFS)[None, :]


def audit(root: Path, source: Path) -> dict[str, Any]:
    """Verify provenance and recompute metrics, Shapley, and sensitivity summaries."""
    lock = read_json(root / "source_lock.json")
    for relative, expected in lock["sources"].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Locked source changed: {relative}")
    confirmation = read_json(root / "confirmation.json")
    if not confirmation["complete"]:
        raise ValueError("Confirmation is incomplete")
    observations = {}
    common = None
    for name in ("alphaphant", "curvrank", "miewid"):
        queries, candidates, values = score_array(confirmation[name]["scores"])
        if common is not None and common != (queries, candidates):
            raise ValueError("Matchers have different query or candidate keys")
        common = queries, candidates
        observations[name] = hits(values, queries, candidates).astype(float)
        for column, cutoff in enumerate(CUTOFFS):
            expected = confirmation[name]["pool_matched"][f"top_{cutoff}"]
            if abs(observations[name][:, column].mean() - expected) > 1e-12:
                raise ValueError("Reported metric disagrees with canonical scores")
    queries, candidates = common
    ablation = read_json(root / "published_ablation.json")
    keys = read_json(root / "published_ablation/score_keys.json")
    if (
        keys["query_keys"] != [list(key) for key in queries]
        or keys["candidates"] != candidates
    ):
        raise ValueError("Ablation keys differ from confirmation")
    coalition_hits = []
    for mask in range(128):
        values = np.load(root / f"published_ablation/coalition_{mask:03d}.npz")[
            "scores"
        ]
        observed = hits(values, queries, candidates).astype(float)
        coalition_hits.append(observed)
        for column, cutoff in enumerate(CUTOFFS):
            expected = ablation["rows"][str(mask)]["pool_matched"][f"top_{cutoff}"]
            if abs(observed[:, column].mean() - expected) > 1e-12:
                raise ValueError("Ablation metric disagrees with canonical scores")
        if mask == 127:
            full = score_array(confirmation["alphaphant"]["scores"])[2]
            if np.max(np.abs(values - full)) > 1e-12:
                raise ValueError("Full ablation differs from confirmation")
    means = np.asarray(coalition_hits).mean(axis=1)
    contributions = []
    decisions = ablation["provenance"]["decisions"]
    for bit, name in enumerate(decisions):
        contribution = np.zeros(len(CUTOFFS))
        for mask in range(128):
            if mask & (1 << bit):
                continue
            size = mask.bit_count()
            weight = 1 / (7 * math.comb(6, size))
            contribution += weight * (means[mask | (1 << bit)] - means[mask])
        expected = [ablation["shapley"][name][str(k)] for k in CUTOFFS]
        if not np.allclose(contribution, expected, rtol=0, atol=1e-12):
            raise ValueError("Published Shapley differs from independent calculation")
        contributions.append(contribution)
    if not np.allclose(
        np.sum(contributions, axis=0), means[-1] - means[0], rtol=0, atol=1e-12
    ):
        raise ValueError("Shapley contributions do not sum to the full gain")
    elephants = sorted({name for name, _ in queries})
    groups = [
        np.asarray([i for i, query in enumerate(queries) if query[0] == name])
        for name in elephants
    ]
    counts = np.asarray([len(group) for group in groups])
    draws = np.random.default_rng(42).integers(
        0, len(groups), size=(100_000, len(groups))
    )
    sensitivity = {}
    for column, cutoff in ((0, 1), (2, 5)):
        difference = (
            observations["alphaphant"][:, column] - observations["curvrank"][:, column]
        )
        totals = np.asarray([difference[group].sum() for group in groups])
        samples = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
        original = confirmation["paired_deltas"]["alphaphant - curvrank"][str(cutoff)]
        interval = np.quantile(samples, [0.025, 0.975])
        if not np.allclose(interval, original[1], rtol=0, atol=1e-12):
            raise ValueError("Paired interval does not reproduce")
        omitted = (difference.sum() - totals) / (len(difference) - counts)
        sensitivity[str(cutoff)] = {
            "query_weighted_delta": float(difference.mean()),
            "interval_95": interval.tolist(),
            "bonferroni_interval_97_5": np.quantile(samples, [0.0125, 0.9875]).tolist(),
            "equal_elephant_weight_delta": float((totals / counts).mean()),
            "leave_one_query_elephant_out_delta_range": [
                float(omitted.min()),
                float(omitted.max()),
            ],
            "elephants_positive_zero_negative": [
                int((totals > 0).sum()),
                int((totals == 0).sum()),
                int((totals < 0).sum()),
            ],
        }
    evidence = [
        root / "confirmation.json",
        root / "published_ablation.json",
        root / "source_lock.json",
    ]
    evidence += sorted((root / "published_ablation").glob("coalition_*.npz"))
    return {
        "scope": "Post hoc saved-score audit; no new matcher evaluation or data selection",
        "conditional_uncertainty": "Fixed observed catalogs; does not cover historical selection or new-catalog variation",
        "source_files_verified": len(lock["sources"]),
        "eligible_queries": len(queries),
        "queried_elephants": len(elephants),
        "candidate_elephants": len(candidates),
        "ablation_subsets_verified": 128,
        "sensitivity": sensitivity,
        "evidence_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in evidence
        },
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    """Write a new audit artifact while preserving all original evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output already exists")
    result = audit(args.artifacts, args.source)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "evidence_sha256"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
