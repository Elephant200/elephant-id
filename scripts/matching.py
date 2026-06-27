"""Database-vs-query tear-matching experiment with alignment-audit visuals.

Pipeline:

1. Load the labeled image set and split each (identity, side) group into an
   enrolled **database** and held-out **queries** (singletons enroll as pure
   distractors).
2. Re-derive tear profiles (cached photo analysis) and reduce them to descriptors.
3. Match every query against the same-side database and score top-1 / top-5 /
   mAP / median rank.
4. Render successful and failed cases with ``tear_matching.visualize`` -- query,
   true match, and strongest distractor as ear crops, plus the query embedding
   against the truth and distractor after alignment.

Run:  uv run python -m scripts.matching [--config wide_sqrt] [--kind mixed] [--count 12]
                                        [--min-truth-mass 0.3]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from elephant_id.coding.photo_analyzer import PhotoAnalyzer
from elephant_id.log import configure_logging
from scripts.tear_matching.matcher import CONFIGS, SETTLED, TearMatcher
from scripts.tear_matching.profiles import (
    OUTPUT_DIR,
    ProfileBank,
    build_sample_bank,
    make_dataset,
)
from scripts.tear_matching.visualize import (
    Case,
    embedding_ylim,
    render_case,
    select_cases,
)

QUERY_FRACTION = 0.5  # share of each multi-photo group held out as queries
_DEFAULT_MATCHER = TearMatcher(SETTLED)


def compute_match_score(tear_profile1: np.ndarray, tear_profile2: np.ndarray) -> float:
    """Similarity of two raw tear profiles in [0, 1] (1 = identical, 0 = disjoint)."""
    return 1.0 - _DEFAULT_MATCHER.match(tear_profile1, tear_profile2)


def split_database_queries(bank: ProfileBank) -> tuple[np.ndarray, np.ndarray]:
    """Split each (identity, side) group into database and query rows.

    Photos are ordered by id; the last ``QUERY_FRACTION`` (at least one, but
    never all) become queries, the rest enroll in the database. Singletons
    enroll database-only, so every query keeps at least one true match.
    """
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in range(len(bank)):
        groups[(bank.elephants[row], bank.sides[row])].append(row)

    database: list[int] = []
    queries: list[int] = []
    for rows in groups.values():
        rows = sorted(rows, key=lambda row: bank.photo_ids[row])
        n_query = min(max(1, int(len(rows) * QUERY_FRACTION)), len(rows) - 1)
        database.extend(rows[: len(rows) - n_query])
        queries.extend(rows[len(rows) - n_query:])
    return np.array(sorted(database)), np.array(sorted(queries))


def query_database_distances(
    matcher: TearMatcher,
    descriptors: list[np.ndarray],
    queries: np.ndarray,
    database: np.ndarray,
    bank: ProfileBank,
) -> np.ndarray:
    """Distance from every query to every same-side database entry (else inf)."""
    distances = np.full((len(queries), len(database)), np.inf)
    for qi, query in enumerate(queries):
        for di, entry in enumerate(database):
            if bank.sides[query] == bank.sides[entry]:
                distances[qi, di] = matcher.distance(descriptors[query], descriptors[entry])
    return distances


def evaluate(
    distances: np.ndarray, bank: ProfileBank, queries: np.ndarray, database: np.ndarray
) -> tuple[list[Case], dict[str, float]]:
    """Score query→database retrieval and collect viewable cases.

    A case is kept only when the query has both a true match and a distractor in
    the same-side database. The summary covers every scored query.
    """
    cases: list[Case] = []
    top1, top5, average_precisions, ranks = [], [], [], []
    for qi, query in enumerate(queries):
        order = np.argsort(distances[qi])
        order = order[np.isfinite(distances[qi][order])]
        if len(order) == 0:
            continue
        entries = database[order]
        relevant = bank.elephants[entries] == bank.elephants[query]
        if not relevant.any():
            continue
        top1.append(float(relevant[0]))
        top5.append(float(relevant[:5].any()))
        ranks.append(int(np.argmax(relevant)) + 1)
        hits = np.flatnonzero(relevant)
        average_precisions.append(float(((np.arange(len(hits)) + 1) / (hits + 1)).mean()))
        if relevant.all():
            continue  # no distractor to compare against
        cases.append(
            Case(
                query=int(query),
                truth=int(entries[np.argmax(relevant)]),
                distractor=int(entries[np.argmax(~relevant)]),
                correct=bool(relevant[0]),
                true_rank=int(np.argmax(relevant)) + 1,
            )
        )
    summary = {
        "queries": float(len(top1)),
        "top1": float(np.mean(top1)),
        "top5": float(np.mean(top5)),
        "mAP": float(np.mean(average_precisions)),
        "median_rank": float(np.median(ranks)),
    }
    return cases, summary


def main() -> None:
    """Run the database/query experiment and render audit cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=tuple(CONFIGS), default="settled")
    parser.add_argument("--kind", choices=("failure", "success", "mixed"), default="mixed")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--min-truth-mass", type=float, default=0.0,
                        help="skip cases whose true match carries less tear mass")
    args = parser.parse_args()

    load_dotenv()
    configure_logging()
    bank = build_sample_bank()
    matcher = TearMatcher(CONFIGS[args.config])
    descriptors = [matcher.descriptor(profile) for profile in bank.profiles]

    database, queries = split_database_queries(bank)
    logger.info(f"{len(queries)} queries vs {len(database)} database entries")
    distances = query_database_distances(matcher, descriptors, queries, database, bank)
    cases, summary = evaluate(distances, bank, queries, database)
    logger.info(
        f"{args.config}: queries={summary['queries']:.0f} top1={summary['top1']:.3f} "
        f"top5={summary['top5']:.3f} mAP={summary['mAP']:.3f} "
        f"median_rank={summary['median_rank']:.0f}"
    )

    dataset = make_dataset()
    analyzer = PhotoAnalyzer(dataset=dataset)
    ylim = embedding_ylim(descriptors)
    case_dir = OUTPUT_DIR / "matching_cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    chosen = select_cases(cases, args.kind, args.count, descriptors, args.min_truth_mass)
    saved = 0
    for case in chosen:
        verdict = "success" if case.correct else "failure"
        path = case_dir / f"{verdict}_{saved:02d}_{bank.elephants[case.query]}_{bank.sides[case.query]}.png"
        if render_case(matcher, dataset, analyzer, bank, descriptors, case, ylim, path):
            saved += 1
    logger.info(f"Saved {saved} case figures to {case_dir}")


if __name__ == "__main__":
    main()
