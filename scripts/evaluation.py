"""Evaluate tear-profile retrieval on the high-quality ear image set.

The scoring stack under test (see docs/reference/matching.md): tear profiles ->
``TearMatcher`` pair scores -> ``symmetrized_cohort_z`` normalization ->
tear-mass-conditioned calibration -> left+right score averaging.

Two protocols are reported:

- Photo-level leave-one-out (diagnostic). Every profile queries all other
  same-side profiles; sides are never combined. A query counts only when
  another photo of the same elephant's same ear exists. ``top-k`` is the
  fraction of queries whose first correct photo ranks in the top k.
- Elephant-level leave-one-sighting-out (the official metric). Every
  sighting queries all other sightings, grouped and ranked by elephant.
  Per gallery elephant and side, the score is the best calibrated score over
  all matching-side photo pairs; "combined" averages the available left and
  right side scores after those per-side maxima have been selected, and
  "combined both-sides" restricts to queries that have both ears.
  Calibrators are fitted on identity-disjoint folds so no query is ever
  scored by a calibrator trained on its own elephant.
"""

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from scipy.signal import find_peaks

from elephant_id.coding.photo_analyzer import PhotoAnalyzer
from elephant_id.constants import TEAR_PROFILE_BINS
from elephant_id.dataset import Dataset
from elephant_id.log import configure_logging
from elephant_id.matching import (
    TearMatcher,
    TearScoreCalibrator,
    symmetrized_cohort_z,
    tear_mass,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = REPO_ROOT / "outputs" / "high_quality" / "manifest.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "tear_matching_eval"
PROFILE_CACHE = OUTPUT_DIR / "hq_profiles.npz"
RESULTS_CSV = OUTPUT_DIR / "hq_results.csv"
SUMMARY_TXT = OUTPUT_DIR / "hq_summary.txt"

CALIBRATION_FOLDS = 2
NEGATIVES_PER_POSITIVE = 5
TAXONOMY_PEAK_HEIGHT = 0.015
TAXONOMY_PEAK_PROMINENCE = 0.01
TAXONOMY_MAX_ALIGNED_OFFSET_DEGREES = 27.0

RANKING_SIDES = {
    "left": ("left",),
    "right": ("right",),
    "combined": ("left", "right"),
}
REPORTED_MODES = (*RANKING_SIDES, "combined both-sides")


@dataclass(frozen=True)
class ProfileSet:
    """Tear profiles for every usable high-quality manifest row."""

    profiles: np.ndarray
    photo_ids: np.ndarray
    identities: np.ndarray
    sides: np.ndarray
    dates: np.ndarray
    skipped_count: int


@dataclass(frozen=True, eq=False)
class Sighting:
    """One elephant sighting: the profile rows shot on one date."""

    identity: str
    date: str
    rows: np.ndarray


@dataclass(frozen=True)
class QueryOutcome:
    """Photo-level retrieval outcome for one query row."""

    query_index: int
    positive_rank: int
    best_positive_index: int
    top_prediction_index: int
    top_prediction_score: float


def main() -> None:
    """Run the high-quality evaluation and write metrics."""
    load_dotenv()
    configure_logging()
    args = parse_args()

    profile_set = load_or_build_profiles(rebuild=args.rebuild_cache)
    matcher = TearMatcher()
    pair_scores = symmetrized_cohort_z(pairwise_scores(profile_set, matcher))

    outcomes = photo_level_evaluation(profile_set, pair_scores)
    taxonomy = failure_taxonomy(profile_set, outcomes)
    elephant_metrics = elephant_level_evaluation(profile_set, pair_scores, seed=args.seed)

    write_results(profile_set, outcomes)
    write_summary(profile_set, outcomes, taxonomy, elephant_metrics)


def parse_args() -> argparse.Namespace:
    """Parse the evaluation knobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def load_or_build_profiles(*, rebuild: bool = False) -> ProfileSet:
    """Load the profile cache, rebuilding when missing or requested."""
    if PROFILE_CACHE.exists() and not rebuild:
        cached = np.load(PROFILE_CACHE, allow_pickle=False)
        logger.info(f"Loading cached profiles from {PROFILE_CACHE}")
        return ProfileSet(
            profiles=cached["profiles"],
            photo_ids=cached["photo_ids"],
            identities=cached["identities"],
            sides=cached["sides"],
            dates=cached["dates"],
            skipped_count=int(cached["skipped_count"]),
        )
    return build_profiles()


def build_profiles() -> ProfileSet:
    """Analyze every high-quality manifest row into the profile cache."""
    manifest_rows = read_manifest()
    logger.info(f"Building profiles for {len(manifest_rows)} manifest rows")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(
        dataset_root=REPO_ROOT / "dataset" / "elephants-alive" / "coded",
        metadata_path=REPO_ROOT / "dataset" / "elephants-alive" / "images.csv",
    )
    analyzer = PhotoAnalyzer(dataset=dataset)
    analysis_cache: dict[str, dict | None] = {}

    profiles: list[np.ndarray] = []
    kept_rows: list[dict[str, str]] = []
    skipped_count = 0
    for row in manifest_rows:
        profile = profile_for(row, dataset, analyzer, analysis_cache)
        if profile is None:
            skipped_count += 1
            continue
        profiles.append(profile)
        kept_rows.append(row)

    if not profiles:
        raise RuntimeError("No profiles extracted; is the model cache warm?")

    profile_set = ProfileSet(
        profiles=np.vstack(profiles),
        photo_ids=np.asarray([row["photo_identifier"] for row in kept_rows]),
        identities=np.asarray([row["identity"] for row in kept_rows]),
        sides=np.asarray([row["side"] for row in kept_rows]),
        dates=np.asarray([row["sighting_date"] for row in kept_rows]),
        skipped_count=skipped_count,
    )
    np.savez_compressed(
        PROFILE_CACHE,
        profiles=profile_set.profiles,
        photo_ids=profile_set.photo_ids,
        identities=profile_set.identities,
        sides=profile_set.sides,
        dates=profile_set.dates,
        skipped_count=np.asarray(skipped_count),
    )
    logger.info(
        f"Cached {len(profile_set.profiles)} profiles "
        f"({skipped_count} skipped) at {PROFILE_CACHE}"
    )
    return profile_set


def read_manifest() -> list[dict[str, str]]:
    """Read the high-quality manifest rows.

    Raises:
        RuntimeError: If the manifest is missing or empty.
    """
    if not MANIFEST_CSV.exists():
        raise RuntimeError(f"High-quality manifest not found: {MANIFEST_CSV}")
    with MANIFEST_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"High-quality manifest is empty: {MANIFEST_CSV}")
    return rows


def profile_for(
    manifest_row: dict[str, str],
    dataset: Dataset,
    analyzer: PhotoAnalyzer,
    analysis_cache: dict[str, dict | None],
) -> np.ndarray | None:
    """Return the requested ear profile for a manifest row, or None if unusable."""
    photo_id = manifest_row["photo_identifier"]
    if photo_id not in analysis_cache:
        try:
            analysis_cache[photo_id] = analyzer.analyze(dataset.get_photo(photo_id))
        except Exception as error:
            logger.warning(f"Analysis failed for {photo_id}: {error}")
            analysis_cache[photo_id] = None

    analysis = analysis_cache[photo_id]
    if analysis is None:
        return None
    for ear_data in analysis["ears"]:
        if ear_data["ear"].side == manifest_row["side"]:
            profile = np.asarray(ear_data["tear_profile"].profile, dtype=np.float64)
            if profile.shape == (TEAR_PROFILE_BINS,):
                return profile
    return None


def pairwise_scores(profile_set: ProfileSet, matcher: TearMatcher) -> np.ndarray:
    """Score every same-side profile pair; other entries stay NaN."""
    row_count = len(profile_set.profiles)
    scores = np.full((row_count, row_count), np.nan)
    for side in ("left", "right"):
        rows = np.flatnonzero(profile_set.sides == side)
        for query_index in rows:
            gallery = rows[rows != query_index]
            gallery_result = matcher.match_gallery(
                profile_set.profiles[query_index],
                profile_set.profiles[gallery],
            )
            scores[query_index, gallery] = gallery_result.score
    logger.info(f"Scored all same-side pairs for {row_count} profiles")
    return scores


def photo_level_evaluation(
    profile_set: ProfileSet,
    pair_scores: np.ndarray,
) -> list[QueryOutcome]:
    """Rank same-side rows for every query with a same-identity partner."""
    row_count = len(profile_set.profiles)
    group_counts = Counter(
        zip(profile_set.identities, profile_set.sides, strict=True)
    )
    outcomes: list[QueryOutcome] = []
    for query_index in range(row_count):
        key = (profile_set.identities[query_index], profile_set.sides[query_index])
        if group_counts[key] < 2:
            continue
        gallery = np.flatnonzero(
            (profile_set.sides == profile_set.sides[query_index])
            & (np.arange(row_count) != query_index)
        )
        order = gallery[np.argsort(pair_scores[query_index, gallery])[::-1]]
        positive_positions = np.flatnonzero(
            profile_set.identities[order] == profile_set.identities[query_index]
        )
        positive_position = int(positive_positions[0])
        outcomes.append(
            QueryOutcome(
                query_index=query_index,
                positive_rank=positive_position + 1,
                best_positive_index=int(order[positive_position]),
                top_prediction_index=int(order[0]),
                top_prediction_score=float(pair_scores[query_index, order[0]]),
            )
        )
    return outcomes


def retrieval_metrics(ranks: np.ndarray) -> dict[str, float]:
    """Summarize retrieval quality over positive ranks."""
    ranks = np.asarray(ranks, dtype=np.float64)
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top3": float(np.mean(ranks <= 3)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "count": float(len(ranks)),
    }


def failure_taxonomy(
    profile_set: ProfileSet,
    outcomes: list[QueryOutcome],
) -> Counter:
    """Classify failed queries by profile-signal relationship to the positive."""
    taxonomy: Counter = Counter()
    degrees_per_bin = 180.0 / profile_set.profiles.shape[1]
    for outcome in outcomes:
        if outcome.positive_rank == 1:
            continue
        query_peaks, query_heights = profile_peaks(
            profile_set.profiles[outcome.query_index]
        )
        positive_peaks, positive_heights = profile_peaks(
            profile_set.profiles[outcome.best_positive_index]
        )
        if len(query_peaks) == 0 and len(positive_peaks) == 0:
            taxonomy["both low-signal"] += 1
        elif len(query_peaks) == 0 or len(positive_peaks) == 0:
            taxonomy["one-sided signal"] += 1
        else:
            offset_degrees = degrees_per_bin * abs(
                int(query_peaks[np.argmax(query_heights)])
                - int(positive_peaks[np.argmax(positive_heights)])
            )
            if offset_degrees > TAXONOMY_MAX_ALIGNED_OFFSET_DEGREES:
                taxonomy["peaks misaligned"] += 1
            else:
                taxonomy["aligned but outscored"] += 1
    return taxonomy


def profile_peaks(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return major peak indices and heights for taxonomy classification."""
    clipped = np.maximum(profile, 0.0)
    indices, properties = find_peaks(
        clipped,
        height=TAXONOMY_PEAK_HEIGHT,
        prominence=TAXONOMY_PEAK_PROMINENCE,
    )
    return indices, properties.get("peak_heights", np.zeros(0))


def elephant_level_evaluation(
    profile_set: ProfileSet,
    pair_scores: np.ndarray,
    *,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Evaluate elephant-level retrieval: withhold one sighting, rank elephants."""
    masses = tear_mass(profile_set.profiles)
    calibrators = identity_calibrators(profile_set, pair_scores, masses, seed=seed)
    sightings = group_sightings(profile_set)
    sightings_per_identity = Counter(sighting.identity for sighting in sightings)

    ranks: dict[str, list[int]] = {mode: [] for mode in REPORTED_MODES}
    for query in sightings:
        if sightings_per_identity[query.identity] < 2:
            continue
        elephant_side_scores = best_side_scores_per_elephant(
            profile_set, pair_scores, masses, calibrators[query.identity], query, sightings
        )
        query_sides = set(profile_set.sides[query.rows])
        for mode, sides in RANKING_SIDES.items():
            rank = elephant_rank(elephant_side_scores, query.identity, sides)
            if rank is None:
                continue
            ranks[mode].append(rank)
            if mode == "combined" and query_sides == {"left", "right"}:
                ranks["combined both-sides"].append(rank)

    return {mode: retrieval_metrics(np.asarray(mode_ranks)) for mode, mode_ranks in ranks.items()}


def group_sightings(profile_set: ProfileSet) -> list[Sighting]:
    """Group profile rows into sightings by (identity, date)."""
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    pairs = zip(profile_set.identities, profile_set.dates, strict=True)
    for row_index, key in enumerate(pairs):
        grouped[key].append(row_index)
    return [
        Sighting(identity=identity, date=date, rows=np.asarray(rows))
        for (identity, date), rows in sorted(grouped.items())
    ]


def identity_calibrators(
    profile_set: ProfileSet,
    pair_scores: np.ndarray,
    masses: np.ndarray,
    *,
    seed: int,
) -> dict[str, TearScoreCalibrator]:
    """Map each identity to a calibrator fitted without that identity.

    Identities are shuffled into ``CALIBRATION_FOLDS`` folds; each fold's
    calibrator is trained on the other folds only.
    """
    identities = np.unique(profile_set.identities)
    rng = np.random.default_rng(seed)
    rng.shuffle(identities)
    fold_of_identity = {
        identity: index % CALIBRATION_FOLDS for index, identity in enumerate(identities)
    }
    calibrator_of_fold = {
        fold: fit_fold_calibrator(profile_set, pair_scores, masses, fold_of_identity, fold)
        for fold in range(CALIBRATION_FOLDS)
    }
    return {identity: calibrator_of_fold[fold] for identity, fold in fold_of_identity.items()}


def fit_fold_calibrator(
    profile_set: ProfileSet,
    pair_scores: np.ndarray,
    masses: np.ndarray,
    fold_of_identity: dict[str, int],
    fold: int,
) -> TearScoreCalibrator:
    """Fit one calibrator on same-side pairs whose identities are outside the fold.

    Every genuine pair is kept; impostor pairs are subsampled per query to
    roughly ``NEGATIVES_PER_POSITIVE`` per genuine pair.
    """
    train_rows = np.flatnonzero(
        [fold_of_identity[identity] != fold for identity in profile_set.identities]
    )
    rng = np.random.default_rng(fold)
    query_groups: list[np.ndarray] = []
    candidate_groups: list[np.ndarray] = []
    for query_index in train_rows:
        candidates = train_rows[
            (profile_set.sides[train_rows] == profile_set.sides[query_index])
            & (train_rows != query_index)
        ]
        genuine = profile_set.identities[candidates] == profile_set.identities[query_index]
        positives = candidates[genuine]
        negatives = rng.choice(
            candidates[~genuine],
            size=min(int((~genuine).sum()), NEGATIVES_PER_POSITIVE * max(len(positives), 1)),
            replace=False,
        )
        chosen = np.concatenate([positives, negatives])
        query_groups.append(np.full(len(chosen), query_index))
        candidate_groups.append(chosen)

    queries = np.concatenate(query_groups)
    candidates = np.concatenate(candidate_groups)
    calibrator = TearScoreCalibrator()
    calibrator.fit(
        pair_scores[queries, candidates],
        masses[queries],
        masses[candidates],
        profile_set.identities[queries] == profile_set.identities[candidates],
    )
    return calibrator


def best_side_scores_per_elephant(
    profile_set: ProfileSet,
    pair_scores: np.ndarray,
    masses: np.ndarray,
    calibrator: TearScoreCalibrator,
    query: Sighting,
    sightings: list[Sighting],
) -> dict[str, dict[str, float]]:
    """Return each gallery elephant's best calibrated score per ear side."""
    best: dict[str, dict[str, float]] = defaultdict(dict)
    for gallery in sightings:
        if gallery is query:
            continue
        side_scores = best[gallery.identity]
        for side, score in sighting_pair_scores(
            profile_set, pair_scores, masses, calibrator, query, gallery
        ).items():
            side_scores[side] = max(side_scores.get(side, -np.inf), score)
    return best


def sighting_pair_scores(
    profile_set: ProfileSet,
    pair_scores: np.ndarray,
    masses: np.ndarray,
    calibrator: TearScoreCalibrator,
    query: Sighting,
    gallery: Sighting,
) -> dict[str, float]:
    """Return the best calibrated score between two sightings, per shared side."""
    scores_by_side: dict[str, float] = {}
    for side in ("left", "right"):
        query_rows = query.rows[profile_set.sides[query.rows] == side]
        gallery_rows = gallery.rows[profile_set.sides[gallery.rows] == side]
        if len(query_rows) == 0 or len(gallery_rows) == 0:
            continue
        pair_scores_for_side = calibrator.calibrated_score(
            pair_scores[np.ix_(query_rows, gallery_rows)].ravel(),
            np.repeat(masses[query_rows], len(gallery_rows)),
            np.tile(masses[gallery_rows], len(query_rows)),
        )
        scores_by_side[side] = float(pair_scores_for_side.max())
    return scores_by_side


def elephant_rank(
    elephant_side_scores: dict[str, dict[str, float]],
    query_identity: str,
    sides: tuple[str, ...],
) -> int | None:
    """Rank the query elephant among gallery elephants scored on the given sides.

    Each elephant first keeps its best score per requested side, then averages
    the side scores it has. Returns None when the query elephant has no score
    on any requested side.
    """
    elephant_scores = {
        elephant: float(np.mean([side_scores[side] for side in sides if side in side_scores]))
        for elephant, side_scores in elephant_side_scores.items()
        if any(side in side_scores for side in sides)
    }
    if query_identity not in elephant_scores:
        return None
    query_score = elephant_scores[query_identity]
    return 1 + sum(score > query_score for score in elephant_scores.values())


def write_results(profile_set: ProfileSet, outcomes: list[QueryOutcome]) -> None:
    """Write one CSV row per photo-level query."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "identity",
        "side",
        "positive_rank",
        "best_positive_id",
        "top_prediction_id",
        "top_prediction_identity",
        "top_prediction_score",
    ]
    with RESULTS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for outcome in outcomes:
            writer.writerow(
                {
                    "query_id": profile_set.photo_ids[outcome.query_index],
                    "identity": profile_set.identities[outcome.query_index],
                    "side": profile_set.sides[outcome.query_index],
                    "positive_rank": outcome.positive_rank,
                    "best_positive_id": profile_set.photo_ids[outcome.best_positive_index],
                    "top_prediction_id": profile_set.photo_ids[outcome.top_prediction_index],
                    "top_prediction_identity": profile_set.identities[
                        outcome.top_prediction_index
                    ],
                    "top_prediction_score": f"{outcome.top_prediction_score:.6f}",
                }
            )
    logger.info(f"Wrote {len(outcomes)} query rows to {RESULTS_CSV}")


def write_summary(
    profile_set: ProfileSet,
    outcomes: list[QueryOutcome],
    taxonomy: Counter,
    elephant_metrics: dict[str, dict[str, float]],
) -> None:
    """Write and log the metric summary."""
    photo_metrics = retrieval_metrics(
        np.asarray([outcome.positive_rank for outcome in outcomes])
    )
    lines = [
        "High-quality tear matching evaluation",
        f"profiles: {len(profile_set.profiles)} ({profile_set.skipped_count} skipped)",
        "scores: symmetrized cohort z-normalized",
        "",
        "photo-level leave-one-out (same side):",
        format_metrics(photo_metrics),
        "",
        "failure taxonomy:",
    ]
    lines.extend(
        f"  {name}: {count}" for name, count in taxonomy.most_common()
    )
    lines.append("")
    lines.append("elephant-level calibrated retrieval (leave one sighting out):")
    for mode in REPORTED_MODES:
        lines.append(f"  {mode}: {format_metrics(elephant_metrics[mode])}")
    SUMMARY_TXT.write_text("\n".join(lines) + "\n")
    for line in lines:
        if line:
            logger.info(line)


def format_metrics(metrics: dict[str, float]) -> str:
    """Format one retrieval metric row."""
    return (
        f"n={metrics['count']:.0f} top1={metrics['top1']:.3f} "
        f"top3={metrics['top3']:.3f} top5={metrics['top5']:.3f} "
        f"MRR={metrics['mrr']:.3f} median_rank={metrics['median_rank']:.1f}"
    )


if __name__ == "__main__":
    main()
