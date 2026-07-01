"""Evaluate tear-profile retrieval on the filtered good-ear set."""

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from elephant_id.coding.ears.anchored_ear import AnchoredEar
from elephant_id.coding.ears.tear_profile import TearProfile
from elephant_id.coding.photo_analyzer import PhotoAnalyzer
from elephant_id.constants import TEAR_PROFILE_BINS
from elephant_id.dataset import Dataset
from elephant_id.image import BgrImage
from elephant_id.log import configure_logging
from elephant_id.matching import TearMatcher, TearMatchGallery
from elephant_id.visualize import (
    align_tear_profile_for_plot,
    plot_aligned_tear_profiles,
    plot_tear_profile_geometry,
    tear_profile_ymax,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FILTERED_DIR = REPO_ROOT / "outputs" / "ear_segmentation_filtered"
OUTPUT_DIR = REPO_ROOT / "outputs" / "tear_matching_eval"
PROFILE_CACHE = OUTPUT_DIR / "filtered_profiles.npz"
RESULTS_CSV = OUTPUT_DIR / "results.csv"
SUMMARY_TXT = OUTPUT_DIR / "summary.txt"
REVIEW_DIR = OUTPUT_DIR / "review_cases"

DEFAULT_GALLERY_SIZE = 500
DEFAULT_QUERY_SIZE = 200
DEFAULT_SUCCESS_CASES = 20
DEFAULT_FAILURE_CASES = 20
DEFAULT_SEED = 0


@dataclass(frozen=True)
class FilteredEar:
    """One filtered ear image mapped back to a source photo and side."""

    filename: str
    photo_id: str
    elephant: str
    side: Literal["left", "right"]
    size_bytes: int
    modified_ns: int


@dataclass(frozen=True)
class ProfileSet:
    """All usable profiles extracted from the filtered ear directory."""

    profiles: np.ndarray
    elephants: np.ndarray
    sides: np.ndarray
    photo_ids: np.ndarray
    tear_mass: np.ndarray
    profile_files: np.ndarray
    filtered_fingerprint: np.ndarray
    skipped_count: int


@dataclass(frozen=True)
class EvaluationSplit:
    """Disjoint gallery and query row indices."""

    gallery: np.ndarray
    queries: np.ndarray


@dataclass(frozen=True)
class GalleryCandidate:
    """One ranked gallery candidate with alignment diagnostics."""

    index: int
    score: float
    overlap_score: float
    shift_bins: int
    shift_fraction: float
    stretch: float
    penalty: float


@dataclass(frozen=True)
class QueryResult:
    """Retrieval outcome for one query profile."""

    query_index: int
    positive_rank: int
    positive: GalleryCandidate
    negative: GalleryCandidate
    top_prediction: GalleryCandidate


@dataclass(frozen=True)
class EarEvidence:
    """Source evidence needed for one visual review panel."""

    identifier: str
    label: str
    image: BgrImage
    ear: AnchoredEar
    tear_profile: TearProfile


@dataclass(frozen=True)
class ReviewCandidate:
    """Visual wrapper for a positive or negative candidate."""

    evidence: EarEvidence
    result: GalleryCandidate
    color: str


def main() -> None:
    """Run the filtered-set evaluation and write metrics plus review plates."""
    load_dotenv()
    configure_logging()
    args = parse_args()

    profile_set = load_or_build_profiles(rebuild=args.rebuild_cache)
    split = choose_split(
        profile_set,
        gallery_size=args.gallery_size,
        query_size=args.query_size,
        seed=args.seed,
    )
    results = evaluate_queries(profile_set, split, TearMatcher())
    write_results(profile_set, split, results)
    write_summary(profile_set, split, results)

    if not args.skip_visuals:
        render_review_cases(
            profile_set,
            results,
            success_count=args.success_cases,
            failure_count=args.failure_cases,
            seed=args.seed,
        )


def parse_args() -> argparse.Namespace:
    """Parse the small set of evaluation knobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-size", "--database-size", dest="gallery_size", type=int, default=DEFAULT_GALLERY_SIZE)
    parser.add_argument("--query-size", type=int, default=DEFAULT_QUERY_SIZE)
    parser.add_argument("--success-cases", type=int, default=DEFAULT_SUCCESS_CASES)
    parser.add_argument("--failure-cases", type=int, default=DEFAULT_FAILURE_CASES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-visuals", action="store_true")
    return parser.parse_args()


def load_or_build_profiles(*, rebuild: bool = False) -> ProfileSet:
    """Load the profile cache, rebuilding when the filtered directory changes."""
    filtered_ears = list_filtered_ears()
    filtered_fingerprint = filtered_directory_fingerprint(filtered_ears)
    if PROFILE_CACHE.exists() and not rebuild:
        cached = np.load(PROFILE_CACHE, allow_pickle=False)
        cache_keys = set(cached.files)
        if cache_matches_filtered_directory(cached, cache_keys, filtered_fingerprint):
            logger.info(f"Loading cached profiles from {PROFILE_CACHE}")
            return ProfileSet(
                profiles=cached["profiles"],
                elephants=cached["elephants"],
                sides=cached["sides"],
                photo_ids=cached["photo_ids"],
                tear_mass=cached["tear_mass"],
                profile_files=cached["profile_files"],
                filtered_fingerprint=filtered_fingerprint,
                skipped_count=int(cached["skipped_count"]),
            )
        logger.info("Filtered ear directory changed; rebuilding profile cache")
    elif rebuild:
        logger.info("Rebuilding profile cache because --rebuild-cache was passed")

    return build_profiles(filtered_ears, filtered_fingerprint)


def cache_matches_filtered_directory(
    cached: np.lib.npyio.NpzFile,
    cache_keys: set[str],
    filtered_fingerprint: np.ndarray,
) -> bool:
    """Return whether a cached profile set matches the current filtered ears."""
    if "filtered_fingerprint" in cache_keys:
        return bool(np.array_equal(cached["filtered_fingerprint"], filtered_fingerprint))
    return False


def list_filtered_ears() -> list[FilteredEar]:
    """Read filtered ear filenames into source-photo rows."""
    ears: list[FilteredEar] = []
    for path in sorted(FILTERED_DIR.glob("*.jpg")):
        photo_id, _, side = path.stem.rpartition("_")
        if side not in {"left", "right"}:
            continue
        stat = path.stat()
        ears.append(
            FilteredEar(
                filename=path.name,
                photo_id=photo_id,
                elephant=photo_id.rsplit("_", 2)[0],
                side=side,
                size_bytes=stat.st_size,
                modified_ns=stat.st_mtime_ns,
            )
        )
    if not ears:
        raise RuntimeError(f"No filtered ear images found in {FILTERED_DIR}")
    return ears


def filtered_directory_fingerprint(filtered_ears: list[FilteredEar]) -> np.ndarray:
    """Return cache metadata for detecting filtered-directory changes."""
    return np.asarray(
        [
            f"{ear.filename}\t{ear.size_bytes}\t{ear.modified_ns}"
            for ear in filtered_ears
        ]
    )


def build_profiles(
    filtered_ears: list[FilteredEar],
    filtered_fingerprint: np.ndarray,
) -> ProfileSet:
    """Analyze every filtered ear into the reusable profile cache."""
    logger.info(f"Building profile cache from {len(filtered_ears)} filtered ears")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset()
    analyzer = PhotoAnalyzer(dataset=dataset)
    analysis_cache: dict[str, dict | None] = {}

    profiles: list[np.ndarray] = []
    elephants: list[str] = []
    sides: list[str] = []
    photo_ids: list[str] = []
    profile_files: list[str] = []
    skipped_count = 0

    for ear in filtered_ears:
        profile = profile_for(ear, dataset, analyzer, analysis_cache)
        if profile is None:
            skipped_count += 1
            continue
        profiles.append(profile.astype(np.float64))
        elephants.append(ear.elephant)
        sides.append(ear.side)
        photo_ids.append(ear.photo_id)
        profile_files.append(ear.filename)

    if not profiles:
        raise RuntimeError("No profiles extracted; is the model cache warm?")

    profile_set = ProfileSet(
        profiles=np.vstack(profiles),
        elephants=np.asarray(elephants),
        sides=np.asarray(sides),
        photo_ids=np.asarray(photo_ids),
        tear_mass=np.asarray([np.maximum(profile, 0.0).sum() for profile in profiles]),
        profile_files=np.asarray(profile_files),
        filtered_fingerprint=filtered_fingerprint,
        skipped_count=skipped_count,
    )
    np.savez_compressed(
        PROFILE_CACHE,
        profiles=profile_set.profiles,
        elephants=profile_set.elephants,
        sides=profile_set.sides,
        photo_ids=profile_set.photo_ids,
        tear_mass=profile_set.tear_mass,
        profile_files=profile_set.profile_files,
        filtered_fingerprint=profile_set.filtered_fingerprint,
        skipped_count=np.asarray(skipped_count),
    )
    logger.info(
        f"Cached {len(profile_set.profiles)} profiles "
        f"({skipped_count} skipped) at {PROFILE_CACHE}"
    )
    return profile_set


def make_dataset() -> Dataset:
    """Create the source dataset used by the filtered ear list."""
    return Dataset(
        dataset_root=REPO_ROOT / "dataset" / "elephants-alive" / "coded",
        metadata_path=REPO_ROOT / "dataset" / "elephants-alive" / "images.csv",
    )


def profile_for(
    filtered_ear: FilteredEar,
    dataset: Dataset,
    analyzer: PhotoAnalyzer,
    analysis_cache: dict[str, dict | None],
) -> np.ndarray | None:
    """Return the requested ear profile for a filtered ear, or None if unusable."""
    if filtered_ear.photo_id not in analysis_cache:
        try:
            photo = dataset.get_photo(filtered_ear.photo_id)
            analysis_cache[filtered_ear.photo_id] = analyzer.analyze(photo)
        except Exception as error:
            logger.warning(f"Analysis failed for {filtered_ear.photo_id}: {error}")
            analysis_cache[filtered_ear.photo_id] = None

    analysis = analysis_cache[filtered_ear.photo_id]
    if analysis is None:
        return None

    for ear_data in analysis["ears"]:
        if ear_data["ear"].side == filtered_ear.side:
            profile = np.asarray(ear_data["tear_profile"].profile, dtype=np.float64)
            if profile.shape == (TEAR_PROFILE_BINS,):
                return profile
    return None


def choose_split(
    profile_set: ProfileSet,
    *,
    gallery_size: int,
    query_size: int,
    seed: int,
) -> EvaluationSplit:
    """Choose disjoint query/gallery rows with a same-side positive per query."""
    rng = np.random.default_rng(seed)
    groups = group_by_identity_and_side(profile_set)
    eligible_groups = [indices.copy() for indices in groups.values() if len(indices) >= 2]
    if not eligible_groups:
        raise RuntimeError("No identity+side groups have at least two profiles")

    for indices in eligible_groups:
        rng.shuffle(indices)
    rng.shuffle(eligible_groups)

    query_indices: list[int] = []
    max_queries_per_group = max(len(indices) - 1 for indices in eligible_groups)
    for offset in range(max_queries_per_group):
        for indices in eligible_groups:
            if offset < len(indices) - 1:
                query_indices.append(int(indices[offset]))
                if len(query_indices) == query_size:
                    break
        if len(query_indices) == query_size:
            break
    if len(query_indices) < query_size:
        raise RuntimeError(
            f"Only {len(query_indices)} eligible queries are available; "
            f"requested {query_size}"
        )

    # Reserve one positive gallery row for every queried identity+side, then
    # fill the remaining gallery slots with non-query rows.
    query_set = set(query_indices)
    query_identity_sides = {
        (str(profile_set.elephants[index]), str(profile_set.sides[index]))
        for index in query_indices
    }
    required_positive_rows = sorted(
        positive_rows_for_queries(
            groups,
            query_identity_sides,
            query_set,
        )
    )
    required_positive_set = set(required_positive_rows)
    gallery_candidates = [
        index for index in range(len(profile_set.profiles)) if index not in query_set
    ]
    if len(gallery_candidates) < gallery_size:
        raise RuntimeError(
            f"Only {len(gallery_candidates)} gallery rows are available; "
            f"requested {gallery_size}"
        )
    if len(required_positive_rows) > gallery_size:
        raise RuntimeError(
            f"{len(required_positive_rows)} required positive rows exceed gallery size "
            f"{gallery_size}"
        )

    remaining = [index for index in gallery_candidates if index not in required_positive_set]
    rng.shuffle(remaining)
    gallery_indices = required_positive_rows
    gallery_indices.extend(remaining[: gallery_size - len(gallery_indices)])
    rng.shuffle(gallery_indices)

    return EvaluationSplit(
        gallery=np.asarray(gallery_indices, dtype=np.int32),
        queries=np.asarray(query_indices, dtype=np.int32),
    )


def positive_rows_for_queries(
    groups: dict[tuple[str, str], np.ndarray],
    query_identity_sides: set[tuple[str, str]],
    query_set: set[int],
) -> set[int]:
    """Choose one gallery positive row for every queried identity+side."""
    return {
        int(next(index for index in groups[key] if index not in query_set))
        for key in query_identity_sides
    }


def group_by_identity_and_side(profile_set: ProfileSet) -> dict[tuple[str, str], np.ndarray]:
    """Group row indices by identity and ear side."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (elephant, side) in enumerate(zip(profile_set.elephants, profile_set.sides, strict=True)):
        groups[(str(elephant), str(side))].append(index)
    return {
        key: np.asarray(indices, dtype=np.int32)
        for key, indices in groups.items()
    }


def evaluate_queries(
    profile_set: ProfileSet,
    split: EvaluationSplit,
    matcher: TearMatcher,
) -> list[QueryResult]:
    """Rank same-side gallery rows for every query."""
    results: list[QueryResult] = []
    for query_index in split.queries:
        side = profile_set.sides[query_index]
        same_side_gallery = split.gallery[profile_set.sides[split.gallery] == side]
        if len(same_side_gallery) == 0:
            raise RuntimeError(f"No {side} gallery rows for query {query_index}")

        gallery = matcher.match_gallery(
            profile_set.profiles[query_index],
            profile_set.profiles[same_side_gallery],
        )
        ordered_local = gallery.order
        ordered_global = same_side_gallery[ordered_local]
        positive_positions = np.flatnonzero(
            profile_set.elephants[ordered_global] == profile_set.elephants[query_index]
        )
        negative_positions = np.flatnonzero(
            profile_set.elephants[ordered_global] != profile_set.elephants[query_index]
        )
        if len(positive_positions) == 0:
            raise RuntimeError(f"No positive row found for query {query_index}")
        if len(negative_positions) == 0:
            raise RuntimeError(f"No negative row found for query {query_index}")

        positive_position = int(positive_positions[0])
        negative_position = int(negative_positions[0])
        results.append(
            QueryResult(
                query_index=int(query_index),
                positive_rank=positive_position + 1,
                positive=candidate_from_gallery(
                    gallery,
                    int(ordered_local[positive_position]),
                    int(ordered_global[positive_position]),
                ),
                negative=candidate_from_gallery(
                    gallery,
                    int(ordered_local[negative_position]),
                    int(ordered_global[negative_position]),
                ),
                top_prediction=candidate_from_gallery(
                    gallery,
                    int(ordered_local[0]),
                    int(ordered_global[0]),
                ),
            )
        )
    return results


def candidate_from_gallery(
    gallery: TearMatchGallery,
    local_index: int,
    global_index: int,
) -> GalleryCandidate:
    """Extract one candidate result from a gallery match."""
    return GalleryCandidate(
        index=global_index,
        score=float(gallery.score[local_index]),
        overlap_score=float(gallery.overlap_score[local_index]),
        shift_bins=int(gallery.shift_bins[local_index]),
        shift_fraction=float(gallery.shift_fraction[local_index]),
        stretch=float(gallery.stretch[local_index]),
        penalty=float(gallery.penalty[local_index]),
    )


def metrics_for(results: list[QueryResult]) -> dict[str, float]:
    """Summarize retrieval quality over query results."""
    ranks = np.asarray([result.positive_rank for result in results], dtype=np.float64)
    return {
        "top1": float(np.mean(ranks <= 1)),
        "top3": float(np.mean(ranks <= 3)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_positive_rank": float(np.median(ranks)),
    }


def write_results(
    profile_set: ProfileSet,
    split: EvaluationSplit,
    results: list[QueryResult],
) -> None:
    """Write one CSV row per evaluated query."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "elephant",
        "side",
        "positive_rank",
        "best_positive_id",
        "positive_score",
        "positive_overlap_score",
        "positive_shift_bins",
        "positive_shift_fraction",
        "positive_stretch",
        "positive_penalty",
        "best_negative_id",
        "negative_elephant",
        "negative_score",
        "negative_overlap_score",
        "negative_shift_bins",
        "negative_shift_fraction",
        "negative_stretch",
        "negative_penalty",
        "top_prediction_id",
        "top_prediction_elephant",
        "top_prediction_score",
        "gallery_size",
        "same_side_gallery_size",
    ]
    with RESULTS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            query_index = result.query_index
            side = profile_set.sides[query_index]
            same_side_count = int(np.count_nonzero(profile_set.sides[split.gallery] == side))
            writer.writerow(
                {
                    "query_id": profile_set.photo_ids[query_index],
                    "elephant": profile_set.elephants[query_index],
                    "side": side,
                    "positive_rank": result.positive_rank,
                    "best_positive_id": profile_set.photo_ids[result.positive.index],
                    "positive_score": f"{result.positive.score:.6f}",
                    "positive_overlap_score": f"{result.positive.overlap_score:.6f}",
                    "positive_shift_bins": result.positive.shift_bins,
                    "positive_shift_fraction": f"{result.positive.shift_fraction:.6f}",
                    "positive_stretch": f"{result.positive.stretch:.6f}",
                    "positive_penalty": f"{result.positive.penalty:.6f}",
                    "best_negative_id": profile_set.photo_ids[result.negative.index],
                    "negative_elephant": profile_set.elephants[result.negative.index],
                    "negative_score": f"{result.negative.score:.6f}",
                    "negative_overlap_score": f"{result.negative.overlap_score:.6f}",
                    "negative_shift_bins": result.negative.shift_bins,
                    "negative_shift_fraction": f"{result.negative.shift_fraction:.6f}",
                    "negative_stretch": f"{result.negative.stretch:.6f}",
                    "negative_penalty": f"{result.negative.penalty:.6f}",
                    "top_prediction_id": profile_set.photo_ids[result.top_prediction.index],
                    "top_prediction_elephant": profile_set.elephants[result.top_prediction.index],
                    "top_prediction_score": f"{result.top_prediction.score:.6f}",
                    "gallery_size": len(split.gallery),
                    "same_side_gallery_size": same_side_count,
                }
            )


def write_summary(
    profile_set: ProfileSet,
    split: EvaluationSplit,
    results: list[QueryResult],
) -> None:
    """Write and log a compact metric summary."""
    metrics = metrics_for(results)
    eligible_count = eligible_query_count(profile_set)
    side_counts = Counter(str(side) for side in profile_set.sides)
    lines = [
        "Tear matching evaluation",
        f"profiles: {len(profile_set.profiles)}",
        f"gallery rows: {len(split.gallery)}",
        f"query rows: {len(split.queries)}",
        f"eligible query rows: {eligible_count}",
        f"skipped filtered rows: {profile_set.skipped_count}",
        f"left rows: {side_counts.get('left', 0)}",
        f"right rows: {side_counts.get('right', 0)}",
        f"top-1: {metrics['top1']:.3f}",
        f"top-3: {metrics['top3']:.3f}",
        f"top-5: {metrics['top5']:.3f}",
        f"MRR: {metrics['mrr']:.3f}",
        f"median positive rank: {metrics['median_positive_rank']:.1f}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n")
    logger.info(" | ".join(lines[1:]))


def eligible_query_count(profile_set: ProfileSet) -> int:
    """Count rows with at least one same-identity same-side partner."""
    groups = group_by_identity_and_side(profile_set)
    return sum(len(indices) for indices in groups.values() if len(indices) >= 2)


def render_review_cases(
    profile_set: ProfileSet,
    results: list[QueryResult],
    *,
    success_count: int,
    failure_count: int,
    seed: int,
) -> None:
    """Render source-backed review plates for sampled successes and failures."""
    logger.info(
        "Rendering review plates by reloading source evidence for selected cases only"
    )
    successes, failures = choose_review_cases(
        results,
        success_count=success_count,
        failure_count=failure_count,
        seed=seed,
    )
    if not successes and not failures:
        logger.warning("No review cases selected")
        return

    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset()
    analyzer = PhotoAnalyzer(dataset=dataset)
    evidence_cache: dict[int, EarEvidence] = {}
    rows: list[dict[str, str]] = []
    y_max = tear_profile_ymax(profile_set.profiles)

    for category, selected in (("success", successes), ("failure", failures)):
        for number, result in enumerate(selected, start=1):
            try:
                query = evidence_for(
                    result.query_index,
                    profile_set,
                    dataset,
                    analyzer,
                    evidence_cache,
                )
                positive = ReviewCandidate(
                    evidence=evidence_for(
                        result.positive.index,
                        profile_set,
                        dataset,
                        analyzer,
                        evidence_cache,
                    ),
                    result=result.positive,
                    color="tab:green",
                )
                negative = ReviewCandidate(
                    evidence=evidence_for(
                        result.negative.index,
                        profile_set,
                        dataset,
                        analyzer,
                        evidence_cache,
                    ),
                    result=result.negative,
                    color="tab:red",
                )
            except RuntimeError as error:
                logger.warning(f"Skipping review plate for query {result.query_index}: {error}")
                continue
            filename = f"{category}_{number:02d}_{safe_filename(query.identifier)}.png"
            output = REVIEW_DIR / filename
            render_comparison(
                output,
                query,
                positive,
                negative,
                title=(
                    f"{category.title()} rank={result.positive_rank} — "
                    f"query {query.label}; positive {positive.evidence.label}; "
                    f"negative {negative.evidence.label}"
                ),
                y_max=y_max,
            )
            rows.append(
                {
                    "category": category,
                    "plate": filename,
                    "query": query.identifier,
                    "positive_rank": str(result.positive_rank),
                    "positive": positive.evidence.identifier,
                    "positive_score": f"{result.positive.score:.6f}",
                    "negative": negative.evidence.identifier,
                    "negative_score": f"{result.negative.score:.6f}",
                }
            )

    write_visual_manifest(rows)


def choose_review_cases(
    results: list[QueryResult],
    *,
    success_count: int,
    failure_count: int,
    seed: int,
) -> tuple[list[QueryResult], list[QueryResult]]:
    """Randomly sample review successes and failures."""
    rng = np.random.default_rng(seed)
    successes = [result for result in results if result.positive_rank == 1]
    failures = [result for result in results if result.positive_rank > 1]
    rng.shuffle(successes)
    rng.shuffle(failures)
    return (
        successes[:success_count],
        failures[:failure_count],
    )


def evidence_for(
    index: int,
    profile_set: ProfileSet,
    dataset: Dataset,
    analyzer: PhotoAnalyzer,
    cache: dict[int, EarEvidence],
) -> EarEvidence:
    """Load source image and analysis evidence for one profile row."""
    if index in cache:
        return cache[index]

    photo = dataset.get_photo(str(profile_set.photo_ids[index]))
    analysis = analyzer.analyze(photo)
    if analysis is None:
        raise RuntimeError(f"Photo analysis returned no result: {photo.identifier}")
    side = str(profile_set.sides[index])
    ear_data = next(
        (item for item in analysis["ears"] if item["ear"].side == side),
        None,
    )
    if ear_data is None:
        raise RuntimeError(f"Photo has no {side} ear: {photo.identifier}")

    evidence = EarEvidence(
        identifier=photo.identifier,
        label=f"{profile_set.elephants[index]} {side}",
        image=dataset.read_image(photo),
        ear=ear_data["ear"],
        tear_profile=ear_data["tear_profile"],
    )
    cache[index] = evidence
    return evidence


def render_comparison(
    output: Path,
    query: EarEvidence,
    positive: ReviewCandidate,
    negative: ReviewCandidate,
    *,
    title: str,
    y_max: float,
) -> None:
    """Render coordinate views above positive and negative overlap graphs."""
    figure = plt.figure(figsize=(18, 9), layout="constrained")
    grid = figure.add_gridspec(2, 6, height_ratios=(1.1, 1.0))
    geometry_axes = [
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
    ]
    positive_axis = figure.add_subplot(grid[1, 0:3])
    negative_axis = figure.add_subplot(grid[1, 3:6])
    top_row = (
        (query, "query", "tab:blue"),
        (positive.evidence, "best positive", positive.color),
        (negative.evidence, "best negative", negative.color),
    )
    for column, (axis, (evidence, label, color)) in enumerate(
        zip(geometry_axes, top_row, strict=True)
    ):
        plot_tear_profile_geometry(axis, evidence.image, evidence.ear, evidence.tear_profile)
        axis.set_title(
            f"{label}\n{evidence.identifier} ({evidence.label})",
            fontsize=8,
            color=color,
        )
        if column == 0:
            axis.legend(fontsize=6, loc="lower right")

    plot_candidate_comparison(
        positive_axis,
        query,
        positive,
        candidate_label="best positive",
        y_max=y_max,
    )
    plot_candidate_comparison(
        negative_axis,
        query,
        negative,
        candidate_label="best negative",
        y_max=y_max,
    )
    figure.suptitle(title, fontsize=13, fontweight="bold")
    figure.savefig(output, dpi=145)
    plt.close(figure)


def plot_candidate_comparison(
    axis: plt.Axes,
    query: EarEvidence,
    candidate: ReviewCandidate,
    *,
    candidate_label: str,
    y_max: float,
) -> None:
    """Plot one candidate profile against the query after scored alignment."""
    aligned_query = align_tear_profile_for_plot(
        query.tear_profile.profile,
        candidate.result.shift_fraction,
        candidate.result.stretch,
    )
    plot_aligned_tear_profiles(
        axis,
        candidate.evidence.tear_profile.profile,
        aligned_query,
        candidate_label=candidate_label,
        color=candidate.color,
        y_max=y_max,
        shift_fraction=candidate.result.shift_fraction,
        shift_bins=candidate.result.shift_bins,
        stretch=candidate.result.stretch,
        penalty=candidate.result.penalty,
        overlap_score=candidate.result.overlap_score,
        score=candidate.result.score,
    )


def write_visual_manifest(rows: list[dict[str, str]]) -> None:
    """Write an index for rendered review plates."""
    if not rows:
        return
    with (REVIEW_DIR / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(value: str) -> str:
    """Return a filesystem-safe stem fragment."""
    return value.replace("/", "_").replace(":", "_")


if __name__ == "__main__":
    main()
