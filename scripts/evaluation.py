"""Evaluate tear-profile retrieval on the high-quality ear image set.

The scoring stack under test (see docs/reference/matching.md): tear profiles ->
``TearMatcher`` pair scores -> ``symmetrized_cohort_z`` normalization ->
tear-mass-conditioned calibration -> side-score averaging.

The input rows come from ``outputs/high_quality/manifest.csv``. Each kept row
must identify one dataset photo, one known elephant, and one ear side. Profile
extraction reads the historical coded dataset, then caches usable profile rows
in ``outputs/tear_matching_eval/hq_profiles.npz``. Raw same-side pairwise
matcher scores are cached separately and strictly tied to that profile cache.

The reported protocol is two-ear retrieval from high-quality, high-resolution
image pairs. Each image pair queries all other image pairs, grouped and ranked
by elephant. Per gallery elephant and side, the score is the best score over
all matching-side image pairs; "combined" averages the left and right side
scores after those per-side maxima have been selected. By default, scores are
cohort-normalized and calibrated; ``--no-normalization`` and
``--no-calibration`` disable exactly those steps for one ablation run.
Calibrators are fitted on identity-disjoint folds so no query is ever scored by
a calibrator trained on its own elephant.
"""

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from loguru import logger

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
PAIRWISE_CACHE = OUTPUT_DIR / "hq_pairwise_scores.npz"

PROFILE_CACHE_VERSION = "high-quality-strict-v2"
PAIRWISE_CACHE_VERSION = "tear-pairwise-strict-v2"
CALIBRATION_FOLDS = 2
NEGATIVES_PER_POSITIVE = 5
MANIFEST_FIELDS = ("photo_identifier", "identity", "side", "sighting_date")
TOP_KS = (1, 3, 5, 10, 15)

RANKING_SIDES = {
    "left": ("left",),
    "right": ("right",),
    "combined": ("left", "right"),
}
REPORTED_MODES = tuple(RANKING_SIDES)


@dataclass(frozen=True)
class ProfileSet:
    """Tear profiles and labels for every usable high-quality manifest row."""

    profiles: np.ndarray
    photo_ids: np.ndarray
    identities: np.ndarray
    sides: np.ndarray
    dates: np.ndarray
    skipped_count: int
    manifest_fingerprint: str


@dataclass(frozen=True, eq=False)
class ImagePair:
    """One paired left/right high-quality image unit for evaluation."""

    identity: str
    pair_id: str
    rows: np.ndarray


def main() -> None:
    """Run the high-quality evaluation and write metrics."""
    load_dotenv()
    configure_logging()
    args = parse_args()

    profile_set = load_or_build_profiles(rebuild=args.rebuild_cache)
    matcher = TearMatcher()
    raw_pair_scores = load_or_compute_pairwise_scores(
        profile_set,
        matcher,
        rebuild=args.rebuild_pairwise_cache,
    )
    score_matrix = evaluation_score_matrix(
        raw_pair_scores,
        normalize=not args.no_normalization,
    )
    calibrators = None
    if not args.no_calibration:
        calibrators = identity_calibrators(
            profile_set,
            score_matrix,
            tear_mass(profile_set.profiles),
            seed=args.calibration_seed,
        )
    seeds = tuple(range(args.seed, args.seed + args.seeds))

    evaluation = evaluate_seeds(
        profile_set,
        score_matrix,
        calibrators,
        seeds=seeds,
    )
    write_summary(
        profile_set,
        evaluation,
        seeds=seeds,
        normalized=not args.no_normalization,
        calibrated=not args.no_calibration,
    )


def parse_args() -> argparse.Namespace:
    """Parse the evaluation knobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="First image-pair seed.")
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of consecutive image-pair seeds to evaluate.",
    )
    parser.add_argument(
        "--calibration-seed",
        type=int,
        default=0,
        help="Seed for identity-disjoint calibration folds and negative sampling.",
    )
    parser.add_argument(
        "--no-normalization",
        action="store_true",
        help="Use symmetrized raw matcher scores instead of cohort-normalized scores.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Rank directly on selected scores instead of calibrated evidence logits.",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--rebuild-pairwise-cache", action="store_true")
    args = parser.parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")
    return args


def load_or_build_profiles(*, rebuild: bool = False) -> ProfileSet:
    """Load the profile cache, rebuilding when missing or requested."""
    manifest_rows = read_manifest()
    fingerprint = manifest_fingerprint(manifest_rows)
    if PROFILE_CACHE.exists() and not rebuild:
        cached = np.load(PROFILE_CACHE, allow_pickle=False)
        validate_profile_cache(cached, manifest_rows, fingerprint)
        logger.info(f"Loading cached profiles from {PROFILE_CACHE}")
        return ProfileSet(
            profiles=cached["profiles"],
            photo_ids=cached["photo_ids"],
            identities=cached["identities"],
            sides=cached["sides"],
            dates=cached["dates"],
            skipped_count=int(cached["skipped_count"]),
            manifest_fingerprint=fingerprint,
        )
    return build_profiles(manifest_rows)


def build_profiles(manifest_rows: list[dict[str, str]] | None = None) -> ProfileSet:
    """Analyze every high-quality manifest row into the profile cache."""
    if manifest_rows is None:
        manifest_rows = read_manifest()
    fingerprint = manifest_fingerprint(manifest_rows)
    logger.info(f"Building profiles for {len(manifest_rows)} manifest rows")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(
        dataset_root=REPO_ROOT / "dataset" / "elephants-alive" / "coded",
        metadata_path=REPO_ROOT / "dataset" / "elephants-alive" / "images.csv",
    )
    analyzer = PhotoAnalyzer(dataset=dataset)
    analysis_cache: dict[str, dict] = {}

    profiles: list[np.ndarray] = []
    kept_rows: list[dict[str, str]] = []
    for row in manifest_rows:
        profiles.append(profile_for(row, dataset, analyzer, analysis_cache))
        kept_rows.append(row)

    if not profiles:
        raise RuntimeError("No profiles extracted; is the model cache warm?")

    profile_set = ProfileSet(
        profiles=np.vstack(profiles),
        photo_ids=np.asarray([row["photo_identifier"] for row in kept_rows]),
        identities=np.asarray([row["identity"] for row in kept_rows]),
        sides=np.asarray([row["side"] for row in kept_rows]),
        dates=np.asarray([row["sighting_date"] for row in kept_rows]),
        skipped_count=0,
        manifest_fingerprint=fingerprint,
    )
    np.savez_compressed(
        PROFILE_CACHE,
        profiles=profile_set.profiles,
        photo_ids=profile_set.photo_ids,
        identities=profile_set.identities,
        sides=profile_set.sides,
        dates=profile_set.dates,
        skipped_count=np.asarray(0),
        manifest_fingerprint=np.asarray(fingerprint),
        manifest_row_count=np.asarray(len(manifest_rows)),
        profile_cache_version=np.asarray(PROFILE_CACHE_VERSION),
        tear_profile_bins=np.asarray(TEAR_PROFILE_BINS),
    )
    logger.info(
        f"Cached {len(profile_set.profiles)} strict profiles at {PROFILE_CACHE}"
    )
    return profile_set


def manifest_fingerprint(rows: list[dict[str, str]]) -> str:
    """Return a stable fingerprint for the matching-relevant manifest fields."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update("\x1f".join(row[field] for field in MANIFEST_FIELDS).encode())
        digest.update(b"\x1e")
    return digest.hexdigest()


def validate_profile_cache(
    cached: np.lib.npyio.NpzFile,
    manifest_rows: list[dict[str, str]],
    expected: str,
) -> None:
    """Reject stale profile caches before evaluation."""
    required_fields = {
        "profiles",
        "photo_ids",
        "identities",
        "sides",
        "dates",
        "skipped_count",
        "manifest_fingerprint",
        "manifest_row_count",
        "profile_cache_version",
        "tear_profile_bins",
    }
    missing = required_fields.difference(cached.files)
    if missing:
        raise RuntimeError(
            f"Profile cache is missing strict metadata: {sorted(missing)}. "
            "Run `uv run python scripts/evaluation.py --rebuild-cache`."
        )

    actual = str(cached["manifest_fingerprint"].item())
    row_count = int(cached["manifest_row_count"])
    cache_version = str(cached["profile_cache_version"].item())
    bins = int(cached["tear_profile_bins"])
    if actual != expected or row_count != len(manifest_rows):
        raise RuntimeError(
            "Profile cache is stale relative to outputs/high_quality/manifest.csv. "
            "Run `uv run python scripts/evaluation.py --rebuild-cache`."
        )
    if cache_version != PROFILE_CACHE_VERSION:
        raise RuntimeError(
            "Profile cache was built by an older evaluation contract. "
            "Run `uv run python scripts/evaluation.py --rebuild-cache`."
        )
    if bins != TEAR_PROFILE_BINS or cached["profiles"].shape[1] != TEAR_PROFILE_BINS:
        raise RuntimeError(
            "Profile cache tear-profile width does not match this code. "
            "Run `uv run python scripts/evaluation.py --rebuild-cache`."
        )
    if int(cached["skipped_count"]) != 0:
        raise RuntimeError(
            "Profile cache contains skipped manifest rows. "
            "Run `uv run python scripts/evaluation.py --rebuild-cache`."
        )


def read_manifest() -> list[dict[str, str]]:
    """Read the high-quality manifest rows.

    Required columns are ``photo_identifier``, ``identity``, ``side``, and
    ``sighting_date``. Other columns describe the selected crop and source
    paths but are not used by matching.

    Raises:
        RuntimeError: If the manifest is missing or empty.
    """
    if not MANIFEST_CSV.exists():
        raise RuntimeError(f"High-quality manifest not found: {MANIFEST_CSV}")
    with MANIFEST_CSV.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"High-quality manifest has no header: {MANIFEST_CSV}")
        missing = set(MANIFEST_FIELDS).difference(reader.fieldnames)
        if missing:
            raise RuntimeError(
                f"High-quality manifest is missing required columns: {sorted(missing)}"
            )
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"High-quality manifest is empty: {MANIFEST_CSV}")
    validate_manifest_rows(rows)
    return rows


def validate_manifest_rows(rows: list[dict[str, str]]) -> None:
    """Reject manifest rows that would make the evaluation ambiguous."""
    seen_ears: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=2):
        for field in MANIFEST_FIELDS:
            if not row[field].strip():
                raise RuntimeError(f"Manifest row {index} has an empty {field!r}")
        if row["side"] not in {"left", "right"}:
            raise RuntimeError(
                f"Manifest row {index} has invalid side {row['side']!r}; "
                "expected 'left' or 'right'"
            )
        ear_key = (row["photo_identifier"], row["side"])
        if ear_key in seen_ears:
            raise RuntimeError(
                f"Manifest row {index} duplicates photo/side {ear_key!r}; "
                "duplicate evaluation ears would leak exact-image matches"
            )
        seen_ears.add(ear_key)


def profile_for(
    manifest_row: dict[str, str],
    dataset: Dataset,
    analyzer: PhotoAnalyzer,
    analysis_cache: dict[str, dict],
) -> np.ndarray:
    """Return the requested ear profile for a manifest row.

    Raises:
        RuntimeError: If analysis fails, the requested side is missing, or the
            profile shape is incompatible with the evaluator.
    """
    photo_id = manifest_row["photo_identifier"]
    if photo_id not in analysis_cache:
        try:
            analysis_cache[photo_id] = analyzer.analyze(dataset.get_photo(photo_id))
        except Exception as error:
            raise RuntimeError(f"Analysis failed for {photo_id}: {error}") from error

    analysis = analysis_cache[photo_id]
    for ear_data in analysis["ears"]:
        if ear_data["ear"].side == manifest_row["side"]:
            profile = np.asarray(ear_data["tear_profile"].profile, dtype=np.float64)
            if profile.shape == (TEAR_PROFILE_BINS,):
                return profile
            raise RuntimeError(
                f"{photo_id} {manifest_row['side']} profile has shape "
                f"{profile.shape}; expected {(TEAR_PROFILE_BINS,)}"
            )
    raise RuntimeError(f"{photo_id} has no {manifest_row['side']} ear analysis")


def load_or_compute_pairwise_scores(
    profile_set: ProfileSet,
    matcher: TearMatcher,
    *,
    rebuild: bool,
) -> np.ndarray:
    """Load strict raw pairwise scores, or compute them when no cache exists."""
    if PAIRWISE_CACHE.exists() and not rebuild:
        cached = np.load(PAIRWISE_CACHE, allow_pickle=False)
        validate_pairwise_cache(cached, profile_set, matcher)
        logger.info(f"Loading cached pairwise scores from {PAIRWISE_CACHE}")
        return cached["raw_pair_scores"]

    raw_pair_scores = pairwise_scores(profile_set, matcher)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PAIRWISE_CACHE,
        raw_pair_scores=raw_pair_scores,
        profile_fingerprint=np.asarray(profile_set_fingerprint(profile_set)),
        matcher_fingerprint=np.asarray(matcher_fingerprint(matcher)),
        pairwise_cache_version=np.asarray(PAIRWISE_CACHE_VERSION),
    )
    logger.info(f"Cached raw pairwise scores at {PAIRWISE_CACHE}")
    return raw_pair_scores


def validate_pairwise_cache(
    cached: np.lib.npyio.NpzFile,
    profile_set: ProfileSet,
    matcher: TearMatcher,
) -> None:
    """Reject raw pairwise score caches that do not belong to this run."""
    required_fields = {
        "raw_pair_scores",
        "profile_fingerprint",
        "matcher_fingerprint",
        "pairwise_cache_version",
    }
    missing = required_fields.difference(cached.files)
    if missing:
        raise RuntimeError(
            f"Pairwise cache is missing strict metadata: {sorted(missing)}. "
            "Run `uv run python scripts/evaluation.py --rebuild-pairwise-cache`."
        )

    row_count = len(profile_set.profiles)
    if cached["raw_pair_scores"].shape != (row_count, row_count):
        raise RuntimeError(
            "Pairwise cache shape does not match the profile cache. "
            "Run `uv run python scripts/evaluation.py --rebuild-pairwise-cache`."
        )
    if str(cached["profile_fingerprint"].item()) != profile_set_fingerprint(profile_set):
        raise RuntimeError(
            "Pairwise cache was computed from a different profile cache. "
            "Run `uv run python scripts/evaluation.py --rebuild-pairwise-cache`."
        )
    if str(cached["pairwise_cache_version"].item()) != PAIRWISE_CACHE_VERSION:
        raise RuntimeError(
            "Pairwise cache was built by an older evaluation contract. "
            "Run `uv run python scripts/evaluation.py --rebuild-pairwise-cache`."
        )
    if str(cached["matcher_fingerprint"].item()) != matcher_fingerprint(matcher):
        raise RuntimeError(
            "Pairwise cache was computed with a different matcher configuration. "
            "Run `uv run python scripts/evaluation.py --rebuild-pairwise-cache`."
        )


def profile_set_fingerprint(profile_set: ProfileSet) -> str:
    """Fingerprint the exact profiles and labels used by pairwise matching."""
    digest = hashlib.sha256()
    digest.update(profile_set.manifest_fingerprint.encode())
    digest.update(np.ascontiguousarray(profile_set.profiles).tobytes())
    for values in (
        profile_set.photo_ids,
        profile_set.identities,
        profile_set.sides,
        profile_set.dates,
    ):
        digest.update("\x1f".join(values.astype(str)).encode())
        digest.update(b"\x1e")
    digest.update(str(profile_set.skipped_count).encode())
    return digest.hexdigest()


def matcher_fingerprint(matcher: TearMatcher) -> str:
    """Fingerprint the matcher configuration that produced raw scores."""
    digest = hashlib.sha256()
    digest.update(repr(matcher.config).encode())
    return digest.hexdigest()


def pairwise_scores(profile_set: ProfileSet, matcher: TearMatcher) -> np.ndarray:
    """Score every same-side profile pair; other entries stay NaN.

    The matrix is directional at this stage because the matcher transforms the
    query profile against the candidate profile. Cohort normalization
    symmetrizes it before ranking.
    """
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


def symmetrize_pairwise_scores(pair_scores: np.ndarray) -> np.ndarray:
    """Average directional pair scores without cohort normalization."""
    return (pair_scores + pair_scores.T) / 2.0


def evaluation_score_matrix(raw_pair_scores: np.ndarray, *, normalize: bool) -> np.ndarray:
    """Return the score matrix selected for this evaluation run."""
    if normalize:
        return symmetrized_cohort_z(raw_pair_scores)
    return symmetrize_pairwise_scores(raw_pair_scores)


def retrieval_metrics(ranks: np.ndarray) -> dict[str, float]:
    """Summarize retrieval quality over positive ranks."""
    ranks = np.asarray(ranks, dtype=np.float64)
    if len(ranks) == 0:
        raise RuntimeError("Cannot summarize retrieval metrics with no scored queries")
    metrics = {
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "count": float(len(ranks)),
    }
    for k in TOP_KS:
        metrics[f"top{k}"] = float(np.mean(ranks <= k))
    return metrics


def evaluate_seeds(
    profile_set: ProfileSet,
    pair_scores: np.ndarray,
    calibrators: dict[str, TearScoreCalibrator] | None,
    *,
    seeds: tuple[int, ...],
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate the selected scoring stack across image-pair seeds."""
    results: dict[str, list[dict[str, float]]] = {
        mode: [] for mode in REPORTED_MODES
    }
    for seed in seeds:
        image_pairs = build_image_pairs(profile_set, seed=seed)
        metrics = elephant_level_evaluation(
            profile_set,
            pair_scores,
            image_pairs,
            calibrators=calibrators,
        )
        for mode, mode_metrics in metrics.items():
            results[mode].append(mode_metrics)
    return {
        mode: summarize_seed_metrics(mode_results)
        for mode, mode_results in results.items()
    }


def summarize_seed_metrics(
    metrics_by_seed: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Return mean and sample standard deviation for metric dictionaries."""
    if not metrics_by_seed:
        raise RuntimeError("Cannot summarize zero evaluation seeds")
    summary: dict[str, dict[str, float]] = {}
    keys = metrics_by_seed[0].keys()
    for key in keys:
        values = np.asarray([metrics[key] for metrics in metrics_by_seed], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def elephant_level_evaluation(
    profile_set: ProfileSet,
    pair_scores: np.ndarray,
    image_pairs: list[ImagePair],
    *,
    calibrators: dict[str, TearScoreCalibrator] | None,
) -> dict[str, dict[str, float]]:
    """Evaluate elephant-level retrieval: withhold one image pair.

    This is the paper-facing protocol. It uses cohort-normalized scores,
    identity-disjoint calibration, same-side photo-pair maxima, and side-score
    averaging to rank known elephants for each query image pair.
    """
    masses = tear_mass(profile_set.profiles)
    pairs_per_identity = Counter(image_pair.identity for image_pair in image_pairs)

    ranks: dict[str, list[int]] = {mode: [] for mode in REPORTED_MODES}
    for query in image_pairs:
        if pairs_per_identity[query.identity] < 2:
            continue
        calibrator = None if calibrators is None else calibrators[query.identity]
        elephant_side_scores = best_side_scores_per_elephant(
            profile_set, pair_scores, masses, calibrator, query, image_pairs
        )
        for mode, sides in RANKING_SIDES.items():
            rank = elephant_rank(elephant_side_scores, query.identity, sides)
            if rank is None:
                continue
            ranks[mode].append(rank)

    return {mode: retrieval_metrics(np.asarray(mode_ranks)) for mode, mode_ranks in ranks.items()}


def build_image_pairs(profile_set: ProfileSet, *, seed: int) -> list[ImagePair]:
    """Pair each elephant's left/right profiles into image pairs.

    For each elephant, shuffled left and right rows are paired up to the
    smaller side count; surplus one-sided rows are ignored.
    """
    rng = np.random.default_rng(seed)
    image_pairs: list[ImagePair] = []
    for identity in sorted(np.unique(profile_set.identities)):
        identity_rows = np.flatnonzero(profile_set.identities == identity)
        left_rows = identity_rows[profile_set.sides[identity_rows] == "left"].copy()
        right_rows = identity_rows[profile_set.sides[identity_rows] == "right"].copy()
        rng.shuffle(left_rows)
        rng.shuffle(right_rows)
        pair_count = min(len(left_rows), len(right_rows))
        for pair_index in range(pair_count):
            image_pairs.append(
                ImagePair(
                    identity=str(identity),
                    pair_id=f"{identity}_{pair_index:02d}",
                    rows=np.asarray(
                        [left_rows[pair_index], right_rows[pair_index]],
                        dtype=np.int64,
                    ),
                )
            )
    logger.info(f"Built {len(image_pairs)} paired image pairs")
    return image_pairs


def image_pair_count(profile_set: ProfileSet) -> int:
    """Return the number of left/right pairs available across identities."""
    total = 0
    for identity in np.unique(profile_set.identities):
        identity_rows = np.flatnonzero(profile_set.identities == identity)
        left_count = int((profile_set.sides[identity_rows] == "left").sum())
        right_count = int((profile_set.sides[identity_rows] == "right").sum())
        total += min(left_count, right_count)
    return total


def identity_calibrators(
    profile_set: ProfileSet,
    pair_scores: np.ndarray,
    masses: np.ndarray,
    *,
    seed: int,
) -> dict[str, TearScoreCalibrator]:
    """Map each identity to a calibrator fitted without that identity.

    Identities are shuffled into ``CALIBRATION_FOLDS`` folds; each fold's
    calibrator is trained on the other folds only. This prevents a query
    elephant from being scored by a calibrator that learned from that
    elephant's positive pairs.
    """
    identities = np.unique(profile_set.identities)
    rng = np.random.default_rng(seed)
    rng.shuffle(identities)
    fold_of_identity = {
        identity: index % CALIBRATION_FOLDS for index, identity in enumerate(identities)
    }
    calibrator_of_fold = {
        fold: fit_fold_calibrator(
            profile_set,
            pair_scores,
            masses,
            fold_of_identity,
            fold,
            seed=seed,
        )
        for fold in range(CALIBRATION_FOLDS)
    }
    return {identity: calibrator_of_fold[fold] for identity, fold in fold_of_identity.items()}


def fit_fold_calibrator(
    profile_set: ProfileSet,
    pair_scores: np.ndarray,
    masses: np.ndarray,
    fold_of_identity: dict[str, int],
    fold: int,
    *,
    seed: int,
) -> TearScoreCalibrator:
    """Fit one calibrator on same-side pairs whose identities are outside the fold.

    Every genuine pair is kept; impostor pairs are subsampled per query to
    roughly ``NEGATIVES_PER_POSITIVE`` per genuine pair. ``pair_scores`` are
    the selected score matrix for this run, and labels are same-identity vs
    different-identity within the same ear side.
    """
    train_rows = np.flatnonzero(
        [fold_of_identity[identity] != fold for identity in profile_set.identities]
    )
    rng = np.random.default_rng(seed * CALIBRATION_FOLDS + fold)
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
    calibrator: TearScoreCalibrator | None,
    query: ImagePair,
    image_pairs: list[ImagePair],
) -> dict[str, dict[str, float]]:
    """Return each gallery elephant's best selected score per ear side.

    Aggregation is intentionally max-based: best photo pair per side for a
    high-quality image pair, then best score per side for a gallery elephant.
    """
    best: dict[str, dict[str, float]] = defaultdict(dict)
    for gallery in image_pairs:
        if gallery is query:
            continue
        side_scores = best[gallery.identity]
        for side, score in image_pair_scores(
            profile_set, pair_scores, masses, calibrator, query, gallery
        ).items():
            side_scores[side] = max(side_scores.get(side, -np.inf), score)
    return best


def image_pair_scores(
    profile_set: ProfileSet,
    pair_scores: np.ndarray,
    masses: np.ndarray,
    calibrator: TearScoreCalibrator | None,
    query: ImagePair,
    gallery: ImagePair,
) -> dict[str, float]:
    """Return the score between two image pairs, per side.

    Left profiles are compared only with left profiles, and right profiles only
    with right profiles.
    """
    scores_by_side: dict[str, float] = {}
    for side in ("left", "right"):
        query_rows = query.rows[profile_set.sides[query.rows] == side]
        gallery_rows = gallery.rows[profile_set.sides[gallery.rows] == side]
        if len(query_rows) == 0 or len(gallery_rows) == 0:
            continue
        raw_scores = pair_scores[np.ix_(query_rows, gallery_rows)].ravel()
        if calibrator is None:
            pair_scores_for_side = raw_scores
        else:
            pair_scores_for_side = calibrator.calibrated_score(
                raw_scores,
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
    the side scores. Candidates must have all requested sides, so ``combined``
    cannot degrade into one-ear ranking. Returns None when the query elephant
    lacks a requested side. Ties with the query identity are assigned the same
    rank because only strictly higher scores count ahead.
    """
    elephant_scores = {
        elephant: float(np.mean([side_scores[side] for side in sides]))
        for elephant, side_scores in elephant_side_scores.items()
        if all(side in side_scores for side in sides)
    }
    if query_identity not in elephant_scores:
        return None
    query_score = elephant_scores[query_identity]
    return 1 + sum(score > query_score for score in elephant_scores.values())


def write_summary(
    profile_set: ProfileSet,
    evaluation: dict[str, dict[str, dict[str, float]]],
    *,
    seeds: tuple[int, ...],
    normalized: bool,
    calibrated: bool,
) -> None:
    """Write and print the metric summary."""
    score_label = score_stack_label(normalized=normalized, calibrated=calibrated)
    lines = [
        "Two-ear retrieval from high-quality, high-resolution image pairs",
        f"profiles: {len(profile_set.profiles)} ({profile_set.skipped_count} skipped)",
        f"constructed image pairs: {image_pair_count(profile_set)}",
        f"pairing seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)})",
        f"score stack: {score_label}",
        f"manifest fingerprint: {profile_set.manifest_fingerprint}",
        f"profile cache fingerprint: {profile_set_fingerprint(profile_set)}",
        "",
    ]
    lines.extend(format_modes(evaluation))
    text = "\n".join(lines) + "\n"
    summary_path(normalized=normalized, calibrated=calibrated).write_text(text)
    print("\n".join(lines))


def score_stack_label(*, normalized: bool, calibrated: bool) -> str:
    """Return a concise name for the selected scoring stack."""
    score = "cohort-normalized" if normalized else "raw"
    suffix = "calibrated" if calibrated else "uncalibrated"
    return f"{score} + {suffix}"


def summary_path(*, normalized: bool, calibrated: bool) -> Path:
    """Return the stack-specific summary path for this run."""
    if normalized and calibrated:
        name = "full"
    elif normalized:
        name = "no_calibration"
    elif calibrated:
        name = "no_normalization"
    else:
        name = "raw_only"
    return OUTPUT_DIR / f"hq_summary_{name}.txt"


def format_modes(
    metrics_by_mode: dict[str, dict[str, dict[str, float]]],
) -> list[str]:
    """Format all reported side modes."""
    return [
        f"  {mode}: {format_metrics(metrics_by_mode[mode])}"
        for mode in REPORTED_MODES
    ]


def format_metrics(metrics: dict[str, dict[str, float]]) -> str:
    """Format one retrieval metric row."""
    return (
        f"n={format_metric(metrics['count'], precision=0)} "
        f"top1={format_metric(metrics['top1'])} "
        f"top3={format_metric(metrics['top3'])} "
        f"top5={format_metric(metrics['top5'])} "
        f"top10={format_metric(metrics['top10'])} "
        f"top15={format_metric(metrics['top15'])} "
        f"MRR={format_metric(metrics['mrr'])} "
        f"median_rank={format_metric(metrics['median_rank'], precision=1)}"
    )


def format_metric(
    metric: dict[str, float],
    *,
    precision: int = 3,
) -> str:
    """Format a metric mean, with spread when multiple seeds were evaluated."""
    mean = metric["mean"]
    std = metric["std"]
    if precision == 0:
        base = f"{mean:.0f}"
        spread = f"{std:.0f}"
    else:
        base = f"{mean:.{precision}f}"
        spread = f"{std:.{precision}f}"
    if std == 0.0:
        return base
    return f"{base}±{spread}"


if __name__ == "__main__":
    main()
"""
Outputs:

1. With calibration and normalization:
  left: n=429 top1=0.483±0.005 top3=0.625±0.005 top5=0.693±0.004 top10=0.768±0.004 top15=0.802±0.003 MRR=0.577±0.004 median_rank=2.0
  right: n=429 top1=0.433±0.007 top3=0.562±0.007 top5=0.612±0.009 top10=0.691±0.006 top15=0.743±0.005 MRR=0.524±0.006 median_rank=2.0
  combined: n=429 top1=0.648±0.011 top3=0.763±0.007 top5=0.813±0.006 top10=0.875±0.009 top15=0.911±0.009 MRR=0.724±0.007 median_rank=1.0
uv run python scripts/evaluation.py --seed 0 --seeds 50  535.40s user 2.03s system 99% cpu 8:58.73 total

2. With normalization but no calibration:
  left: n=429 top1=0.484±0.005 top3=0.620±0.005 top5=0.690±0.004 top10=0.766±0.004 top15=0.804±0.002 MRR=0.576±0.004 median_rank=2.0
  right: n=429 top1=0.429±0.007 top3=0.556±0.008 top5=0.601±0.008 top10=0.690±0.006 top15=0.740±0.006 MRR=0.521±0.006 median_rank=2.0
  combined: n=429 top1=0.660±0.010 top3=0.768±0.008 top5=0.816±0.008 top10=0.872±0.009 top15=0.909±0.008 MRR=0.732±0.006 median_rank=1.0
uv run python scripts/evaluation.py --seed 0 --seeds 50 --no-calibration  73.33s user 0.72s system 99% cpu 1:14.51 total

3. With no normalization but with calibration:
  left: n=429 top1=0.481±0.006 top3=0.608±0.005 top5=0.690±0.004 top10=0.762±0.003 top15=0.802±0.003 MRR=0.572±0.004 median_rank=2.0
  right: n=429 top1=0.402±0.006 top3=0.553±0.007 top5=0.603±0.007 top10=0.698±0.006 top15=0.751±0.005 MRR=0.506±0.005 median_rank=2.1±0.3
  combined: n=429 top1=0.638±0.010 top3=0.757±0.007 top5=0.810±0.008 top10=0.868±0.009 top15=0.902±0.008 MRR=0.717±0.006 median_rank=1.0
uv run python scripts/evaluation.py --seed 0 --seeds 50 --no-normalization  535.36s user 2.00s system 99% cpu 8:58.63 total

4. With no normalization and no calibration:
  left: n=429 top1=0.477±0.006 top3=0.610±0.005 top5=0.694±0.004 top10=0.761±0.004 top15=0.796±0.003 MRR=0.571±0.004 median_rank=2.0
  right: n=429 top1=0.437±0.006 top3=0.554±0.007 top5=0.613±0.007 top10=0.692±0.006 top15=0.751±0.006 MRR=0.525±0.006 median_rank=2.1±0.3
  combined: n=429 top1=0.659±0.009 top3=0.764±0.007 top5=0.809±0.008 top10=0.872±0.007 top15=0.905±0.008 MRR=0.730±0.006 median_rank=1.0
uv run python scripts/evaluation.py --seed 0 --seeds 50 --no-normalization --no-calibration  68.08s user 0.39s system 99% cpu 1:08.54 total
"""
