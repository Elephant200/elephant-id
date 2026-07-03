"""Build the Alphaphant demo: a held-out gallery plus demo sighting folders.

Reads the high-quality profile cache and manifest (never modifying
``outputs/high_quality``), selects a few sightings whose elephant remains
well-represented in the gallery, and writes under ``outputs/alphaphant_demo``:

- ``gallery_profiles.npz``: the gallery with the held-out sightings removed.
- ``sightings/{Name}_{date}/``: copies of the held-out sightings' original
  photos (only manifest-quality images), ready to import in the app.
- ``holdout_summary.txt``: which sightings were held out.

Run with ``uv run python scripts/make_demo_holdout.py``.
"""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

from elephant_id.log import configure_logging
from elephant_id.matching import (
    TearMatcher,
    TearScoreCalibrator,
    symmetrized_cohort_z,
    tear_mass,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_CACHE = REPO_ROOT / "outputs" / "tear_matching_eval" / "hq_profiles.npz"
PAIRWISE_CACHE = REPO_ROOT / "outputs" / "alphaphant" / "gallery_pairwise.npy"
MANIFEST_CSV = REPO_ROOT / "outputs" / "high_quality" / "manifest.csv"
DATASET_ROOT = REPO_ROOT / "dataset" / "elephants-alive" / "coded"
DEMO_DIR = REPO_ROOT / "outputs" / "alphaphant_demo"

MAX_ROWS_PER_SIDE = 2
MIN_REMAINING_SIGHTINGS = 2
MAX_TRUTH_RANK = 2
NEGATIVES_PER_POSITIVE = 5


def main() -> None:
    """Select held-out sightings and write the demo gallery and folders."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=int, default=4)
    args = parser.parse_args()

    cached = np.load(PROFILE_CACHE, allow_pickle=False)
    profiles = np.asarray(cached["profiles"], dtype=np.float64)
    identities = np.asarray([str(value) for value in cached["identities"]])
    dates = np.asarray([str(value) for value in cached["dates"]])
    sides = np.asarray([str(value) for value in cached["sides"]])
    photo_ids = np.asarray([str(value) for value in cached["photo_ids"]])

    ranker = HoldoutRanker(profiles, identities, sides)
    held_out = select_holdouts(identities, dates, sides, args.holdouts, ranker)
    held_rows = np.zeros(len(identities), dtype=bool)
    for identity, date in held_out:
        held_rows |= (identities == identity) & (dates == date)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    write_filtered_gallery(cached, ~held_rows)
    source_paths = manifest_source_paths()
    summary_lines = []
    for identity, date in held_out:
        rows = np.flatnonzero((identities == identity) & (dates == date))
        folder = copy_sighting_photos(identity, date, photo_ids[rows], source_paths)
        row_sides = sides[rows]
        summary_lines.append(
            f"{folder.name}: {identity} on {date} "
            f"({int((row_sides == 'left').sum())} left, "
            f"{int((row_sides == 'right').sum())} right) -> {folder}"
        )

    summary = "\n".join(summary_lines) + "\n"
    (DEMO_DIR / "holdout_summary.txt").write_text(summary)
    logger.info(f"Held out {len(held_out)} sightings:\n{summary}")


def select_holdouts(
    identities: np.ndarray,
    dates: np.ndarray,
    sides: np.ndarray,
    count: int,
    ranker: "HoldoutRanker",
) -> list[tuple[str, str]]:
    """Pick held-out sightings that also demo well.

    A sighting qualifies when it has one or two profiles on each side, its
    elephant keeps at least ``MIN_REMAINING_SIGHTINGS`` other sightings with
    both sides still represented in the remaining gallery, and — so the demo
    shows the matcher succeeding honestly — the true elephant ranks within
    ``MAX_TRUTH_RANK`` when the sighting is matched against the rest.
    """
    sighting_rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row, key in enumerate(zip(identities, dates, strict=True)):
        sighting_rows[key].append(row)

    sightings_per_identity: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in sorted(sighting_rows):
        sightings_per_identity[key[0]].append(key)

    held_out: list[tuple[str, str]] = []
    for identity in sorted(sightings_per_identity):
        if len(held_out) >= count:
            break
        keys = sightings_per_identity[identity]
        if len(keys) < MIN_REMAINING_SIGHTINGS + 1:
            continue
        for key in keys:
            rows = sighting_rows[key]
            row_sides = sides[rows]
            left = int((row_sides == "left").sum())
            right = int((row_sides == "right").sum())
            if not (1 <= left <= MAX_ROWS_PER_SIDE and 1 <= right <= MAX_ROWS_PER_SIDE):
                continue
            remaining = [other for other in keys if other != key]
            if not remaining_sides_covered(remaining, sighting_rows, sides):
                continue
            rank = ranker.truth_rank(np.asarray(rows))
            if rank > MAX_TRUTH_RANK:
                logger.info(f"Skipping {key}: truth rank {rank}")
                continue
            logger.info(f"Selected {key}: truth rank {rank}")
            held_out.append(key)
            break
    if len(held_out) < count:
        logger.warning(f"Only found {len(held_out)} qualifying holdout sightings")
    return held_out


class HoldoutRanker:
    """Rank an elephant for a candidate held-out sighting, evaluation-style.

    Uses the precomputed full pairwise matrix, cohort normalization over all
    rows, and one calibrator fitted on all same-side pairs. This is a
    selection heuristic that closely mirrors the app's matching engine.
    """

    def __init__(
        self,
        profiles: np.ndarray,
        identities: np.ndarray,
        sides: np.ndarray,
    ) -> None:
        self.identities = identities
        self.sides = sides
        self.normalized = symmetrized_cohort_z(
            _load_or_compute_pairwise(profiles, sides)
        )
        self.masses = tear_mass(profiles)
        self.calibrator = _fit_calibrator(self.normalized, self.masses, identities, sides)
        self.rows_by_identity: dict[str, np.ndarray] = {
            identity: np.flatnonzero(identities == identity)
            for identity in np.unique(identities)
        }

    def truth_rank(self, held_rows: np.ndarray) -> int:
        """Return the true elephant's rank when held rows query the rest."""
        held = set(held_rows.tolist())
        truth = self.identities[held_rows[0]]
        scores: dict[str, float] = {}
        for identity, rows in self.rows_by_identity.items():
            side_scores = []
            for side in ("left", "right"):
                query = [row for row in held_rows if self.sides[row] == side]
                gallery = [
                    row for row in rows if self.sides[row] == side and row not in held
                ]
                if not query or not gallery:
                    continue
                pair_z = self.normalized[np.ix_(query, gallery)].ravel()
                calibrated = self.calibrator.calibrated_score(
                    pair_z,
                    np.repeat(self.masses[query], len(gallery)),
                    np.tile(self.masses[gallery], len(query)),
                )
                side_scores.append(float(calibrated.max()))
            if side_scores:
                scores[identity] = float(np.mean(side_scores))
        if truth not in scores:
            return len(scores) + 1
        return 1 + sum(score > scores[truth] for score in scores.values())


def _load_or_compute_pairwise(profiles: np.ndarray, sides: np.ndarray) -> np.ndarray:
    """Load the full-gallery pairwise matrix, computing it if missing."""
    row_count = len(profiles)
    if PAIRWISE_CACHE.exists():
        scores = np.load(PAIRWISE_CACHE)
        if scores.shape == (row_count, row_count):
            return scores
    logger.info(f"Computing pairwise scores for {row_count} profiles")
    matcher = TearMatcher()
    scores = np.full((row_count, row_count), np.nan)
    for side in ("left", "right"):
        rows = np.flatnonzero(sides == side)
        for query_index in rows:
            others = rows[rows != query_index]
            scores[query_index, others] = matcher.match_gallery(
                profiles[query_index], profiles[others]
            ).score
    PAIRWISE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(PAIRWISE_CACHE, scores)
    return scores


def _fit_calibrator(
    normalized: np.ndarray,
    masses: np.ndarray,
    identities: np.ndarray,
    sides: np.ndarray,
) -> TearScoreCalibrator:
    """Fit one calibrator on all labeled same-side pairs."""
    rng = np.random.default_rng(0)
    row_indices = np.arange(len(identities))
    queries: list[np.ndarray] = []
    candidates_list: list[np.ndarray] = []
    for query_index in row_indices:
        candidates = row_indices[
            (sides == sides[query_index]) & (row_indices != query_index)
        ]
        if len(candidates) == 0:
            continue
        genuine = identities[candidates] == identities[query_index]
        positives = candidates[genuine]
        negative_count = min(
            int((~genuine).sum()), NEGATIVES_PER_POSITIVE * max(len(positives), 1)
        )
        negatives = rng.choice(candidates[~genuine], size=negative_count, replace=False)
        chosen = np.concatenate([positives, negatives])
        queries.append(np.full(len(chosen), query_index))
        candidates_list.append(chosen)
    query_rows = np.concatenate(queries)
    candidate_rows = np.concatenate(candidates_list)
    calibrator = TearScoreCalibrator()
    calibrator.fit(
        normalized[query_rows, candidate_rows],
        masses[query_rows],
        masses[candidate_rows],
        identities[query_rows] == identities[candidate_rows],
    )
    return calibrator


def remaining_sides_covered(
    remaining: list[tuple[str, str]],
    sighting_rows: dict[tuple[str, str], list[int]],
    sides: np.ndarray,
) -> bool:
    """Whether the elephant's remaining sightings still cover both sides."""
    remaining_sides = {
        side for key in remaining for side in sides[sighting_rows[key]]
    }
    return {"left", "right"} <= remaining_sides


def write_filtered_gallery(cached: np.lib.npyio.NpzFile, keep: np.ndarray) -> None:
    """Write the gallery npz with only the kept rows."""
    output = DEMO_DIR / "gallery_profiles.npz"
    np.savez_compressed(
        output,
        profiles=cached["profiles"][keep],
        photo_ids=cached["photo_ids"][keep],
        identities=cached["identities"][keep],
        sides=cached["sides"][keep],
        dates=cached["dates"][keep],
        skipped_count=cached["skipped_count"],
    )
    logger.info(f"Wrote filtered gallery ({int(keep.sum())} rows) to {output}")


def manifest_source_paths() -> dict[str, Path]:
    """Map photo identifiers to their original dataset photo paths."""
    paths: dict[str, Path] = {}
    with MANIFEST_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            paths[row["photo_identifier"]] = DATASET_ROOT / row["source_image_path"]
    return paths


def copy_sighting_photos(
    identity: str,
    date: str,
    photo_ids: np.ndarray,
    source_paths: dict[str, Path],
) -> Path:
    """Copy one held-out sighting's original photos into a demo folder."""
    folder = DEMO_DIR / "sightings" / f"{identity}_{date}"
    folder.mkdir(parents=True, exist_ok=True)
    for photo_id in sorted(set(photo_ids)):
        source = source_paths.get(photo_id)
        if source is None or not source.exists():
            logger.warning(f"Source photo missing for {photo_id}: {source}")
            continue
        shutil.copy2(source, folder / source.name)
    return folder


if __name__ == "__main__":
    main()
