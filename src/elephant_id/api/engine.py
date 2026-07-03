"""Rank catalog elephants against query tear profiles.

Mirrors the elephant-level protocol in ``scripts/evaluation.py``: raw
``TearMatcher`` pair scores over an extended gallery+query matrix,
``symmetrized_cohort_z`` normalization, tear-mass-conditioned calibration,
best calibrated score per elephant per ear side, then averaging the
available sides. One calibrator is fitted on all gallery pairs because the
query elephant is unknown at match time.
"""

import threading
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from elephant_id.api.gallery import GalleryData
from elephant_id.api.profiles import plot_profile
from elephant_id.matching import (
    TearMatcher,
    TearScoreCalibrator,
    symmetrized_cohort_z,
    tear_mass,
)

EAR_SIDES = ("left", "right")
NEGATIVES_PER_POSITIVE = 5
CALIBRATION_SEED = 0
# Strength labels come from the impostor score distribution at fit time:
# "strong" beats this fraction of impostor pairs, "moderate" the lower one.
STRONG_IMPOSTOR_PERCENTILE = 99.0
MODERATE_IMPOSTOR_PERCENTILE = 90.0


@dataclass(frozen=True)
class SideEvidence:
    """Best same-side profile pair supporting one candidate elephant."""

    side: str
    score: float
    query_photo_id: str
    gallery_photo_id: str
    gallery_date: str
    gallery_crop_path: str | None
    query_profile: tuple[float, ...] = ()
    gallery_profile: tuple[float, ...] = ()
    strength: str = ""


@dataclass(frozen=True)
class RankedElephant:
    """One catalog elephant's match result for a query sighting."""

    identity: str
    score: float
    confidence: float
    evidence: tuple[SideEvidence, ...]

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)


class MatchingEngine:
    """Score query sightings against the known-elephant profile gallery."""

    def __init__(
        self,
        gallery: GalleryData,
        pairwise_cache_path: Path,
        matcher: TearMatcher | None = None,
    ) -> None:
        """Load or compute the gallery pairwise matrix and fit the calibrator.

        Args:
            gallery: Precomputed gallery profiles and metadata.
            pairwise_cache_path: ``.npy`` cache for the raw same-side pairwise
                score matrix; recomputed and rewritten when stale.
            matcher: Matcher override, mainly for tests.
        """
        self._matcher = matcher or TearMatcher()
        self._lock = threading.Lock()
        self._profiles = np.asarray(gallery.profiles, dtype=np.float64).copy()
        self._photo_ids = list(gallery.photo_ids)
        self._identities = list(gallery.identities)
        self._sides = list(gallery.sides)
        self._dates = list(gallery.dates)
        self._crop_paths = list(gallery.crop_paths)
        self._rows_by_identity: dict[str, list[int]] = defaultdict(list)
        for row, identity in enumerate(self._identities):
            self._rows_by_identity[identity].append(row)

        self._raw = self._load_or_compute_pairwise(pairwise_cache_path)
        self._strength_thresholds: tuple[float, float] = (2.0, 0.5)
        self._calibrator = self._fit_calibrator()
        logger.info(
            f"Matching engine ready: {len(self._profiles)} profiles, "
            f"{len(self._rows_by_identity)} elephants"
        )

    @property
    def elephant_count(self) -> int:
        """Number of distinct elephants in the gallery."""
        with self._lock:
            return len(self._rows_by_identity)

    @property
    def profile_count(self) -> int:
        """Number of profile rows in the gallery."""
        with self._lock:
            return len(self._profiles)

    def has_identity(self, identity: str) -> bool:
        """Whether the gallery already contains the named elephant."""
        with self._lock:
            return identity in self._rows_by_identity

    def catalog(self) -> list[dict]:
        """Summarize every gallery elephant for the catalog view."""
        with self._lock:
            summaries = []
            for identity in sorted(self._rows_by_identity):
                rows = self._rows_by_identity[identity]
                side_counts = {
                    side: sum(1 for row in rows if self._sides[row] == side)
                    for side in EAR_SIDES
                }
                thumbnail = next(
                    (self._crop_paths[row] for row in rows if self._crop_paths[row]),
                    None,
                )
                summaries.append(
                    {
                        "name": identity,
                        "photo_count": len(rows),
                        "sighting_dates": sorted({self._dates[row] for row in rows}),
                        "side_counts": side_counts,
                        "thumbnail": thumbnail,
                    }
                )
            return summaries

    def elephant_detail(self, identity: str) -> dict:
        """Return per-photo rows for one gallery elephant.

        Raises:
            KeyError: If the elephant is not in the gallery.
        """
        with self._lock:
            if identity not in self._rows_by_identity:
                raise KeyError(f"Unknown elephant: {identity}")
            rows = self._rows_by_identity[identity]
            return {
                "name": identity,
                "photos": [
                    {
                        "photo_id": self._photo_ids[row],
                        "side": self._sides[row],
                        "date": self._dates[row],
                        "crop_path": self._crop_paths[row],
                    }
                    for row in sorted(rows, key=lambda r: (self._dates[r], self._photo_ids[r]))
                ],
            }

    def rank(
        self,
        profiles: np.ndarray,
        sides: Sequence[str],
        photo_ids: Sequence[str],
        top_n: int = 12,
    ) -> list[RankedElephant]:
        """Rank gallery elephants against one query sighting.

        Args:
            profiles: Query tear profiles, one row per usable ear.
            sides: Ear side per query row.
            photo_ids: Photo identifier per query row.
            top_n: Maximum candidates to return.

        Returns:
            Candidates in descending score order.

        Raises:
            ValueError: If the query is empty or rows are misaligned.
        """
        query = np.asarray(profiles, dtype=np.float64)
        if query.ndim != 2 or len(query) == 0:
            raise ValueError("Query profiles must be a non-empty 2-D array")
        if not (len(query) == len(sides) == len(photo_ids)):
            raise ValueError("Query profiles, sides, and photo_ids must align")

        with self._lock:
            gallery_count = len(self._profiles)
            extended = self._extended_matrix(query, sides)
            normalized = symmetrized_cohort_z(extended)
            masses = tear_mass(np.vstack([self._profiles, query]))
            gallery_sides = np.asarray(self._sides)
            # Skip exact-photo twins so already-cataloged photos cannot
            # match themselves; ranking must rest on cross-sighting evidence.
            query_photo_set = {str(photo_id) for photo_id in photo_ids}

            results: list[RankedElephant] = []
            for identity, rows in self._rows_by_identity.items():
                evidence: list[SideEvidence] = []
                for side in EAR_SIDES:
                    gallery_rows = [
                        row
                        for row in rows
                        if gallery_sides[row] == side
                        and self._photo_ids[row] not in query_photo_set
                    ]
                    query_rows = [
                        index for index, query_side in enumerate(sides) if query_side == side
                    ]
                    if not gallery_rows or not query_rows:
                        continue
                    global_query_rows = [gallery_count + index for index in query_rows]
                    pair_z = normalized[np.ix_(global_query_rows, gallery_rows)]
                    calibrated = self._calibrator.calibrated_score(
                        pair_z.ravel(),
                        np.repeat(masses[global_query_rows], len(gallery_rows)),
                        np.tile(masses[gallery_rows], len(query_rows)),
                    )
                    best = int(np.argmax(calibrated))
                    best_query = query_rows[best // len(gallery_rows)]
                    best_gallery = gallery_rows[best % len(gallery_rows)]
                    evidence.append(
                        SideEvidence(
                            side=side,
                            score=float(calibrated[best]),
                            query_photo_id=str(photo_ids[best_query]),
                            gallery_photo_id=self._photo_ids[best_gallery],
                            gallery_date=self._dates[best_gallery],
                            gallery_crop_path=self._crop_paths[best_gallery],
                            query_profile=plot_profile(query[best_query]),
                            gallery_profile=plot_profile(self._profiles[best_gallery]),
                            strength=self._strength_label(float(calibrated[best])),
                        )
                    )
                if not evidence:
                    continue
                score = float(np.mean([side_evidence.score for side_evidence in evidence]))
                results.append(
                    RankedElephant(
                        identity=identity,
                        score=score,
                        confidence=float(1.0 / (1.0 + np.exp(-score))),
                        evidence=tuple(evidence),
                    )
                )

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_n]

    def extend(
        self,
        profiles: np.ndarray,
        sides: Sequence[str],
        identity: str,
        date: str,
        photo_ids: Sequence[str],
        crop_paths: Sequence[str | None],
    ) -> None:
        """File new profiles under an elephant and refit the calibrator.

        Used when a reviewed sighting is confirmed against an existing
        elephant or enrolled as a new one.

        Raises:
            ValueError: If rows are misaligned or empty.
        """
        new_profiles = np.asarray(profiles, dtype=np.float64)
        if new_profiles.ndim != 2 or len(new_profiles) == 0:
            raise ValueError("Profiles must be a non-empty 2-D array")
        if not (len(new_profiles) == len(sides) == len(photo_ids) == len(crop_paths)):
            raise ValueError("Profiles, sides, photo_ids, and crop_paths must align")

        with self._lock:
            self._raw = self._extended_matrix(new_profiles, sides)
            start_row = len(self._profiles)
            self._profiles = np.vstack([self._profiles, new_profiles])
            for offset in range(len(new_profiles)):
                self._photo_ids.append(str(photo_ids[offset]))
                self._identities.append(identity)
                self._sides.append(str(sides[offset]))
                self._dates.append(date)
                self._crop_paths.append(crop_paths[offset])
                self._rows_by_identity[identity].append(start_row + offset)
            self._calibrator = self._fit_calibrator()
        logger.info(f"Filed {len(new_profiles)} profiles under {identity} ({date})")

    def _extended_matrix(self, query: np.ndarray, sides: Sequence[str]) -> np.ndarray:
        """Extend the raw pairwise matrix with same-side query scores.

        Both directions are scored so the downstream symmetrization averages
        real values, matching how the gallery matrix is built.
        """
        gallery_count = len(self._profiles)
        query_count = len(query)
        total = gallery_count + query_count
        extended = np.full((total, total), np.nan)
        extended[:gallery_count, :gallery_count] = self._raw

        gallery_sides = np.asarray(self._sides)
        for index in range(query_count):
            same_side = np.flatnonzero(gallery_sides == sides[index])
            row = gallery_count + index
            if len(same_side) > 0:
                extended[row, same_side] = self._matcher.match_gallery(
                    query[index], self._profiles[same_side]
                ).score
                extended[same_side, row] = self._matcher.match_row_pairs(
                    self._profiles[same_side],
                    np.broadcast_to(query[index], (len(same_side), query.shape[1])),
                ).score
            for other in range(query_count):
                if other != index and sides[other] == sides[index]:
                    extended[row, gallery_count + other] = self._matcher.match_pair(
                        query[index], query[other]
                    ).score
        return extended

    def _load_or_compute_pairwise(self, cache_path: Path) -> np.ndarray:
        """Load the gallery pairwise matrix, recomputing when missing or stale."""
        row_count = len(self._profiles)
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (row_count, row_count):
                logger.info(f"Loaded pairwise matrix from {cache_path}")
                return cached
            logger.warning(
                f"Pairwise cache {cache_path} has shape {cached.shape}, "
                f"expected {(row_count, row_count)}; recomputing"
            )

        logger.info(f"Computing pairwise scores for {row_count} gallery profiles")
        scores = np.full((row_count, row_count), np.nan)
        gallery_sides = np.asarray(self._sides)
        for side in EAR_SIDES:
            rows = np.flatnonzero(gallery_sides == side)
            for query_index in rows:
                others = rows[rows != query_index]
                if len(others) == 0:
                    continue
                scores[query_index, others] = self._matcher.match_gallery(
                    self._profiles[query_index], self._profiles[others]
                ).score
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, scores)
        logger.info(f"Cached pairwise matrix at {cache_path}")
        return scores

    def _fit_calibrator(self) -> TearScoreCalibrator:
        """Fit the tear-mass calibrator on all labeled gallery pairs.

        Every genuine same-side pair is kept; impostor pairs are subsampled to
        roughly ``NEGATIVES_PER_POSITIVE`` per genuine pair, as in the
        evaluation protocol.
        """
        normalized = symmetrized_cohort_z(self._raw)
        masses = tear_mass(self._profiles)
        identities = np.asarray(self._identities)
        gallery_sides = np.asarray(self._sides)
        rng = np.random.default_rng(CALIBRATION_SEED)

        query_groups: list[np.ndarray] = []
        candidate_groups: list[np.ndarray] = []
        row_indices = np.arange(len(self._profiles))
        for query_index in row_indices:
            candidates = row_indices[
                (gallery_sides == gallery_sides[query_index])
                & (row_indices != query_index)
            ]
            if len(candidates) == 0:
                continue
            genuine = identities[candidates] == identities[query_index]
            positives = candidates[genuine]
            negative_count = min(
                int((~genuine).sum()),
                NEGATIVES_PER_POSITIVE * max(len(positives), 1),
            )
            negatives = rng.choice(candidates[~genuine], size=negative_count, replace=False)
            chosen = np.concatenate([positives, negatives])
            query_groups.append(np.full(len(chosen), query_index))
            candidate_groups.append(chosen)

        queries = np.concatenate(query_groups)
        candidates = np.concatenate(candidate_groups)
        labels = identities[queries] == identities[candidates]
        calibrator = TearScoreCalibrator()
        calibrator.fit(
            normalized[queries, candidates],
            masses[queries],
            masses[candidates],
            labels,
        )

        calibrated = calibrator.calibrated_score(
            normalized[queries, candidates], masses[queries], masses[candidates]
        )
        impostor_scores = calibrated[~labels]
        self._strength_thresholds = (
            float(np.percentile(impostor_scores, STRONG_IMPOSTOR_PERCENTILE)),
            float(np.percentile(impostor_scores, MODERATE_IMPOSTOR_PERCENTILE)),
        )
        logger.info(
            f"Strength thresholds from impostor pairs: "
            f"strong>={self._strength_thresholds[0]:.2f}, "
            f"moderate>={self._strength_thresholds[1]:.2f}"
        )
        return calibrator

    def _strength_label(self, score: float) -> str:
        """Label a calibrated score against the impostor-derived thresholds."""
        strong_min, moderate_min = self._strength_thresholds
        if score >= strong_min:
            return "strong"
        if score >= moderate_min:
            return "moderate"
        return "weak"
