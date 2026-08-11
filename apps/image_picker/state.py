"""In-memory picker state: eligibility scan, candidate cache, and picks."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

from loguru import logger

from elephant_id.dataset import Dataset
from elephant_id.domain import Sighting

from .analysis import (
    CandidateAnalyzer,
    EarCandidate,
    SightingCandidates,
    encode_crop_jpeg,
)
from .catalog import PhotoCatalog
from .config import (
    HIGH_QUALITY_MANIFEST,
    MAX_SELECTED_SIGHTINGS,
    MIN_QUALIFYING_SIGHTINGS,
    MIN_SELECTED_SIGHTINGS,
    QUALITY_THRESHOLD,
    SIDES,
)
from .manifest import ManifestStore


class PickerState:
    """Own the eligibility scan, the candidate cache, and pick recording."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        catalog: PhotoCatalog,
        analyzer: CandidateAnalyzer,
        manifest: ManifestStore,
    ) -> None:
        """Start a single background worker scanning for eligible elephants."""
        self.dataset = dataset
        self.catalog = catalog
        self.analyzer = analyzer
        self.manifest = manifest

        self._lock = threading.RLock()
        self._sighting_cache: dict[str, SightingCandidates] = {}
        self._eligible: list[str] = []
        self._eligible_set: set[str] = set()
        self._scanned = 0
        self._scan_current: str | None = None

        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="image-picker-scan"
        )
        self._scan_future: Future[None] = self._executor.submit(self._scan)

    # --- eligibility scan ------------------------------------------------
    def _scan(self) -> None:
        """Scan every elephant once, admitting those with enough qualifiers.

        The manifest is read once up front so prior-session picks can be
        grandfathered without re-reading the file per elephant.
        """
        picks_by_identity = self.manifest.picks_by_identity()
        for identity in self.catalog.elephants():
            with self._lock:
                self._scan_current = identity
            try:
                eligible = self._is_eligible(
                    identity, picks_by_identity.get(identity, {})
                )
            except Exception as error:
                logger.exception(f"Eligibility scan failed for {identity}: {error}")
                eligible = False
            with self._lock:
                if eligible and identity not in self._eligible_set:
                    self._eligible.append(identity)
                    self._eligible_set.add(identity)
                self._scanned += 1
        with self._lock:
            self._scan_current = None
        logger.info(f"Eligibility scan complete: {len(self._eligible)} elephants")

    def _is_eligible(
        self, identity: str, picks: dict[tuple[str, str], str]
    ) -> bool:
        """Whether an elephant has enough eligible sightings (stops early).

        A sighting counts toward ``MIN_QUALIFYING_SIGHTINGS`` when it qualifies
        under the current heuristic or is "grandfathered" -- already represented
        by a manifest pick from a prior session -- so past review work survives a
        heuristic change. Grandfathered sightings are checked first to avoid
        analyzing them.
        """
        sightings = self.catalog.sightings(identity)
        if len(sightings) < MIN_QUALIFYING_SIGHTINGS:
            return False
        grandfathered = self._grandfathered_dates(picks)
        eligible = 0
        for sighting in sightings:
            if (
                sighting.sighting_date.isoformat() in grandfathered
                or self._sighting_candidates(sighting).qualifies(QUALITY_THRESHOLD)
            ):
                eligible += 1
                if eligible >= MIN_QUALIFYING_SIGHTINGS:
                    return True
        return False

    def _sighting_candidates(self, sighting: Sighting) -> SightingCandidates:
        """Return cached candidates for a sighting, analyzing on first use."""
        with self._lock:
            cached = self._sighting_cache.get(sighting.sighting_id)
        if cached is not None:
            return cached
        result = self.analyzer.analyze_sighting(sighting)
        with self._lock:
            self._sighting_cache[sighting.sighting_id] = result
        return result

    # --- views -----------------------------------------------------------
    def elephants_view(self) -> dict:
        """Return the eligible-elephant list and scan/review progress."""
        with self._lock:
            eligible = list(self._eligible)
            running = not self._scan_future.done()
            scan = {
                "running": running,
                "scanned": self._scanned,
                "total": len(self.catalog.elephants()),
                "current": self._scan_current,
                "eligible": len(eligible),
            }
        picks_by_identity = self.manifest.picks_by_identity()
        elephants = []
        done_count = 0
        for identity in eligible:
            summary = self._selection_summary(picks_by_identity.get(identity, {}))
            if summary["done"]:
                done_count += 1
            elephants.append({"identity": identity, **summary})
        return {
            "elephants": elephants,
            "scan": scan,
            "doneCount": done_count,
            "manifestPath": str(HIGH_QUALITY_MANIFEST),
        }

    def elephant_view(self, identity: str) -> dict:
        """Return one elephant's qualifying sightings with ranked candidates."""
        sightings = self.catalog.sightings(identity)
        if not sightings:
            raise KeyError(identity)
        picks = self.manifest.picks_for_identity(identity)
        grandfathered = self._grandfathered_dates(picks)
        payloads = []
        for sighting in sightings:
            candidates = self._sighting_candidates(sighting)
            if (
                candidates.qualifies(QUALITY_THRESHOLD)
                or candidates.sighting_date in grandfathered
            ):
                payloads.append(self._sighting_payload(candidates, picks))
        return {
            "identity": identity,
            "minQualifying": MIN_QUALIFYING_SIGHTINGS,
            "qualifyingCount": len(payloads),
            "selection": self._selection_summary(picks),
            "sightings": payloads,
        }

    def _sighting_payload(
        self,
        candidates: SightingCandidates,
        picks: dict[tuple[str, str], str],
    ) -> dict:
        """Return one qualifying sighting's candidates split by side."""
        sides = {}
        picked_sides = set()
        for side in SIDES:
            picked_photo = picks.get((side, candidates.sighting_date))
            if picked_photo is not None:
                picked_sides.add(side)
            sides[side] = [
                candidate.to_json(picked=candidate.photo_identifier == picked_photo)
                for candidate in candidates.side(side)
            ]
        return {
            "sightingId": candidates.sighting_id,
            "sightingDate": candidates.sighting_date,
            "selected": bool(picked_sides),
            "complete": self._is_complete(picked_sides),
            "left": sides["left"],
            "right": sides["right"],
        }

    @staticmethod
    def _grandfathered_dates(picks: dict[tuple[str, str], str]) -> set[str]:
        """Sighting dates already represented by a manifest pick."""
        return {sighting_date for _side, sighting_date in picks}

    @staticmethod
    def _sides_by_date(picks: dict[tuple[str, str], str]) -> dict[str, set[str]]:
        """Group an elephant's picks into the set of picked sides per sighting."""
        grouped: dict[str, set[str]] = {}
        for side, sighting_date in picks:
            grouped.setdefault(sighting_date, set()).add(side)
        return grouped

    @staticmethod
    def _is_complete(picked_sides: set[str]) -> bool:
        """Whether both ear sides have been picked for a sighting."""
        return len(picked_sides) == len(SIDES)

    def _selection_summary(self, picks: dict[tuple[str, str], str]) -> dict:
        """Summarize an elephant's selection progress from its manifest picks.

        A sighting is "selected" once it has any pick and "complete" once it has
        both sides. An elephant is "done" when it has between the minimum and
        maximum sightings complete with no partially picked sighting left over.
        """
        sides_by_date = self._sides_by_date(picks)
        selected_count = len(sides_by_date)
        complete_count = sum(
            1 for sides in sides_by_date.values() if self._is_complete(sides)
        )
        done = (
            selected_count == complete_count
            and MIN_SELECTED_SIGHTINGS <= complete_count <= MAX_SELECTED_SIGHTINGS
        )
        return {
            "selectedCount": selected_count,
            "completeCount": complete_count,
            "minSightings": MIN_SELECTED_SIGHTINGS,
            "maxSightings": MAX_SELECTED_SIGHTINGS,
            "done": done,
        }

    # --- mutations -------------------------------------------------------
    def record_pick(
        self,
        *,
        identity: str,
        sighting_date: str,
        side: str,
        candidate_id: str,
    ) -> dict:
        """Persist a pick and return the sighting payload plus review progress.

        Raises:
            ValueError: If the pick would select more than ``MAX_SELECTED_SIGHTINGS``
                sightings for the elephant.
        """
        candidate = self._find_candidate(identity, sighting_date, side, candidate_id)
        picks = self.manifest.picks_for_identity(identity)
        selected_dates = set(self._sides_by_date(picks))
        if (
            sighting_date not in selected_dates
            and len(selected_dates) >= MAX_SELECTED_SIGHTINGS
        ):
            logger.warning(
                f"Rejected pick for {identity} {sighting_date} {side}: already at "
                f"the {MAX_SELECTED_SIGHTINGS}-sighting limit"
            )
            raise ValueError(
                f"Cannot select more than {MAX_SELECTED_SIGHTINGS} sightings for "
                f"this elephant; re-pick within an already-selected sighting instead"
            )
        self.manifest.record_pick(candidate)
        candidates = self._sighting_candidates(self._sighting(identity, sighting_date))
        picks = self.manifest.picks_for_identity(identity)
        return {
            "sighting": self._sighting_payload(candidates, picks),
            "selection": self._selection_summary(picks),
        }

    def crop_jpeg(
        self,
        *,
        identity: str,
        sighting_date: str,
        side: str,
        candidate_id: str,
    ) -> bytes:
        """Render one candidate's ear crop as JPEG bytes."""
        candidate = self._find_candidate(identity, sighting_date, side, candidate_id)
        photo = self.dataset.get_photo(candidate.photo_identifier)
        return encode_crop_jpeg(self.dataset.read_image(photo), candidate.crop_xyxy)

    # --- lookup helpers --------------------------------------------------
    def _sighting(self, identity: str, sighting_date: str) -> Sighting:
        """Resolve a sighting from an elephant name and ISO date."""
        sighting = self.catalog.sighting_by_id.get(f"{identity}_{sighting_date}")
        if sighting is None:
            raise KeyError(f"Unknown sighting: {identity}_{sighting_date}")
        return sighting

    def _find_candidate(
        self,
        identity: str,
        sighting_date: str,
        side: str,
        candidate_id: str,
    ) -> EarCandidate:
        """Find a candidate by id within a sighting side."""
        if side not in SIDES:
            raise ValueError(f"Invalid side: {side}")
        candidates = self._sighting_candidates(self._sighting(identity, sighting_date))
        for candidate in candidates.side(side):
            if candidate.candidate_id == candidate_id:
                return candidate
        raise KeyError(f"Unknown candidate: {candidate_id}")
