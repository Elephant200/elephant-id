"""In-memory picker state: eligibility scan, candidate cache, and picks."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
from loguru import logger

from elephant_id.dataset import Dataset
from elephant_id.domain import Sighting
from elephant_id.image.transforms import apply_crop

from .analysis import CandidateAnalyzer, EarCandidate, SightingCandidates
from .catalog import PhotoCatalog
from .config import (
    HIGH_QUALITY_MANIFEST,
    MIN_QUALIFYING_SIGHTINGS,
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
        """Scan every elephant once, admitting those with enough qualifiers."""
        for identity in self.catalog.elephants():
            with self._lock:
                self._scan_current = identity
            try:
                eligible = self._is_eligible(identity)
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

    def _is_eligible(self, identity: str) -> bool:
        """Whether an elephant has enough qualifying sightings (stops early)."""
        sightings = self.catalog.sightings(identity)
        if len(sightings) < MIN_QUALIFYING_SIGHTINGS:
            return False
        qualifying = 0
        for sighting in sightings:
            if self._sighting_candidates(sighting).qualifies(QUALITY_THRESHOLD):
                qualifying += 1
                if qualifying >= MIN_QUALIFYING_SIGHTINGS:
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
        """Return the eligible-elephant list and scan progress."""
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
        elephants = [
            {"identity": identity, "pickedCount": self._picked_count(identity)}
            for identity in eligible
        ]
        return {
            "elephants": elephants,
            "scan": scan,
            "manifestPath": str(HIGH_QUALITY_MANIFEST),
        }

    def elephant_view(self, identity: str) -> dict:
        """Return one elephant's qualifying sightings with ranked candidates."""
        sightings = self.catalog.sightings(identity)
        if not sightings:
            raise KeyError(identity)
        picks = self.manifest.picks_for_identity(identity)
        payloads = []
        for sighting in sightings:
            candidates = self._sighting_candidates(sighting)
            if candidates.qualifies(QUALITY_THRESHOLD):
                payloads.append(self._sighting_payload(candidates, picks))
        return {
            "identity": identity,
            "minQualifying": MIN_QUALIFYING_SIGHTINGS,
            "qualifyingCount": len(payloads),
            "sightings": payloads,
        }

    def _sighting_payload(
        self,
        candidates: SightingCandidates,
        picks: dict[tuple[str, str], str],
    ) -> dict:
        """Return one qualifying sighting's candidates split by side."""
        sides = {}
        for side in SIDES:
            picked_photo = picks.get((side, candidates.sighting_date))
            sides[side] = [
                candidate.to_json(picked=candidate.photo_identifier == picked_photo)
                for candidate in candidates.side(side)
            ]
        return {
            "sightingId": candidates.sighting_id,
            "sightingDate": candidates.sighting_date,
            "left": sides["left"],
            "right": sides["right"],
        }

    def _picked_count(self, identity: str) -> int:
        """Return how many (side, sighting) picks exist for an elephant."""
        return len(self.manifest.picks_for_identity(identity))

    # --- mutations -------------------------------------------------------
    def record_pick(
        self,
        *,
        identity: str,
        sighting_date: str,
        side: str,
        candidate_id: str,
    ) -> dict:
        """Persist a pick and return the sighting's refreshed payload."""
        candidate = self._find_candidate(identity, sighting_date, side, candidate_id)
        self.manifest.record_pick(candidate)
        candidates = self._sighting_candidates(self._sighting(identity, sighting_date))
        picks = self.manifest.picks_for_identity(identity)
        return self._sighting_payload(candidates, picks)

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
        crop = apply_crop(self.dataset.read_image(photo), candidate.crop_xyxy)
        ok, encoded = cv2.imencode(".jpg", crop)
        if not ok:
            raise RuntimeError(f"Could not encode crop for {candidate.photo_identifier}")
        return encoded.tobytes()

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
