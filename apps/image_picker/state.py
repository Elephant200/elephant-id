"""Thread-safe picker state, selection validation, and export."""

from __future__ import annotations

import csv
import random
import re
import shutil
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
from loguru import logger

from elephant_id.dataset import Dataset
from elephant_id.image.transforms import apply_crop

from .catalog import PhotoCatalog, identity_is_ready
from .config import (
    CODED_ROOT,
    CROP_PREVIEW_PAD,
    FALLBACK_READY_IMAGES,
    HIGH_QUALITY_IMAGES_ROOT,
    HIGH_QUALITY_MANIFEST,
    HIGH_QUALITY_ROOT,
    MAX_PHOTOS_PER_IDENTITY_ANALYSIS,
    MAX_SELECTIONS_PER_IDENTITY,
    MIN_READY_IMAGES,
    MIN_READY_SIGHTINGS,
    MIN_SELECTIONS_PER_IDENTITY,
    QUEUE_SEED,
    SIDES,
    TARGET_DONE_IDENTITIES,
)
from .model import CandidateAnalyzer, EarCandidate, PickerModelUnavailableError

MANIFEST_FIELDS = [
    "side",
    "identity",
    "photo_identifier",
    "sighting_date",
    "source_image_path",
    "source_abs_path",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "crop_confidence",
    "exported_path",
    "exported_abs_path",
    "exported_at",
]


@dataclass(frozen=True)
class IdentityResult:
    """Cached candidate-analysis result for one identity and side."""

    side: str
    identity: str
    candidates: tuple[EarCandidate, ...]
    skipped: int
    error: str | None = None

    @property
    def status(self) -> str:
        """Return a compact status string for the frontend."""
        if self.error:
            return "error"
        if len(self.candidates) < MIN_SELECTIONS_PER_IDENTITY:
            return "insufficient"
        return "ready"


@dataclass(frozen=True)
class ReadinessCounts:
    """Unique image and sighting counts for one side."""

    ready: bool
    image_count: int
    sighting_count: int

    def to_json(self) -> dict:
        """Return the JSON representation consumed by the frontend."""
        return {
            "ready": self.ready,
            "imageCount": self.image_count,
            "sightingCount": self.sighting_count,
        }


class PickerState:
    """Own the ready-identity queue, selections, and export state."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        catalog: PhotoCatalog,
        analyzer: CandidateAnalyzer,
    ) -> None:
        """Build the scan pool and start collecting the ready-identity queue."""
        self.dataset = dataset
        self.catalog = catalog
        self.analyzer = analyzer
        self._pool = catalog.scan_pool()
        self._ready_queue: list[str] = []
        self._ready_set: set[str] = set()
        self._index = 0
        self._identity_cache: dict[tuple[str, str], IdentityResult] = {}
        self._selected: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._done: dict[str, set[str]] = {side: set() for side in SIDES}
        self._manifest_exports: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._inflight: dict[tuple[str, str], threading.Event] = {}
        self._queue_scan_current: str | None = None
        self._queue_scan_scanned = 0
        self._queue_scan_errors: list[str] = []
        self._background_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="image-picker-loader",
        )
        self._queue_loader_future: Future[None] | None = None
        self._lock = threading.RLock()
        self._manifest_lock = threading.Lock()
        self._load_manifest_state()
        self.start_background_loading()

    def view(self) -> dict:
        """Return app-level state for the frontend."""
        with self._lock:
            identity = self._current_identity_unlocked()
            queue_size = len(self._ready_queue)
            index = self._index if self._ready_queue else 0
            sides = {
                side: {
                    "queueSize": queue_size,
                    "index": index,
                    "identity": identity,
                    "doneIdentities": len(self._done[side]),
                    "targetDoneIdentities": TARGET_DONE_IDENTITIES,
                }
                for side in SIDES
            }
            return {
                "sides": sides,
                "pool": self.catalog.summary(),
                "queueScan": self._queue_scan_status_unlocked(),
                "manifestPath": str(HIGH_QUALITY_MANIFEST),
            }

    def navigate(self, side: str, delta: int) -> dict:
        """Step the shared ready-identity queue one identity in ``delta``'s direction."""
        _require_side(side)
        with self._lock:
            if self._ready_queue and delta != 0:
                direction = 1 if delta > 0 else -1
                self._index = (self._index + direction) % len(self._ready_queue)
        return self.view()

    def current_identity(self, side: str) -> str | None:
        """Return the current identity for a side."""
        _require_side(side)
        with self._lock:
            return self._current_identity_unlocked()

    def identity_payload(self, side: str, identity: str) -> dict:
        """Return candidates and selection state for one identity."""
        result = self.ensure_identity(side, identity)
        key = (side, identity)
        with self._lock:
            selected = set(self._selected[key])
            done = identity in self._done[side]
            both_done = self._identity_fully_done_unlocked(identity)
            pair = self._pair_status_unlocked(identity)
        candidates = result.candidates[:MAX_PHOTOS_PER_IDENTITY_ANALYSIS] if pair["ready"] else ()
        return {
            "side": side,
            "identity": identity,
            "status": result.status,
            "error": result.error,
            "candidateCount": len(candidates),
            "rawCandidateCount": len(result.candidates),
            "skipped": result.skipped,
            "selectedCount": len({
                candidate.photo_identifier
                for candidate in result.candidates
                if candidate.candidate_id in selected
            }),
            "done": done,
            "bothDone": both_done,
            "pairReady": pair["ready"],
            "pairStatus": pair,
            "readinessRule": readiness_rule_json(),
            "minSelections": MIN_SELECTIONS_PER_IDENTITY,
            "maxSelections": MAX_SELECTIONS_PER_IDENTITY,
            "candidates": [
                candidate.to_json(selected=candidate.candidate_id in selected)
                for candidate in candidates
            ],
        }

    def ensure_identity(self, side: str, identity: str) -> IdentityResult:
        """Analyze one side/identity pair once and cache the result."""
        _require_side(side)
        return self._compute_and_cache_identity(side, identity)

    def start_background_loading(self) -> None:
        """Start scanning the pool to collect the ready-identity queue."""
        with self._lock:
            if self._queue_loader_future is not None and not self._queue_loader_future.done():
                return
            self._queue_loader_future = self._background_executor.submit(self._build_ready_queue)

    def _build_ready_queue(self) -> None:
        """Scan the whole shuffled pool and keep every identity that clears the bar.

        Identities are analyzed in pool order; one is admitted to the ready
        queue when both sides pass ``side_readiness`` (cross-sighting diversity
        or sheer image volume).
        """
        for identity in self._pool:
            with self._lock:
                self._queue_scan_current = identity
            self._scan_identity(identity)
            with self._lock:
                self._queue_scan_scanned += 1
        with self._lock:
            self._queue_scan_current = None

    def _scan_identity(self, identity: str) -> None:
        """Analyze both sides of one identity and admit it if it clears the bar."""
        for side in SIDES:
            try:
                result = self._compute_and_cache_identity(side, identity)
            except Exception as exc:
                logger.exception(f"Image picker scan failed for {identity} {side}: {exc}")
                self._record_scan_error(f"{identity} {side}: {exc}")
                return
            if not side_readiness(result.candidates).ready:
                return
        with self._lock:
            self._admit_identity_unlocked(identity)

    def _admit_identity_unlocked(self, identity: str) -> None:
        """Append a ready identity to the queue if it is not already present."""
        if identity in self._ready_set:
            return
        if not self._pair_status_unlocked(identity)["ready"]:
            return
        self._ready_queue.append(identity)
        self._ready_set.add(identity)

    def _record_scan_error(self, message: str) -> None:
        """Record a recent scan error, keeping only the latest few."""
        with self._lock:
            self._queue_scan_errors.append(message)
            self._queue_scan_errors = self._queue_scan_errors[-10:]

    def _compute_and_cache_identity(self, side: str, identity: str) -> IdentityResult:
        """Compute and cache one identity result, deduping concurrent callers."""
        key = (side, identity)
        with self._lock:
            cached = self._identity_cache.get(key)
            if cached is not None:
                return cached
            event = self._inflight.get(key)
            owner = event is None
            if owner:
                event = threading.Event()
                self._inflight[key] = event

        if not owner:
            event.wait()
            with self._lock:
                cached = self._identity_cache.get(key)
            if cached is not None:
                return cached
            return self._compute_and_cache_identity(side, identity)

        try:
            result = self._compute_identity(side, identity)
            with self._lock:
                cached = self._identity_cache.get(key)
                if cached is None:
                    self._identity_cache[key] = result
                    self._hydrate_selection_from_manifest_unlocked(key, result.candidates)
                    cached = result
            return cached
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            event.set()

    def _compute_identity(self, side: str, identity: str) -> IdentityResult:
        """Analyze one identity without reading or writing the result cache."""
        with self._lock:
            pinned_identifiers = set(self._manifest_exports[(side, identity)])
        records = sampled_identity_records(
            self.catalog.by_identity.get(identity, []),
            identity=identity,
            side=side,
            pinned_identifiers=pinned_identifiers,
        )
        candidates: list[EarCandidate] = []
        skipped = 0
        error: str | None = None
        try:
            for record in records:
                try:
                    photo = record.to_photo()
                    candidates.extend(
                        self.analyzer.candidates_for_photo(
                            photo,
                            side=side,
                            identity=identity,
                            date=record.date,
                            image_path=str(record.image_path).replace("\\", "/"),
                            seek_code=record.seek_code,
                        )
                    )
                except PickerModelUnavailableError:
                    raise
                except Exception as exc:
                    logger.warning(
                        f"Skipped picker photo {record.identifier} for {identity} {side}: {exc}"
                    )
                    skipped += 1
        except PickerModelUnavailableError as exc:
            error = str(exc)
        except Exception as exc:
            logger.exception(f"Image picker analysis failed for {identity} {side}: {exc}")
            error = f"Analysis failed: {exc}"

        result = IdentityResult(
            side=side,
            identity=identity,
            candidates=tuple(candidates),
            skipped=skipped,
            error=error,
        )
        return result

    def _current_identity_unlocked(self) -> str | None:
        """Return the current ready-queue identity, or ``None`` if none yet."""
        if not self._ready_queue:
            return None
        return self._ready_queue[self._index % len(self._ready_queue)]

    def _identity_fully_done_unlocked(self, identity: str) -> bool:
        """Return whether an identity is exported on both sides."""
        return all(identity in self._done[side] for side in SIDES)

    def _pair_status_unlocked(self, identity: str) -> dict:
        """Return whether an identity has enough cached candidates on both sides."""
        counts: dict[str, ReadinessCounts | None] = {}
        cached_sides = 0
        for side in SIDES:
            result = self._identity_cache.get((side, identity))
            if result is None:
                counts[side] = None
                continue
            cached_sides += 1
            counts[side] = side_readiness(result.candidates)
        ready = all(
            counts[side] is not None and counts[side].ready
            for side in SIDES
        )
        loading = cached_sides < len(SIDES)
        return {
            "ready": ready,
            "loading": loading,
            "left": counts["left"].to_json() if counts["left"] is not None else None,
            "right": counts["right"].to_json() if counts["right"] is not None else None,
        }

    def _queue_scan_status_unlocked(self) -> dict:
        """Return ready-queue collection progress."""
        running = (
            self._queue_loader_future is not None
            and not self._queue_loader_future.done()
        )
        future_error = None
        if self._queue_loader_future is not None and self._queue_loader_future.done():
            exception = self._queue_loader_future.exception()
            if exception is not None:
                future_error = str(exception)
        return {
            "running": running,
            "ready": len(self._ready_queue),
            "scanned": self._queue_scan_scanned,
            "poolSize": len(self._pool),
            "current": self._queue_scan_current,
            "errors": list(self._queue_scan_errors),
            "futureError": future_error,
            "readinessRule": readiness_rule_json(),
        }

    def toggle_selection(
        self,
        *,
        side: str,
        identity: str,
        candidate_id: str,
        selected: bool,
    ) -> dict:
        """Select or unselect a candidate, enforcing the per-identity photo limit."""
        result = self.ensure_identity(side, identity)
        candidate = _candidate_by_id(result.candidates, candidate_id)
        if candidate is None:
            raise ValueError("Unknown candidate")

        key = (side, identity)
        with self._lock:
            selected_ids = self._selected[key]
            if selected:
                selected_photos = {
                    known.photo_identifier
                    for known in result.candidates
                    if known.candidate_id in selected_ids
                }
                if (
                    candidate.photo_identifier not in selected_photos
                    and len(selected_photos) >= MAX_SELECTIONS_PER_IDENTITY
                ):
                    raise ValueError(
                        f"Choose at most {MAX_SELECTIONS_PER_IDENTITY} images; "
                        "uncheck one before adding another."
                    )
                for known in result.candidates:
                    if (
                        known.photo_identifier == candidate.photo_identifier
                        and known.candidate_id != candidate.candidate_id
                    ):
                        selected_ids.discard(known.candidate_id)
                selected_ids.add(candidate.candidate_id)
            else:
                selected_ids.discard(candidate.candidate_id)
        return self.identity_payload(side, identity)

    def mark_done(self, side: str, identity: str) -> dict:
        """Export selected originals, overwriting any prior export for this side."""
        result = self.ensure_identity(side, identity)
        key = (side, identity)
        with self._lock:
            selected_ids = set(self._selected[key])
        selected_candidates = [
            candidate for candidate in result.candidates
            if candidate.candidate_id in selected_ids
        ]
        unique_identifiers = {candidate.photo_identifier for candidate in selected_candidates}
        if not MIN_SELECTIONS_PER_IDENTITY <= len(unique_identifiers) <= MAX_SELECTIONS_PER_IDENTITY:
            raise ValueError(
                f"Select {MIN_SELECTIONS_PER_IDENTITY} to {MAX_SELECTIONS_PER_IDENTITY} "
                "distinct images before marking done."
            )

        self._export_candidates(side, identity, selected_candidates)
        with self._lock:
            self._done[side].add(identity)
        return {"state": self.view(), "identity": self.identity_payload(side, identity)}

    def crop_jpeg(self, side: str, identity: str, candidate_id: str) -> bytes:
        """Render one accepted crop preview as JPEG bytes."""
        result = self.ensure_identity(side, identity)
        candidate = _candidate_by_id(result.candidates, candidate_id)
        if candidate is None:
            raise ValueError("Unknown candidate")
        record = self.catalog.by_identifier[candidate.photo_identifier]
        image = self.dataset.read_image(record.to_photo())
        crop = apply_crop(image, candidate.crop_xyxy, pad=CROP_PREVIEW_PAD)
        ok, encoded = cv2.imencode(".jpg", crop)
        if not ok:
            raise RuntimeError("Could not encode crop preview")
        return encoded.tobytes()

    def _load_manifest_state(self) -> None:
        """Load already exported identities from the existing manifest."""
        if not HIGH_QUALITY_MANIFEST.is_file():
            return
        try:
            with HIGH_QUALITY_MANIFEST.open(newline="") as file:
                for row in csv.DictReader(file):
                    side = row.get("side", "")
                    identity = row.get("identity", "")
                    identifier = row.get("photo_identifier", "")
                    if side not in SIDES or not identity or not identifier:
                        continue
                    self._manifest_exports[(side, identity)].add(identifier)
        except OSError:
            return
        for (side, identity), identifiers in self._manifest_exports.items():
            if len(identifiers) >= MIN_SELECTIONS_PER_IDENTITY:
                self._done[side].add(identity)

    def _hydrate_selection_from_manifest_unlocked(
        self,
        key: tuple[str, str],
        candidates: tuple[EarCandidate, ...],
    ) -> None:
        """Restore selected candidate IDs from completed manifest rows."""
        if self._selected[key]:
            return
        exported = self._manifest_exports.get(key)
        if not exported:
            return
        for candidate in candidates:
            if candidate.photo_identifier in exported:
                self._selected[key].add(candidate.candidate_id)

    def _export_candidates(
        self,
        side: str,
        identity: str,
        candidates: list[EarCandidate],
    ) -> None:
        """Overwrite the exported originals and manifest rows for one side/identity.

        Replaces any prior export for the same side and identity: previously
        exported images that are no longer selected are deleted and the manifest
        is rewritten so it always matches the current selection.
        """
        exported_at = datetime.now(UTC).isoformat()
        safe_identity = safe_path_component(identity)
        image_dir = HIGH_QUALITY_IMAGES_ROOT / side / safe_identity
        with self._manifest_lock:
            image_dir.mkdir(parents=True, exist_ok=True)
            HIGH_QUALITY_ROOT.mkdir(parents=True, exist_ok=True)

            other_rows, prior_rows = self._partition_manifest_rows(side, identity)
            self._delete_exported_files(prior_rows)

            new_rows = []
            exported_identifiers: set[str] = set()
            for candidate in candidates:
                if candidate.photo_identifier in exported_identifiers:
                    continue
                record = self.catalog.by_identifier[candidate.photo_identifier]
                photo = record.to_photo()
                source = self.dataset.path_for(photo)
                exported = image_dir / source.name
                if exported.exists() and exported.resolve() != source.resolve():
                    exported = image_dir / f"{candidate.photo_identifier}{source.suffix}"
                shutil.copy2(source, exported)
                rel_exported = exported.relative_to(HIGH_QUALITY_ROOT)
                x1, y1, x2, y2 = candidate.crop_xyxy
                new_rows.append(
                    {
                        "side": side,
                        "identity": identity,
                        "photo_identifier": candidate.photo_identifier,
                        "sighting_date": candidate.date,
                        "source_image_path": str(record.image_path).replace("\\", "/"),
                        "source_abs_path": str((CODED_ROOT / record.image_path).resolve()),
                        "crop_x1": f"{x1:.3f}",
                        "crop_y1": f"{y1:.3f}",
                        "crop_x2": f"{x2:.3f}",
                        "crop_y2": f"{y2:.3f}",
                        "crop_confidence": f"{candidate.confidence:.6f}",
                        "exported_path": str(rel_exported).replace("\\", "/"),
                        "exported_abs_path": str(exported.resolve()),
                        "exported_at": exported_at,
                    }
                )
                exported_identifiers.add(candidate.photo_identifier)

            self._write_manifest(other_rows + new_rows)
        with self._lock:
            self._manifest_exports[(side, identity)] = exported_identifiers

    def _partition_manifest_rows(
        self,
        side: str,
        identity: str,
    ) -> tuple[list[dict], list[dict]]:
        """Split manifest rows into ``(other_rows, rows_for_this_side_identity)``."""
        other_rows: list[dict] = []
        matching_rows: list[dict] = []
        if not HIGH_QUALITY_MANIFEST.is_file():
            return other_rows, matching_rows
        with HIGH_QUALITY_MANIFEST.open(newline="") as file:
            for row in csv.DictReader(file):
                if row.get("side") == side and row.get("identity") == identity:
                    matching_rows.append(row)
                else:
                    other_rows.append(row)
        return other_rows, matching_rows

    def _delete_exported_files(self, rows: list[dict]) -> None:
        """Delete exported image files for manifest rows, scoped to the export root."""
        root = HIGH_QUALITY_ROOT.resolve()
        for row in rows:
            raw = row.get("exported_abs_path") or ""
            if raw:
                path = Path(raw)
            elif row.get("exported_path"):
                path = HIGH_QUALITY_ROOT / row["exported_path"]
            else:
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                logger.warning(f"Refusing to delete export outside high-quality root: {resolved}")
                continue
            if resolved.is_file():
                try:
                    resolved.unlink()
                except OSError as exc:
                    logger.warning(f"Could not delete stale export {resolved}: {exc}")

    def _write_manifest(self, rows: list[dict]) -> None:
        """Rewrite the manifest file with the provided rows."""
        with HIGH_QUALITY_MANIFEST.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def safe_path_component(value: str) -> str:
    """Return a conservative filename component for an identity."""
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip(" .")
    return cleaned or "identity"


def sampled_identity_records(
    records,
    *,
    identity: str,
    side: str,
    pinned_identifiers: set[str],
):
    """Return up to the configured number of records, preserving prior exports."""
    records = list(records)
    if len(records) <= MAX_PHOTOS_PER_IDENTITY_ANALYSIS:
        return records
    pinned = [
        record for record in records
        if record.identifier in pinned_identifiers
    ]
    pinned_ids = {record.identifier for record in pinned}
    sample_size = max(0, MAX_PHOTOS_PER_IDENTITY_ANALYSIS - len(pinned))
    sample_pool = [
        record for record in records
        if record.identifier not in pinned_ids
    ]
    rng = random.Random(f"{QUEUE_SEED}:records:{side}:{identity}")
    sampled = pinned + rng.sample(sample_pool, min(sample_size, len(sample_pool)))
    return sorted(sampled, key=lambda record: (record.date, record.identifier))


def side_readiness(candidates: tuple[EarCandidate, ...]) -> ReadinessCounts:
    """Return whether accepted candidates pass the diversity-or-volume rule."""
    image_count = len({candidate.photo_identifier for candidate in candidates})
    sighting_count = len({candidate.date for candidate in candidates})
    return ReadinessCounts(
        ready=identity_is_ready(
            image_count=image_count,
            sighting_count=sighting_count,
        ),
        image_count=image_count,
        sighting_count=sighting_count,
    )


def readiness_rule_json() -> dict:
    """Return the configured picker readiness rule for frontend copy."""
    return {
        "minSightings": MIN_READY_SIGHTINGS,
        "minImages": MIN_READY_IMAGES,
        "fallbackImages": FALLBACK_READY_IMAGES,
    }


def _candidate_by_id(
    candidates: tuple[EarCandidate, ...],
    candidate_id: str,
) -> EarCandidate | None:
    """Find one candidate by ID."""
    for candidate in candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def _require_side(side: str) -> None:
    """Validate a side value."""
    if side not in SIDES:
        raise ValueError(f"Invalid side: {side}")
