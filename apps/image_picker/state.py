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

import cv2
from loguru import logger

from elephant_id.dataset import Dataset
from elephant_id.image.transforms import apply_crop

from .catalog import PhotoCatalog
from .config import (
    CODED_ROOT,
    CROP_PREVIEW_PAD,
    HIGH_QUALITY_IMAGES_ROOT,
    HIGH_QUALITY_MANIFEST,
    HIGH_QUALITY_ROOT,
    MAX_PHOTOS_PER_IDENTITY_ANALYSIS,
    MAX_SELECTIONS_PER_IDENTITY,
    MIN_SELECTIONS_PER_IDENTITY,
    MIN_SIDE_CANDIDATES,
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


class PickerState:
    """Own the in-memory side queues and selections."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        catalog: PhotoCatalog,
        analyzer: CandidateAnalyzer,
    ) -> None:
        """Initialize queue state for both sides."""
        self.dataset = dataset
        self.catalog = catalog
        self.analyzer = analyzer
        shared_queue = catalog.shared_queue()
        self.queues = {side: list(shared_queue) for side in SIDES}
        self.indices = {side: 0 for side in SIDES}
        self._identity_cache: dict[tuple[str, str], IdentityResult] = {}
        self._selected: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._done: dict[str, set[str]] = {side: set() for side in SIDES}
        self._manifest_exports: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._prefetch_futures: dict[tuple[str, str], Future[IdentityResult]] = {}
        self._queue_scan_current: str | None = None
        self._queue_scan_processed = 0
        self._queue_scan_errors: list[str] = []
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="image-picker-prefetch",
        )
        self._queue_loader_future: Future[None] | None = None
        self._lock = threading.RLock()
        self._load_manifest_state()
        self.start_background_loading()

    def view(self) -> dict:
        """Return app-level state for the frontend."""
        with self._lock:
            sides = {}
            for side, queue in self.queues.items():
                index = self.indices[side]
                identity = self._current_display_identity_unlocked(side)
                sides[side] = {
                    "queueSize": len(queue),
                    "index": index,
                    "identity": identity,
                    "doneIdentities": len(self._done[side]),
                    "targetDoneIdentities": TARGET_DONE_IDENTITIES,
                    "prefetch": self._prefetch_status_unlocked(side, count=2),
                }
            return {
                "sides": sides,
                "pool": self.catalog.summary(),
                "queueScan": self._queue_scan_status_unlocked(),
                "manifestPath": str(HIGH_QUALITY_MANIFEST),
            }

    def navigate(self, side: str, delta: int) -> dict:
        """Move the shared identity queue by ``delta`` ready identities."""
        _require_side(side)
        with self._lock:
            queue = self.queues[side]
            if queue:
                next_index = self._navigation_index_unlocked(side, delta)
                for queue_side in SIDES:
                    self.indices[queue_side] = next_index
        return self.view()

    def current_identity(self, side: str) -> str | None:
        """Return the current identity for a side."""
        _require_side(side)
        with self._lock:
            return self._current_display_identity_unlocked(side)

    def identity_payload(self, side: str, identity: str) -> dict:
        """Return candidates and selection state for one identity."""
        result = self.ensure_identity(side, identity)
        self.prefetch_next_for_all_sides(side, identity, count=2)
        key = (side, identity)
        with self._lock:
            selected = set(self._selected[key])
            done = identity in self._done[side]
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
            "pairReady": pair["ready"],
            "pairStatus": pair,
            "minSideCandidates": MIN_SIDE_CANDIDATES,
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
        key = (side, identity)
        with self._lock:
            cached = self._identity_cache.get(key)
            if cached is not None:
                return cached
            future = self._prefetch_futures.get(key)

        if future is not None:
            result = future.result()
            with self._lock:
                self._prefetch_futures.pop(key, None)
                cached = self._identity_cache.get(key)
                if cached is not None:
                    return cached
                self._identity_cache[key] = result
                self._hydrate_selection_from_manifest_unlocked(key, result.candidates)
            return result

        return self._compute_and_cache_identity(side, identity)

    def prefetch_next(self, side: str, identity: str, count: int = 2) -> None:
        """Schedule analysis for the next identities on the same side."""
        _require_side(side)
        with self._lock:
            identities = self._next_identities_unlocked(side, identity, count)
            for next_identity in identities:
                key = (side, next_identity)
                if key in self._identity_cache or key in self._prefetch_futures:
                    continue
                future = self._prefetch_executor.submit(
                    self._compute_identity,
                    side,
                    next_identity,
                )
                self._prefetch_futures[key] = future
                future.add_done_callback(
                    lambda done_future, done_key=key: self._store_prefetch_result(
                        done_key,
                        done_future,
                    )
                )

    def prefetch_next_for_all_sides(
        self,
        active_side: str,
        active_identity: str,
        count: int = 2,
    ) -> None:
        """Schedule next-identity analysis for left and right side queues."""
        _require_side(active_side)
        self.prefetch_next(active_side, active_identity, count=count)
        with self._lock:
            other_sides = [side for side in SIDES if side != active_side]
            other_identities = {
                side: self.queues[side][self.indices[side]]
                for side in other_sides
                if self.queues[side]
            }
        for side, identity in other_identities.items():
            self.prefetch_next(side, identity, count=count)

    def start_background_loading(self) -> None:
        """Start continuously warming candidate caches for the shared queue."""
        with self._lock:
            if self._queue_loader_future is not None and not self._queue_loader_future.done():
                return
            self._queue_loader_future = self._prefetch_executor.submit(self._load_shared_queue)

    def _load_shared_queue(self) -> None:
        """Analyze left and right candidates for every identity in queue order."""
        queue = self.queues["left"]
        for index, identity in enumerate(queue, start=1):
            with self._lock:
                self._queue_scan_current = identity
                self._queue_scan_processed = max(self._queue_scan_processed, index - 1)
            for side in SIDES:
                key = (side, identity)
                with self._lock:
                    if key in self._identity_cache:
                        continue
                try:
                    self._compute_and_cache_identity(side, identity)
                except Exception as exc:
                    logger.exception(
                        f"Image picker queue scan failed for {identity} {side}: {exc}"
                    )
                    result = IdentityResult(
                        side=side,
                        identity=identity,
                        candidates=(),
                        skipped=0,
                        error=f"Queue scan failed: {exc}",
                    )
                    with self._lock:
                        self._identity_cache[key] = result
                        self._queue_scan_errors.append(f"{identity} {side}: {exc}")
                        self._queue_scan_errors = self._queue_scan_errors[-10:]
            with self._lock:
                self._queue_scan_processed = index
        with self._lock:
            self._queue_scan_current = None

    def _compute_and_cache_identity(self, side: str, identity: str) -> IdentityResult:
        """Synchronously compute and cache one identity result."""
        result = self._compute_identity(side, identity)
        key = (side, identity)
        with self._lock:
            cached = self._identity_cache.get(key)
            if cached is not None:
                return cached
            self._identity_cache[key] = result
            self._hydrate_selection_from_manifest_unlocked(key, result.candidates)
        return result

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

    def _store_prefetch_result(
        self,
        key: tuple[str, str],
        future: Future[IdentityResult],
    ) -> None:
        """Move one completed prefetch result into the identity cache."""
        try:
            result = future.result()
        except Exception as exc:
            side, identity = key
            result = IdentityResult(
                side=side,
                identity=identity,
                candidates=(),
                skipped=0,
                error=f"Prefetch failed: {exc}",
            )
        with self._lock:
            self._prefetch_futures.pop(key, None)
            if key in self._identity_cache:
                return
            self._identity_cache[key] = result
            self._hydrate_selection_from_manifest_unlocked(key, result.candidates)

    def _next_identities_unlocked(
        self,
        side: str,
        identity: str,
        count: int,
    ) -> list[str]:
        """Return the next ``count`` identities after ``identity`` in a side queue."""
        queue = self.queues[side]
        if not queue or count <= 0:
            return []
        try:
            start = queue.index(identity)
        except ValueError:
            start = self.indices[side]
        return [queue[(start + offset) % len(queue)] for offset in range(1, count + 1)]

    def _prefetch_status_unlocked(self, side: str, count: int) -> dict:
        """Return cache/prefetch status for the identities after the current one."""
        queue = self.queues[side]
        if not queue:
            return {"cached": 0, "running": 0}
        identity = queue[self.indices[side]]
        cached = 0
        running = 0
        for next_identity in self._next_identities_unlocked(side, identity, count):
            key = (side, next_identity)
            if key in self._identity_cache:
                cached += 1
            elif key in self._prefetch_futures:
                running += 1
        return {"cached": cached, "running": running}

    def _current_display_identity_unlocked(self, side: str) -> str | None:
        """Return the current identity without mutating navigation state."""
        queue = self.queues[side]
        if not queue:
            return None
        return queue[self.indices[side]]

    def _navigation_index_unlocked(self, side: str, delta: int) -> int:
        """Return the next index in ``delta`` direction, preferring ready pairs."""
        queue = self.queues[side]
        if not queue:
            return 0
        if delta == 0:
            return self.indices[side]
        direction = 1 if delta > 0 else -1
        start = self.indices[side]
        for offset in range(1, len(queue) + 1):
            candidate_index = (start + direction * offset) % len(queue)
            if self._pair_status_unlocked(queue[candidate_index])["ready"]:
                return candidate_index
        return (start + direction) % len(queue)

    def _pair_status_unlocked(self, identity: str) -> dict:
        """Return whether an identity has enough cached candidates on both sides."""
        counts: dict[str, int | None] = {}
        cached_sides = 0
        for side in SIDES:
            result = self._identity_cache.get((side, identity))
            if result is None:
                counts[side] = None
                continue
            cached_sides += 1
            counts[side] = len(result.candidates)
        ready = all(
            isinstance(counts[side], int) and counts[side] >= MIN_SIDE_CANDIDATES
            for side in SIDES
        )
        loading = cached_sides < len(SIDES)
        return {
            "ready": ready,
            "loading": loading,
            "leftCount": counts["left"],
            "rightCount": counts["right"],
        }

    def _queue_scan_status_unlocked(self) -> dict:
        """Return background queue loading progress."""
        queue = self.queues["left"]
        pair_cached = 0
        pair_ready = 0
        for identity in queue:
            pair = self._pair_status_unlocked(identity)
            if not pair["loading"]:
                pair_cached += 1
            if pair["ready"]:
                pair_ready += 1
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
            "pairCached": pair_cached,
            "pairReady": pair_ready,
            "queueSize": len(queue),
            "processed": self._queue_scan_processed,
            "current": self._queue_scan_current,
            "errors": list(self._queue_scan_errors),
            "futureError": future_error,
            "minSideCandidates": MIN_SIDE_CANDIDATES,
        }

    def toggle_selection(
        self,
        *,
        side: str,
        identity: str,
        candidate_id: str,
        selected: bool,
    ) -> dict:
        """Select or unselect a candidate, enforcing three unique photos."""
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
                        f"Choose exactly {MAX_SELECTIONS_PER_IDENTITY} images; "
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
        """Export selected originals and mark an identity complete."""
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
                f"Select exactly {MIN_SELECTIONS_PER_IDENTITY} distinct images before marking done."
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
        """Copy selected original images and append manifest rows."""
        exported_at = datetime.now(UTC).isoformat()
        safe_identity = safe_path_component(identity)
        image_dir = HIGH_QUALITY_IMAGES_ROOT / side / safe_identity
        image_dir.mkdir(parents=True, exist_ok=True)
        HIGH_QUALITY_ROOT.mkdir(parents=True, exist_ok=True)

        existing = set(self._manifest_exports[(side, identity)])
        rows = []
        for candidate in candidates:
            if candidate.photo_identifier in existing:
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
            rows.append(
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
            existing.add(candidate.photo_identifier)

        if rows:
            write_header = not HIGH_QUALITY_MANIFEST.exists()
            with HIGH_QUALITY_MANIFEST.open("a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)
        with self._lock:
            self._manifest_exports[(side, identity)] = existing


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
