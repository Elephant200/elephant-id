"""In-memory reviewer state.

A single `ReviewerState` instance backs the running app. Public
methods are thread-safe and serialize their JSON-shaped views.
"""

from __future__ import annotations

import logging
import random
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from elephant_id.dataset import Dataset

from . import filters, samples, seek_codes
from .actions import Action, PriorityToggle, SavedRemoveSighting
from .config import (
    PAGE_SIZE_DEFAULT,
    PAGE_SIZE_MAX,
    SAMPLES_ROOT,
    SAMPLES_SIGHTINGS_ROOT,
)
from .filters import FilterConfig
from .paths import (
    is_image,
    parse_sighting_folder_name,
    safe_coded_rel_image,
    safe_saved_sighting_dir,
    safe_saved_sighting_file,
    samples_folder_rel,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SightingKey:
    name: str
    date: str

    @property
    def id(self) -> str:
        return f"{self.name}_{self.date}"


def list_saved_sighting_entries(folder_rel: str) -> list[dict]:
    """Return saved sighting image rows for routes."""
    folder = safe_saved_sighting_dir(folder_rel)
    rows: list[tuple[str, str, bool]] = []
    try:
        for p in sorted(folder.iterdir()):
            if not p.is_file() or not is_image(p):
                continue
            rel = str(p.relative_to(SAMPLES_ROOT)).replace("\\", "/")
            rows.append((rel, p.name.lower(), samples.is_priority_filename(p.name)))
    except OSError:
        logger.warning("Could not list saved sighting %s", folder, exc_info=True)
    rows.sort(key=lambda t: (not t[2], t[1]))
    return [{"rel": r, "priority": pri} for r, _sortkey, pri in rows]


class ReviewerState:
    """Thread-safe state container for the reviewer.

    The `queue` is the sightings list the user is browsing.
    "Elephant only" mode narrows the queue temporarily; toggling off
    restores the pre-narrow queue.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sighting_images: dict[SightingKey, list[str]] = {}
        self._elephant_seek: dict[str, str] = {}
        self._available_years: list[int] = []
        self._available_ages: list[int] = []

        self.queue: list[SightingKey] = []
        self.current_index: int = 0
        self.page: int = 0
        self.page_size: int = PAGE_SIZE_DEFAULT
        self.shuffle_enabled: bool = True
        self.actions: list[Action] = []
        self.filter_config: FilterConfig = FilterConfig()

        self._last_event: dict | None = None
        self._elephant_only_backup: list[SightingKey] | None = None
        self._elephant_only_name: str | None = None

    # ------------------------------------------------------------------ load

    def load(self, dataset: Dataset) -> None:
        """Pull sightings from `dataset` and reset the queue."""
        sighting_images: dict[SightingKey, list[str]] = {}
        for sighting in dataset.iter_sightings():
            key = SightingKey(
                name=sighting.elephant_name,
                date=sighting.sighting_date.isoformat(),
            )
            sighting_images[key] = [
                str(photo.image_path).replace("\\", "/").lstrip("/")
                for photo in sighting.photos
            ]

        # First non-empty seek_code per elephant, mirroring CSV row order.
        elephant_seek: dict[str, str] = {}
        df = dataset.metadata
        if "seek_code" in df.columns:
            for name, code in zip(df["name"], df["seek_code"], strict=True):
                name = (name or "").strip() if isinstance(name, str) else name
                if not name or name in elephant_seek:
                    continue
                if not isinstance(code, str):
                    continue
                code = code.strip()
                if code:
                    elephant_seek[name] = code

        keys = list(sighting_images.keys())
        years = sorted(
            {y for k in sighting_images if (y := filters.year_from_date(k.date)) is not None}
        )
        # Age is computed per sighting (birth-decade midpoint vs. sighting
        # year), so cache each elephant's decade once and span every sighting.
        elephant_decade = {
            name: seek_codes.parse(code).age for name, code in elephant_seek.items()
        }
        ages = sorted(
            {age for k in sighting_images
             if (age := filters.age_from_decade(
                 elephant_decade.get(k.name), filters.year_from_date(k.date)
             )) is not None}
        )

        with self._lock:
            self._sighting_images = sighting_images
            self._elephant_seek = elephant_seek
            self._available_years = years
            self._available_ages = ages
            self.shuffle_enabled = True
            random.shuffle(keys)
            self.queue = keys
            self.current_index = 0
            self.page = 0
            self.actions = []
            self.filter_config = FilterConfig()
            self._elephant_only_backup = None
            self._elephant_only_name = None

    # ----------------------------------------------------------- queue ops

    def _keys_matching_filter_unlocked(self, cfg: FilterConfig) -> list[SightingKey]:
        return [
            k for k in self._sighting_images
            if filters.matches(k, self._elephant_seek, cfg)
        ]

    def _filtered_candidates_unlocked(self) -> list[SightingKey]:
        cand = self._keys_matching_filter_unlocked(self.filter_config)
        if self._elephant_only_backup is not None and self._elephant_only_name:
            cand = [k for k in cand if k.name == self._elephant_only_name]
        return cand

    def _order_unlocked(self, keys: list[SightingKey]) -> list[SightingKey]:
        if self.shuffle_enabled:
            random.shuffle(keys)
        else:
            keys.sort(key=lambda k: (k.name, k.date))
        return keys

    def _clamp_unlocked(self) -> None:
        if not self.queue:
            self.current_index = 0
            self.page = 0
            return
        self.current_index %= len(self.queue)
        images_len = len(self._current_images_unlocked())
        if images_len == 0:
            self.page = 0
            return
        max_pages = max(0, (images_len - 1) // self.page_size)
        self.page = max(0, min(self.page, max_pages))

    def _current_key_unlocked(self) -> SightingKey | None:
        if not self.queue:
            return None
        return self.queue[self.current_index]

    def _current_images_unlocked(self) -> list[str]:
        key = self._current_key_unlocked()
        if not key:
            return []
        return self._sighting_images.get(key, [])

    def apply_filter(self, cfg: FilterConfig) -> None:
        with self._lock:
            prev = self._current_key_unlocked()
            self.filter_config = cfg
            matching = self._order_unlocked(self._keys_matching_filter_unlocked(cfg))

            if self._elephant_only_backup is not None and self._elephant_only_name:
                self._elephant_only_backup = matching
                new_queue = [k for k in matching if k.name == self._elephant_only_name]
            else:
                new_queue = matching

            if prev is not None and prev in new_queue:
                self.current_index = new_queue.index(prev)
            elif new_queue:
                self.current_index = min(self.current_index, len(new_queue) - 1)
            else:
                self.current_index = 0
            self.queue = new_queue
            self.page = 0
            self._clamp_unlocked()

    def elephant_only_set(self, enabled: bool) -> None:
        with self._lock:
            if enabled:
                if self._elephant_only_backup is not None:
                    return
                key = self._current_key_unlocked()
                if not key:
                    return
                self._elephant_only_backup = list(self.queue)
                self.queue = [k for k in self.queue if k.name == key.name]
                self._elephant_only_name = key.name
                self.current_index = 0
            else:
                if self._elephant_only_backup is None:
                    return
                anchor = self._elephant_only_name
                self.queue = self._elephant_only_backup
                self._elephant_only_backup = None
                self._elephant_only_name = None
                idx = next((i for i, k in enumerate(self.queue) if k.name == anchor), None)
                self.current_index = idx if idx is not None else 0
            self.page = 0
            self._clamp_unlocked()

    def set_shuffle(self, enabled: bool) -> None:
        with self._lock:
            cur = self._current_key_unlocked()
            self.shuffle_enabled = enabled
            self.queue = self._order_unlocked(self._filtered_candidates_unlocked())
            if cur is not None and cur in self.queue:
                self.current_index = self.queue.index(cur)
            else:
                self.current_index = 0
            self.page = 0
            self._clamp_unlocked()

    def nav(self, delta: int) -> None:
        with self._lock:
            if not self.queue:
                return
            self.current_index = (self.current_index + delta) % len(self.queue)
            self.page = 0
            self._clamp_unlocked()

    def page_nav(self, delta: int) -> None:
        with self._lock:
            if not self.queue:
                return
            images_len = len(self._current_images_unlocked())
            if images_len == 0:
                self.page = 0
                return
            max_pages = max(0, (images_len - 1) // self.page_size)
            self.page = max(0, min(self.page + delta, max_pages))

    def set_page_size(self, page_size: int) -> None:
        with self._lock:
            self.page_size = max(1, min(page_size, PAGE_SIZE_MAX))
            self._clamp_unlocked()

    # ---------------------------------------------------------- mutations

    def toggle_priority_image(self, rel_image: str) -> None:
        """Toggle priority for an image in the current sighting."""
        try:
            rel_norm = safe_coded_rel_image(rel_image)
        except ValueError:
            return

        with self._lock:
            key = self._current_key_unlocked()
            if not key:
                return
            queue_index = self.current_index
            allowed = set(self._sighting_images.get(key, []))

        if rel_norm not in allowed:
            return

        basename = Path(rel_norm).name
        try:
            folder_path, created = samples.ensure_sighting_folder(key.name, key.date)
        except (FileNotFoundError, OSError):
            logger.warning("Could not ensure sighting folder for %s", key, exc_info=True)
            return

        toggle_res = samples.toggle_priority(folder_path, key.name, key.date, basename)
        if toggle_res is None:
            return
        from_bn, to_bn = toggle_res

        def _rollback() -> None:
            try:
                if created:
                    shutil.rmtree(folder_path)
                else:
                    cur_from = folder_path / to_bn
                    cur_to = folder_path / from_bn
                    if cur_from.exists():
                        cur_from.rename(cur_to)
            except OSError:
                logger.warning("Rollback failed for %s", folder_path, exc_info=True)

        with self._lock:
            # Race check: if the queue moved, undo the FS change rather than
            # recording an action that points at a stale index.
            if (
                not self.queue
                or queue_index >= len(self.queue)
                or self.queue[queue_index] != key
            ):
                _rollback()
                return
            self.actions.append(
                PriorityToggle(
                    key=key,
                    queue_index=queue_index,
                    folder_rel=samples_folder_rel(folder_path),
                    created_folder=created,
                    from_basename=from_bn,
                    to_basename=to_bn,
                )
            )
            self._last_event = {
                "type": "priority_image_toggled",
                "name": key.name,
                "date": key.date,
            }
        samples.sync_starred_for_basenames([samples.plain_basename(from_bn)])

    def toggle_priority_samples_file(self, samples_rel: str) -> None:
        """Toggle priority for a file under `samples/sightings/`."""
        try:
            path = safe_saved_sighting_file(samples_rel)
        except ValueError:
            return
        folder_path = path.parent
        parsed = parse_sighting_folder_name(folder_path.name)
        if parsed is None:
            return
        sight_name, sight_date = parsed
        plain_bn = samples.plain_basename(path.name)
        toggle_res = samples.toggle_priority(folder_path, sight_name, sight_date, plain_bn)
        if toggle_res is None:
            return
        from_bn, to_bn = toggle_res
        key = SightingKey(name=sight_name, date=sight_date)

        with self._lock:
            self.actions.append(
                PriorityToggle(
                    key=key,
                    queue_index=self.current_index,
                    folder_rel=samples_folder_rel(folder_path),
                    created_folder=False,
                    from_basename=from_bn,
                    to_basename=to_bn,
                )
            )
            self._last_event = {
                "type": "priority_image_toggled",
                "name": sight_name,
                "date": sight_date,
            }
        samples.sync_starred_for_basenames([plain_bn])

    def saved_remove_sighting(self, rel: str) -> None:
        path = safe_saved_sighting_dir(rel)
        if not path.is_dir():
            raise ValueError("Not a saved sighting folder")
        # Capture which basenames were starred so undo can re-mirror them.
        affected = tuple(samples.list_priority_basenames(path))
        samples.remove_sighting_folder(path)
        rel_norm = str(path.relative_to(SAMPLES_ROOT)).replace("\\", "/")
        with self._lock:
            self.actions.append(
                SavedRemoveSighting(saved_rel=rel_norm, affected_priority_basenames=affected)
            )
            self._last_event = {"type": "saved_sighting_removed", "detail": path.name}
        if affected:
            samples.sync_starred_for_basenames(affected)

    # --------------------------------------------------------------- undo

    def undo(self) -> None:
        with self._lock:
            if not self.actions:
                return
            action = self.actions.pop()

        try:
            self._apply_undo(action)
        except Exception:
            with self._lock:
                self.actions.append(action)
            raise

    def _apply_undo(self, action: Action) -> None:
        if isinstance(action, SavedRemoveSighting):
            samples.restore_sighting_folder(action.saved_rel)
            with self._lock:
                self._last_event = {
                    "type": "undo_saved_sighting_removed",
                    "detail": Path(action.saved_rel).name or "sighting",
                }
            if action.affected_priority_basenames:
                samples.sync_starred_for_basenames(action.affected_priority_basenames)
            return

        if isinstance(action, PriorityToggle):
            if action.created_folder:
                self._delete_samples_copy(action.folder_rel)
            else:
                folder = SAMPLES_ROOT / action.folder_rel.replace("\\", "/").lstrip("/")
                src_p = folder / action.to_basename
                dst_p = folder / action.from_basename
                if src_p.exists():
                    src_p.rename(dst_p)
            with self._lock:
                evt: dict = {"type": "undo_priority_toggle"}
                evt["name"] = action.key.name
                evt["date"] = action.key.date
                self._last_event = evt
            samples.sync_starred_for_basenames(action.affected_basenames)
            return

    def _delete_samples_copy(self, copied_to: str) -> None:
        dst = SAMPLES_ROOT / copied_to
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        samples.prune_empty_parents(dst.parent, SAMPLES_ROOT)

    # ------------------------------------------------------- read views

    def saved_list_dict(self) -> dict:
        with self._lock:
            cfg = self.filter_config
            elephant_seek = self._elephant_seek

        sightings: list[dict] = []
        if SAMPLES_SIGHTINGS_ROOT.is_dir():
            try:
                entries = sorted(SAMPLES_SIGHTINGS_ROOT.iterdir())
            except OSError:
                logger.warning("Could not list %s", SAMPLES_SIGHTINGS_ROOT, exc_info=True)
                entries = []
            for p in entries:
                if not p.is_dir():
                    continue
                parsed = parse_sighting_folder_name(p.name)
                if parsed is None:
                    continue
                key = SightingKey(name=parsed[0], date=parsed[1])
                if not filters.matches(key, elephant_seek, cfg):
                    continue
                rel = str(p.relative_to(SAMPLES_ROOT)).replace("\\", "/")
                try:
                    nimg = sum(1 for x in p.iterdir() if x.is_file() and is_image(x))
                except OSError:
                    nimg = 0
                sightings.append({"rel": rel, "folder": p.name, "imageCount": nimg})
        return {"sightings": sightings}

    def view(self) -> dict:
        with self._lock:
            key = self._current_key_unlocked()
            images = self._current_images_unlocked()
            idx = self.current_index
            remaining = len(self.queue)
            page = self.page
            page_size = self.page_size
            event = self._last_event
            self._last_event = None
            meta = self._view_meta_unlocked()
            seek_code = (
                (self._elephant_seek.get(key.name) or "").strip()
                if key is not None else ""
            )

        if not key:
            return {"done": True, "remaining": 0, "event": event, **meta}

        samples_dir = samples.find_sighting_folder(key.name, key.date)
        sighting_saved = samples_dir is not None
        start = page * page_size
        end = min(len(images), start + page_size)
        page_images = images[start:end]
        max_pages = max(1, (len(images) + page_size - 1) // page_size)

        images_payload: list[dict] = []
        for p in page_images:
            bn = Path(p).name
            pri = bool(samples_dir is not None and (samples_dir / samples.priority_basename(bn)).exists())
            images_payload.append({"path": p, "priorityStarred": pri})

        return {
            "done": False,
            "queueIndex": idx,
            "remaining": remaining,
            "name": key.name,
            "date": key.date,
            "seekCode": seek_code,
            "sightingId": key.id,
            "page": page,
            "pages": max_pages,
            "pageSize": page_size,
            "imageCount": len(images),
            "images": images_payload,
            "sightingSaved": sighting_saved,
            "event": event,
            **meta,
        }

    def _view_meta_unlocked(self) -> dict:
        extent: dict[str, int] | None = None
        if self._available_years:
            extent = {"min": self._available_years[0], "max": self._available_years[-1]}
        age_extent: dict[str, int] | None = None
        if self._available_ages:
            age_extent = {"min": self._available_ages[0], "max": self._available_ages[-1]}
        return {
            "filters": self.filter_config.to_json(),
            "yearExtent": extent,
            "ageExtent": age_extent,
            "elephantOnly": self._elephant_only_backup is not None,
            "elephantOnlyName": self._elephant_only_name,
            "shuffleEnabled": self.shuffle_enabled,
        }
