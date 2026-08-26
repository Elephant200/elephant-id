"""Operations on the `dataset/samples/` tree.

The reviewer "starring" model has two related stores:

1. `samples/sightings/<Name_YYYY-MM-DD>/` mirrors a sighting copied
   from `coded/`. Files prefixed with `** ` are priority copies;
   that prefix is the ground truth for star state.
2. `samples/starred/` is a flat directory of priority files for
   external tooling. It is kept consistent with (1) by
   `sync_starred_for_basenames`.

If two sightings hold a priority file with the same plain basename, the
lexicographically smallest sighting folder name wins.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable
from pathlib import Path

from .config import (
    CODED_ROOT,
    PRIORITY_STAR_PREFIX,
    SAMPLES_ROOT,
    SAMPLES_SIGHTINGS_ROOT,
    STARRED_SAMPLES_ROOT,
)
from .paths import (
    coded_sighting_dir,
    is_image,
    parse_sighting_folder_name,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Priority filename helpers
# ---------------------------------------------------------------------------

def priority_basename(plain_basename: str) -> str:
    return f"{PRIORITY_STAR_PREFIX}{plain_basename}"


def is_priority_filename(filename: str) -> bool:
    return filename.startswith(PRIORITY_STAR_PREFIX)


def plain_basename(filename: str) -> str:
    if is_priority_filename(filename):
        return filename[len(PRIORITY_STAR_PREFIX) :]
    return filename


# ---------------------------------------------------------------------------
# Sighting folder discovery / creation
# ---------------------------------------------------------------------------

def find_sighting_folder(name: str, date: str) -> Path | None:
    """Return the saved sighting folder for `(name, date)`, if any."""
    if not SAMPLES_SIGHTINGS_ROOT.is_dir():
        return None
    try:
        candidates = sorted(SAMPLES_SIGHTINGS_ROOT.iterdir())
    except OSError:
        logger.warning("Could not list %s", SAMPLES_SIGHTINGS_ROOT, exc_info=True)
        return None
    for p in candidates:
        if not p.is_dir():
            continue
        if parse_sighting_folder_name(p.name) == (name, date):
            return p
    return None


def _copy_sighting_to_samples(name: str, date: str) -> Path:
    src = coded_sighting_dir(name, date)
    if not src.is_dir():
        raise FileNotFoundError(f"Missing sighting folder: {src}")
    safe_name = name.replace("/", "_").replace("\\", "_").strip()
    safe_date = date.replace("/", "_").replace("\\", "_").strip()
    base_dst = SAMPLES_SIGHTINGS_ROOT / f"{safe_name}_{safe_date}"
    dst = base_dst
    if dst.exists():
        i = 2
        while True:
            candidate = base_dst.with_name(base_dst.name + f"_dup{i}")
            if not candidate.exists():
                dst = candidate
                break
            i += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    return dst


def ensure_sighting_folder(name: str, date: str) -> tuple[Path, bool]:
    """Get or create a saved sighting folder.

    Returns:
        `(path, created)`.
    """
    existing = find_sighting_folder(name, date)
    if existing is not None:
        return existing, False
    return _copy_sighting_to_samples(name, date), True


# ---------------------------------------------------------------------------
# Priority toggle within a sighting folder
# ---------------------------------------------------------------------------

def toggle_priority(
    folder_path: Path,
    sighting_name: str,
    sighting_date: str,
    plain_bn: str,
) -> tuple[str, str] | None:
    """Toggle `plain_bn` within `folder_path`.

    If neither variant currently exists in `folder_path` we copy from
    `coded/` and immediately mark it priority.

    Returns `(from_basename, to_basename)` describing the change, or
    `None` if nothing changed.
    """
    plain = folder_path / plain_bn
    pri = folder_path / priority_basename(plain_bn)
    try:
        if pri.exists():
            if plain.exists():
                return None
            pri.rename(plain)
            return priority_basename(plain_bn), plain_bn
        if plain.exists():
            plain.rename(pri)
            return plain_bn, priority_basename(plain_bn)
        src = coded_sighting_dir(sighting_name, sighting_date) / plain_bn
        if not src.is_file():
            return None
        shutil.copy2(src, plain)
        plain.rename(pri)
        return plain_bn, priority_basename(plain_bn)
    except OSError:
        logger.warning("Failed to toggle priority for %s in %s", plain_bn, folder_path, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Folder utilities
# ---------------------------------------------------------------------------

def first_priority_or_any_image(folder: Path) -> Path | None:
    """Return first priority image, first plain image, or None."""
    try:
        files = sorted(p for p in folder.iterdir() if p.is_file() and is_image(p))
    except OSError:
        return None
    for p in files:
        if is_priority_filename(p.name):
            return p
    return files[0] if files else None


def list_priority_basenames(folder: Path) -> list[str]:
    """Plain basenames currently marked priority in `folder`."""
    try:
        return [
            plain_basename(p.name)
            for p in folder.iterdir()
            if p.is_file() and is_image(p) and is_priority_filename(p.name)
        ]
    except OSError:
        return []


def remove_sighting_folder(folder: Path) -> None:
    """Recursively remove a saved sighting folder. Raises on failure."""
    shutil.rmtree(folder)


def restore_sighting_folder(saved_rel: str) -> None:
    """Restore a removed saved sighting from `coded/<name>/<date>`."""
    saved_rel = (saved_rel or "").replace("\\", "/").strip().lstrip("/")
    folder_name = Path(saved_rel).name
    parsed = parse_sighting_folder_name(folder_name)
    if parsed is None:
        raise ValueError("Cannot parse saved sighting folder name")
    name, date = parsed
    src = CODED_ROOT / name / date
    if not src.is_dir():
        raise FileNotFoundError(f"Cannot restore sighting: missing {src}")
    dst = SAMPLES_ROOT / saved_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def prune_empty_parents(start: Path, stop: Path) -> None:
    """Remove empty parents from `start` up to `stop`."""
    p = start
    while True:
        if p == stop or not p.exists():
            return
        try:
            p.rmdir()
        except OSError:
            return
        p = p.parent


# ---------------------------------------------------------------------------
# Starred mirror — incremental sync
# ---------------------------------------------------------------------------

def _resolve_winner(plain_bn: str) -> Path | None:
    """Find the priority file under `sightings/` for `starred/`."""
    if not SAMPLES_SIGHTINGS_ROOT.is_dir():
        return None
    try:
        folders = sorted(SAMPLES_SIGHTINGS_ROOT.iterdir(), key=lambda q: q.name)
    except OSError:
        return None
    target_name = priority_basename(plain_bn)
    for folder in folders:
        if not folder.is_dir():
            continue
        if parse_sighting_folder_name(folder.name) is None:
            continue
        candidate = folder / target_name
        if candidate.is_file():
            return candidate
    return None


def sync_starred_for_basenames(plain_basenames: Iterable[str]) -> None:
    """Re-mirror the starred/ entries for each given plain basename.

    For each basename, we look for the winning priority file and copy it
    (creating `starred/` as needed). If no winner exists, the starred
    copy is removed. Quiet on filesystem errors; they are logged.
    """
    seen: set[str] = set()
    try:
        STARRED_SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("Could not create %s", STARRED_SAMPLES_ROOT, exc_info=True)
        return

    for basename in plain_basenames:
        if not basename or basename in seen:
            continue
        seen.add(basename)
        dst = STARRED_SAMPLES_ROOT / basename
        winner = _resolve_winner(basename)
        try:
            if winner is None:
                if dst.exists():
                    dst.unlink()
                continue
            if dst.exists() and dst.stat().st_mtime_ns >= winner.stat().st_mtime_ns:
                continue
            shutil.copy2(winner, dst)
        except OSError:
            logger.warning("Failed to sync starred mirror for %s", basename, exc_info=True)


def reconcile_all_starred() -> None:
    """Full sweep: rebuild `starred/` from scratch.

    Invoked once at startup. Incremental sync handles steady state.
    Quiet on filesystem errors.
    """
    try:
        STARRED_SAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("Could not create %s", STARRED_SAMPLES_ROOT, exc_info=True)
        return

    desired: dict[str, Path] = {}
    if SAMPLES_SIGHTINGS_ROOT.is_dir():
        try:
            for folder in sorted(SAMPLES_SIGHTINGS_ROOT.iterdir(), key=lambda q: q.name):
                if not folder.is_dir() or parse_sighting_folder_name(folder.name) is None:
                    continue
                try:
                    files = sorted(
                        (x for x in folder.iterdir() if x.is_file() and is_image(x)),
                        key=lambda x: x.name,
                    )
                except OSError:
                    continue
                for p in files:
                    if not is_priority_filename(p.name):
                        continue
                    plain = plain_basename(p.name)
                    desired.setdefault(plain, p)
        except OSError:
            logger.warning("Could not scan %s", SAMPLES_SIGHTINGS_ROOT, exc_info=True)

    try:
        existing = list(STARRED_SAMPLES_ROOT.iterdir())
    except OSError:
        existing = []

    for entry in existing:
        if not entry.is_file() or not is_image(entry):
            continue
        if entry.name not in desired:
            try:
                entry.unlink()
            except OSError:
                logger.warning("Could not remove stale starred entry %s", entry, exc_info=True)

    for basename, src in desired.items():
        dst = STARRED_SAMPLES_ROOT / basename
        try:
            if dst.exists() and dst.stat().st_mtime_ns >= src.stat().st_mtime_ns:
                continue
            shutil.copy2(src, dst)
        except OSError:
            logger.warning("Could not copy %s -> %s", src, dst, exc_info=True)
