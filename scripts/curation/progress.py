"""Report which elephants have outcomes in run.log.

Run `uv run python scripts/curation/progress.py` while score.py is writing
run.log. Photo IDs in the log are resolved through Dataset names. The
report refreshes in place every 2 seconds. ETA uses the past 10, 5, or 3
minutes of first-seen photos, whichever window is long enough and has
completions.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

from tqdm import tqdm

from elephant_id.dataset import Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
METADATA = ROOT / "dataset/elephants-alive/images.csv"
ARCHIVE = ROOT / "dataset/elephants-alive/coded"
LOG = ROOT / "run.log"
INTERVAL = 2
CLEAR = "\033[H\033[J"
RATE_WINDOWS = (10 * 60, 5 * 60, 3 * 60)
PHOTO_ID = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} "
    r"([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})\b"
)


def first_seen_photos(path: Path) -> tuple[datetime | None, dict[UUID, datetime]]:
    """Return the first log timestamp and each photo's first outcome time."""
    started: datetime | None = None
    found: dict[UUID, datetime] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(line) < 23:
                continue
            try:
                stamp = datetime.fromisoformat(line[:23])
            except ValueError:
                continue
            if started is None:
                started = stamp
            match = PHOTO_ID.match(line)
            if match:
                found.setdefault(UUID(match.group(1)), stamp)
    return started, found


def recent_rate(
    first_seen: Mapping[UUID, datetime], started: datetime, now: datetime
) -> float | None:
    """Photos per second over the longest recent window that has completions."""
    elapsed = (now - started).total_seconds()
    for window in RATE_WINDOWS:
        if elapsed < window:
            continue
        cutoff = now - timedelta(seconds=window)
        count = sum(stamp >= cutoff for stamp in first_seen.values())
        if count:
            return count / window
    return None


def format_bar(n: int, total: int, elapsed: float, rate: float | None) -> str:
    """Format a tqdm-style progress bar, with ETA only when `rate` is set."""
    if rate:
        bar_format = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    else:
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
    return tqdm.format_meter(
        n=n,
        total=total,
        elapsed=max(elapsed, 0.0),
        unit="photo",
        rate=rate,
        ncols=shutil.get_terminal_size().columns,
        bar_format=bar_format,
    )


def photos_by_name(dataset: Dataset) -> dict[str, set[UUID]]:
    """Index every dataset photo ID by known-elephant name."""
    grouped: dict[str, set[UUID]] = defaultdict(set)
    for photo in dataset.iter_photos():
        grouped[dataset.known_elephant_name(photo.sighting_id)].add(photo.photo_id)
    return grouped


def sightings_by_name(dataset: Dataset) -> dict[str, int]:
    """Count sightings per known-elephant name."""
    grouped: dict[str, set[UUID]] = defaultdict(set)
    for sighting in dataset.iter_sightings():
        grouped[dataset.known_elephant_name(sighting.sighting_id)].add(
            sighting.sighting_id
        )
    return {name: len(ids) for name, ids in grouped.items()}


def processing_order(sightings: Mapping[str, int]) -> list[str]:
    """Return names in the archive order used by `score.py --all`."""
    return sorted(sightings, key=lambda name: (-sightings[name], name))


def print_complete(
    complete: list[tuple[str, int]], sightings: Mapping[str, int]
) -> None:
    """Print at most the first 10 and last 5 completed elephants."""
    print("\nComplete:")
    shown = complete if len(complete) <= 15 else complete[:10] + complete[-5:]
    for index, (name, count) in enumerate(shown):
        if len(complete) > 15 and index == 10:
            print("  ...")
        print(f"  {name}  ({count} photos, {sightings[name]} sightings)")


def render(
    inventory: dict[str, set[UUID]],
    names: list[str],
    sightings: Mapping[str, int],
    log: Path,
) -> None:
    """Print complete, in-progress, and upcoming elephants from the log."""
    now = datetime.now()
    run_started, first_seen = first_seen_photos(log)
    seen = set(first_seen)
    total = sum(len(photos) for photos in inventory.values())
    elapsed = (now - run_started).total_seconds() if run_started else 0.0
    rate = recent_rate(first_seen, run_started, now) if run_started else None
    print(format_bar(len(seen), total, elapsed, rate))
    print()
    unknown = sorted(
        seen - {photo_id for photos in inventory.values() for photo_id in photos}
    )
    complete: list[tuple[str, int]] = []
    started: list[tuple[str, int, int]] = []
    upcoming: list[tuple[str, int]] = []
    for name in names:
        photos = inventory[name]
        count = len(photos & seen)
        if count == 0:
            upcoming.append((name, len(photos)))
        elif count == len(photos):
            complete.append((name, count))
        else:
            started.append((name, count, len(photos)))

    print(
        f"{len(complete)} complete, {len(started)} in progress, "
        f"{len(seen)} / {total} photos, "
        f"{len(inventory)} elephants"
    )
    if complete:
        print_complete(complete, sightings)
    if started:
        print("\nIn progress:")
        for name, count, total in started:
            print(f"  {name}  ({count}/{total} photos, {sightings[name]} sightings)")
    if upcoming:
        print("\nNext:")
        for name, total in upcoming[:5]:
            print(f"  {name}  ({total} photos, {sightings[name]} sightings)")
    if unknown:
        print("\nUnknown photo IDs:")
        for photo_id in unknown:
            print(f"  {photo_id}")


def main() -> None:
    """Refresh the log progress report in place every 2 seconds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=LOG, help="Curation run log")
    args = parser.parse_args()
    if not args.log.is_file():
        raise SystemExit(f"Log not found: {args.log}")

    dataset = Dataset(ARCHIVE, METADATA)
    inventory = photos_by_name(dataset)
    sightings = sightings_by_name(dataset)
    names = processing_order(sightings)
    try:
        while True:
            print(CLEAR, end="")
            render(inventory, names, sightings, args.log)
            sys.stdout.flush()
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
