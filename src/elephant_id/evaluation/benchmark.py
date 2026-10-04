"""Loading of retrieval benchmark files."""

import csv
from pathlib import Path
from uuid import UUID

from elephant_id.dataset import Dataset
from elephant_id.domain import SightingEarPair

Benchmark = dict[str, list[SightingEarPair]]
"""Sighting ear pairs grouped by known-elephant name."""

_COLUMNS = ["known_elephant_name", "sighting_id", "left_photo_id", "right_photo_id"]


def load_benchmark(path: Path, dataset: Dataset) -> Benchmark:
    """Read a benchmark file and resolve each row against the Dataset.

    Raises:
        KeyError: If a row names an unknown sighting or photo.
        ValueError: If the columns are wrong, a sighting repeats, or a row disagrees with the Dataset.
    """
    benchmark: Benchmark = {}
    seen: set[UUID] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _COLUMNS:
            raise ValueError(f"{path} columns must be {_COLUMNS}, got {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            try:
                pair = _resolve_row(row, dataset)
                if pair.sighting_id in seen:
                    raise ValueError(f"Duplicate sighting_id {pair.sighting_id}")
            except (KeyError, ValueError) as error:
                error.add_note(f"{path}, row {row_number}")
                raise
            seen.add(pair.sighting_id)
            if row["known_elephant_name"] not in benchmark:
                benchmark[row["known_elephant_name"]] = []
            benchmark[row["known_elephant_name"]].append(pair)
    return benchmark


def _resolve_row(row: dict[str, str], dataset: Dataset) -> SightingEarPair:
    """Return the row's sighting ear pair after checking its known elephant."""
    sighting_id = UUID(row["sighting_id"])
    name = dataset.known_elephant_name(sighting_id)
    if name != row["known_elephant_name"]:
        raise ValueError(f"Sighting {sighting_id} belongs to {name!r}, not {row['known_elephant_name']!r}")
    return SightingEarPair(
        sighting_id=sighting_id,
        left_photo=dataset.photo(UUID(row["left_photo_id"])),
        right_photo=dataset.photo(UUID(row["right_photo_id"])),
    )
