"""Dataset and candidate identity indexing for the image picker."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from elephant_id.domain import Photo

from .config import (
    CSV_PATH,
    FILTERED_CROPS_ROOT,
    IMAGE_EXTENSIONS,
    MIN_IDENTITY_PHOTOS,
    MIN_SIGHTINGS,
    QUEUE_SEED,
    QUEUE_SIZE,
    SIDES,
)


@dataclass(frozen=True)
class PhotoRecord:
    """One metadata row from the elephant image dataset."""

    identifier: str
    date: str
    name: str
    image_path: Path
    seek_code: str

    def to_photo(self) -> Photo:
        """Convert the row into a validated domain photo."""
        return Photo(
            identifier=self.identifier,
            image_path=self.image_path,
            elephant_name=self.name,
            sighting_id=f"{self.name}_{self.date}",
        )


class PhotoCatalog:
    """Read-only metadata index used by the picker."""

    def __init__(self, records: list[PhotoRecord], filtered_sides: dict[str, set[str]]) -> None:
        """Build secondary indexes over dataset rows."""
        self.records = records
        self.filtered_sides = filtered_sides
        self.by_identifier = {record.identifier: record for record in records}
        self.by_identity: dict[str, list[PhotoRecord]] = defaultdict(list)
        self.sightings_by_identity: dict[str, set[str]] = defaultdict(set)
        for record in records:
            self.by_identity[record.name].append(record)
            self.sightings_by_identity[record.name].add(record.date)

    @classmethod
    def from_paths(
        cls,
        csv_path: Path = CSV_PATH,
        filtered_root: Path = FILTERED_CROPS_ROOT,
    ) -> PhotoCatalog:
        """Load the dataset metadata and filtered-crop side hints."""
        records: list[PhotoRecord] = []
        with csv_path.open(newline="") as file:
            for row in csv.DictReader(file):
                records.append(
                    PhotoRecord(
                        identifier=row["identifier"],
                        date=row["date"],
                        name=row["name"],
                        image_path=Path(row["image_path"]),
                        seek_code=row.get("seek_code") or "",
                    )
                )
        return cls(records=records, filtered_sides=_read_filtered_sides(filtered_root))

    def eligible_identities(self) -> list[str]:
        """Return identities with enough source photos and sightings."""
        return [
            name
            for name, records in self.by_identity.items()
            if len(records) >= MIN_IDENTITY_PHOTOS
            and len(self.sightings_by_identity[name]) >= MIN_SIGHTINGS
        ]

    def shared_queue(self, size: int = QUEUE_SIZE) -> list[str]:
        """Return one deterministic identity queue shared by both sides."""
        eligible = set(self.eligible_identities())
        both = [
            name for name in eligible
            if {"left", "right"}.issubset(self.filtered_sides.get(name, set()))
        ]
        one_side = [
            name for name in eligible
            if self.filtered_sides.get(name, set()) and name not in both
        ]
        remaining = [name for name in eligible if name not in both and name not in one_side]

        ordered: list[str] = []
        for label, group in (("both", both), ("one", one_side), ("rest", remaining)):
            shuffled = sorted(group)
            random.Random(f"{QUEUE_SEED}:shared:{label}").shuffle(shuffled)
            ordered.extend(shuffled)
        return ordered[:size]

    def queue_for_side(self, side: str, size: int = QUEUE_SIZE) -> list[str]:
        """Return the shared identity queue for compatibility with callers."""
        if side not in SIDES:
            raise ValueError(f"Invalid side: {side}")
        return self.shared_queue(size=size)

    def summary(self) -> dict:
        """Return aggregate candidate-pool counts for UI diagnostics."""
        eligible = set(self.eligible_identities())
        return {
            "datasetIdentities": len(self.by_identity),
            "eligibleIdentities": len(eligible),
            "filteredIdentities": len(self.filtered_sides),
            "bothFilteredSideHints": sum(
                1
                for name in eligible
                if {"left", "right"}.issubset(self.filtered_sides.get(name, set()))
            ),
            "leftFilteredSideHints": sum(
                1 for name in eligible if "left" in self.filtered_sides.get(name, set())
            ),
            "rightFilteredSideHints": sum(
                1 for name in eligible if "right" in self.filtered_sides.get(name, set())
            ),
        }


def _read_filtered_sides(filtered_root: Path) -> dict[str, set[str]]:
    """Read side hints from existing filtered crop filenames."""
    by_name: dict[str, set[str]] = defaultdict(set)
    if not filtered_root.is_dir():
        return by_name
    for path in filtered_root.iterdir():
        if not path.is_file() or path.suffix not in IMAGE_EXTENSIONS:
            continue
        identifier, side = parse_crop_filename(path)
        if identifier is None or side is None:
            continue
        # Identifiers end in _YYYY-MM-DD_sequence; rsplit preserves names with spaces/underscores.
        name = identifier.rsplit("_", maxsplit=2)[0]
        by_name[name].add(side)
    return by_name


def parse_crop_filename(path: Path) -> tuple[str | None, str | None]:
    """Return ``(photo_identifier, side)`` for a side crop filename."""
    if path.stem.endswith("_left"):
        return path.stem[:-5], "left"
    if path.stem.endswith("_right"):
        return path.stem[:-6], "right"
    return None, None
