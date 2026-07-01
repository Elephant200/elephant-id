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
    FALLBACK_READY_IMAGES,
    MIN_READY_IMAGES,
    MIN_READY_SIGHTINGS,
    QUEUE_SEED,
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

    def __init__(self, records: list[PhotoRecord]) -> None:
        """Build secondary indexes over dataset rows."""
        self.records = records
        self.by_identifier = {record.identifier: record for record in records}
        self.by_identity: dict[str, list[PhotoRecord]] = defaultdict(list)
        self.sightings_by_identity: dict[str, set[str]] = defaultdict(set)
        for record in records:
            self.by_identity[record.name].append(record)
            self.sightings_by_identity[record.name].add(record.date)

    @classmethod
    def from_paths(cls, csv_path: Path = CSV_PATH) -> PhotoCatalog:
        """Load the dataset metadata."""
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
        return cls(records=records)

    def eligible_identities(self) -> list[str]:
        """Return identities with enough source photos and sightings."""
        return [
            name
            for name, records in self.by_identity.items()
            if identity_is_ready(
                image_count=len({record.identifier for record in records}),
                sighting_count=len(self.sightings_by_identity[name]),
            )
        ]

    def scan_pool(self) -> list[str]:
        """Return every eligible identity in a deterministic shuffled scan order.

        The picker scans this pool in order and keeps the first identities that
        clear the per-side candidate bar, so all eligible identities are
        reachable rather than only a fixed prefix.
        """
        pool = sorted(self.eligible_identities())
        random.Random(f"{QUEUE_SEED}:scan-pool").shuffle(pool)
        return pool

    def summary(self) -> dict:
        """Return aggregate candidate-pool counts for UI diagnostics."""
        return {
            "datasetIdentities": len(self.by_identity),
            "eligibleIdentities": len(self.eligible_identities()),
        }


def identity_is_ready(*, image_count: int, sighting_count: int) -> bool:
    """Return whether counts pass the picker diversity-or-volume rule."""
    return (
        sighting_count >= MIN_READY_SIGHTINGS
        and image_count >= MIN_READY_IMAGES
    ) or image_count >= FALLBACK_READY_IMAGES
