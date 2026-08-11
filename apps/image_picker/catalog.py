"""Elephant and sighting index for the matching image picker."""

from __future__ import annotations

from collections import defaultdict

from elephant_id.dataset import Dataset
from elephant_id.domain import Sighting


class PhotoCatalog:
    """Read-only index of the dataset's sightings grouped by elephant."""

    def __init__(self, sightings_by_elephant: dict[str, list[Sighting]]) -> None:
        """Build the by-elephant and by-sighting-id indexes."""
        self.sightings_by_elephant = sightings_by_elephant
        self.sighting_by_id: dict[str, Sighting] = {
            sighting.sighting_id: sighting
            for sightings in sightings_by_elephant.values()
            for sighting in sightings
        }

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> PhotoCatalog:
        """Group every dataset sighting under its elephant in CSV row order."""
        grouped: dict[str, list[Sighting]] = defaultdict(list)
        for sighting in dataset.iter_sightings():
            grouped[sighting.elephant_name].append(sighting)
        return cls(dict(grouped))

    def elephants(self) -> list[str]:
        """Return every elephant name in first-seen order."""
        return list(self.sightings_by_elephant.keys())

    def sightings(self, identity: str) -> list[Sighting]:
        """Return one elephant's sightings, or an empty list if unknown."""
        return self.sightings_by_elephant.get(identity, [])
