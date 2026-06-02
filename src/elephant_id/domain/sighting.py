from dataclasses import dataclass
from datetime import date

from .photo import Photo


@dataclass(frozen=True, slots=True)
class Sighting:
    """A sighting of an elephant, grouping its photos."""

    elephant_name: str
    sighting_date: date
    sighting_id: str # Unique sighting identifier; consists of elephant name and sighting date separated by underscore.
    photos: tuple[Photo, ...]

    def __post_init__(self) -> None:
        """Validate identity, date, and per-photo consistency."""
        if not self.elephant_name:
            raise ValueError(f"Elephant name is empty: {self.elephant_name}")
        if not self.sighting_date:
            raise ValueError(f"Sighting date is empty: {self.sighting_date}")
        if not self.sighting_id:
            raise ValueError(f"Sighting id is empty: {self.sighting_id}")
        if self.sighting_id != f"{self.elephant_name}_{self.sighting_date.isoformat()}":
            raise ValueError(f"Sighting id does not match elephant name and sighting date: {self.sighting_id} != {self.elephant_name}_{self.sighting_date.isoformat()}")
        if not self.photos:
            raise ValueError(f"At least one photo is required: {self.photos}")

        identifiers = set()
        for photo in self.photos:
            if photo.identifier in identifiers:
                raise ValueError(f"Photo {photo.identifier} is duplicated")
            identifiers.add(photo.identifier)
            if photo.sighting_id != self.sighting_id:
                raise ValueError(f"Photo {photo.identifier} has sighting_id {photo.sighting_id}, expected {self.sighting_id}")
            if photo.elephant_name != self.elephant_name:
                raise ValueError(f"Photo {photo.identifier} has elephant name {photo.elephant_name}, expected {self.elephant_name}")

    def __len__(self) -> int:
        """Return the number of photos in the sighting."""
        return len(self.photos)

    def __str__(self) -> str:
        """Return a compact representation with the photo count."""
        return f"Sighting({self.sighting_id}, photos={len(self.photos)})"
