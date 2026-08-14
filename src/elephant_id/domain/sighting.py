"""Neutral sighting values shared by research and applications."""

from dataclasses import dataclass
from datetime import date
from uuid import UUID

from .photo import Photo, _validate_uuid4


@dataclass(frozen=True, slots=True)
class Sighting:
    """One observed event and its distinct immutable Photos."""

    sighting_id: UUID
    sighting_date: date
    photos: tuple[Photo, ...]

    def __post_init__(self) -> None:
        """Validate date, photo uniqueness, and sighting membership."""
        _validate_uuid4(self.sighting_id, "sighting_id")
        if type(self.sighting_date) is not date:
            raise TypeError("sighting_date must be a date")
        if not isinstance(self.photos, tuple):
            raise TypeError("photos must be a tuple")
        if not self.photos:
            raise ValueError("Sighting must contain at least one Photo")

        photo_ids: set[UUID] = set()
        for photo in self.photos:
            if photo.sighting_id != self.sighting_id:
                raise ValueError(f"Photo {photo.photo_id} does not belong to Sighting")
            if photo.photo_id in photo_ids:
                raise ValueError(f"Sighting contains duplicate photo_id {photo.photo_id}")
            photo_ids.add(photo.photo_id)


@dataclass(frozen=True, slots=True)
class SightingEarPair:
    """One declared left-ear Photo and right-ear Photo from a sighting."""

    sighting_id: UUID
    left_photo: Photo
    right_photo: Photo

    def __post_init__(self) -> None:
        """Validate that both declared Photos belong to the sighting."""
        _validate_uuid4(self.sighting_id, "sighting_id")
        for field_name, photo in (
            ("left_photo", self.left_photo),
            ("right_photo", self.right_photo),
        ):
            if not isinstance(photo, Photo):
                raise TypeError(f"{field_name} must be a Photo")
            if photo.sighting_id != self.sighting_id:
                raise ValueError(f"{field_name} does not belong to Sighting")
