"""Neutral photo identity shared across AlphaPhant."""

from dataclasses import dataclass
from uuid import UUID


def _validate_uuid4(value: object, field_name: str) -> None:
    """Require a UUIDv4 value for a domain identity field."""
    if not isinstance(value, UUID):
        raise TypeError(f"{field_name} must be a UUIDv4 value")
    if value.version != 4:
        raise ValueError(f"{field_name} must be a UUIDv4 value")


@dataclass(frozen=True, slots=True)
class Photo:
    """One immutable original photo asset belonging to a sighting."""

    photo_id: UUID
    sighting_id: UUID

    def __post_init__(self) -> None:
        """Validate permanent photo and parent sighting identity."""
        _validate_uuid4(self.photo_id, "photo_id")
        _validate_uuid4(self.sighting_id, "sighting_id")
