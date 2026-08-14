"""Tests for the neutral Photo domain value."""

import dataclasses
from uuid import UUID, uuid1

import pytest

from elephant_id.domain import Photo

PHOTO_ID = UUID("2ba8e9c1-b6e4-4f66-9f3d-66bc5700ef49")
SIGHTING_ID = UUID("919bc2ca-0817-45d5-b81e-780deb7bfbf8")


def test_photo_carries_only_opaque_identity() -> None:
    """A Photo exposes only its permanent and parent UUIDs."""
    photo = Photo(photo_id=PHOTO_ID, sighting_id=SIGHTING_ID)

    assert [field.name for field in dataclasses.fields(photo)] == [
        "photo_id",
        "sighting_id",
    ]
    assert photo.photo_id == PHOTO_ID
    assert photo.sighting_id == SIGHTING_ID


def test_photo_is_immutable_and_hashable() -> None:
    """Photos are immutable values suitable for sets and mapping keys."""
    photo = Photo(photo_id=PHOTO_ID, sighting_id=SIGHTING_ID)

    assert {photo, photo} == {photo}
    with pytest.raises(dataclasses.FrozenInstanceError):
        photo.photo_id = UUID("0042cc08-5de2-4fc9-bb1b-82fb7478b25d")


@pytest.mark.parametrize("value", ["not-a-uuid", uuid1()])
def test_photo_requires_uuid4_identity(value: object) -> None:
    """Photo identity rejects non-UUID and non-v4 UUID values."""
    with pytest.raises((TypeError, ValueError), match="UUIDv4"):
        Photo(photo_id=value, sighting_id=SIGHTING_ID)
