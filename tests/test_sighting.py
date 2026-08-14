"""Tests for neutral sighting domain values."""

import dataclasses
from datetime import date
from uuid import UUID

import pytest

from elephant_id.domain import Photo, Sighting, SightingEarPair

SIGHTING_ID = UUID("919bc2ca-0817-45d5-b81e-780deb7bfbf8")
OTHER_SIGHTING_ID = UUID("36ca67fc-d693-41ec-a7d5-1ae88e36cfa0")
LEFT_PHOTO_ID = UUID("2ba8e9c1-b6e4-4f66-9f3d-66bc5700ef49")
RIGHT_PHOTO_ID = UUID("0042cc08-5de2-4fc9-bb1b-82fb7478b25d")


def _photo(
    photo_id: UUID = LEFT_PHOTO_ID,
    sighting_id: UUID = SIGHTING_ID,
) -> Photo:
    """Construct a neutral Photo for domain tests."""
    return Photo(photo_id=photo_id, sighting_id=sighting_id)


def test_sighting_groups_neutral_photos() -> None:
    """A Sighting exposes only event identity, date, and its Photos."""
    photos = (_photo(), _photo(RIGHT_PHOTO_ID))

    sighting = Sighting(
        sighting_id=SIGHTING_ID,
        sighting_date=date(2020, 1, 2),
        photos=photos,
    )

    assert [field.name for field in dataclasses.fields(sighting)] == [
        "sighting_id",
        "sighting_date",
        "photos",
    ]
    assert sighting.photos == photos


def test_sighting_requires_photos() -> None:
    """An observed Sighting cannot contain no Photos."""
    with pytest.raises(ValueError, match="at least one Photo"):
        Sighting(
            sighting_id=SIGHTING_ID,
            sighting_date=date(2020, 1, 2),
            photos=(),
        )


def test_sighting_rejects_duplicate_photos() -> None:
    """A Sighting cannot contain one photo identity more than once."""
    photo = _photo()

    with pytest.raises(ValueError, match="duplicate photo_id"):
        Sighting(
            sighting_id=SIGHTING_ID,
            sighting_date=date(2020, 1, 2),
            photos=(photo, photo),
        )


def test_sighting_rejects_photo_from_another_sighting() -> None:
    """Every grouped Photo must belong to the Sighting."""
    with pytest.raises(ValueError, match="does not belong"):
        Sighting(
            sighting_id=SIGHTING_ID,
            sighting_date=date(2020, 1, 2),
            photos=(_photo(sighting_id=OTHER_SIGHTING_ID),),
        )


def test_ear_pair_declares_one_photo_per_side() -> None:
    """An ear pair preserves its explicitly declared left and right Photos."""
    left = _photo()
    right = _photo(RIGHT_PHOTO_ID)

    pair = SightingEarPair(
        sighting_id=SIGHTING_ID,
        left_photo=left,
        right_photo=right,
    )

    assert pair.left_photo is left
    assert pair.right_photo is right


def test_ear_pair_allows_one_photo_for_both_sides() -> None:
    """One source Photo may provide both declared ear sides."""
    photo = _photo()

    pair = SightingEarPair(
        sighting_id=SIGHTING_ID,
        left_photo=photo,
        right_photo=photo,
    )

    assert pair.left_photo == pair.right_photo


def test_ear_pair_rejects_photo_from_another_sighting() -> None:
    """Both declared ear Photos must belong to the pair's Sighting."""
    with pytest.raises(ValueError, match="right_photo does not belong"):
        SightingEarPair(
            sighting_id=SIGHTING_ID,
            left_photo=_photo(),
            right_photo=_photo(RIGHT_PHOTO_ID, OTHER_SIGHTING_ID),
        )
