from datetime import date
from pathlib import Path

import pytest

from elephant_id.domain import Photo, Sighting


def _photo(name="Devin", d="2015-11-05", seq=1, subdir="sightings") -> Photo:
    sid = f"{name}_{d}"
    identifier = f"{sid}_{seq:02d}"
    return Photo(
        identifier=identifier,
        image_path=Path(f"{subdir}/{identifier}.jpg"),
        elephant_name=name,
        sighting_id=sid,
    )


def _sighting(photos=None, **overrides) -> Sighting:
    d = date(2015, 11, 5)
    base = dict(
        elephant_name="Devin",
        sighting_date=d,
        sighting_id="Devin_2015-11-05",
        photos=photos if photos is not None else (_photo(),),
    )
    return Sighting(**(base | overrides))


def test_sighting_one_photo_and_len():
    s = _sighting()
    assert len(s) == 1
    assert s.elephant_name == "Devin"
    assert s.sighting_date == date(2015, 11, 5)
    assert s.sighting_id == "Devin_2015-11-05"


def test_sighting_multiple_photos():
    s = _sighting(photos=(_photo(seq=1), _photo(seq=2)))
    assert len(s) == 2


def test_sighting_id_must_match_name_and_date():
    with pytest.raises(ValueError, match="does not match"):
        _sighting(sighting_id="Wrong_2015-11-05")


@pytest.mark.parametrize(
    "field, value",
    [
        ("elephant_name", ""),
        ("sighting_id", ""),
    ],
)
def test_empty_string_fields_raise(field, value):
    with pytest.raises(ValueError):
        _sighting(**{field: value})


def test_requires_at_least_one_photo():
    with pytest.raises(ValueError, match="At least one photo"):
        _sighting(photos=())


def test_duplicate_photo_identifiers_raise():
    p = _photo()
    with pytest.raises(ValueError, match="duplicated"):
        _sighting(photos=(p, p))


def test_photo_sighting_id_must_match_sighting():
    other = _photo(name="Aaron", d="2008-11-24", subdir="Aaron/2008-11-24")
    with pytest.raises(ValueError, match="sighting_id"):
        _sighting(photos=(other,))
