from datetime import date
from pathlib import Path

import pytest

from elephant_id.models import Photo, Sighting


def _photo(name="Nellie", d="2024-05-10", seq=1, subdir="sightings") -> Photo:
    sid = f"{name}_{d}"
    fn = f"{sid}_{seq}.jpg"
    return Photo(
        filename=fn,
        image_path=Path(f"{subdir}/{fn}"),
        elephant_name=name,
        sighting_id=sid,
    )


def _sighting(photos=None, **overrides) -> Sighting:
    d = date(2024, 5, 10)
    base = dict(
        elephant_name="Nellie",
        sighting_date=d,
        sighting_id="Nellie_2024-05-10",
        photos=photos if photos is not None else (_photo(),),
    )
    return Sighting(**(base | overrides))


def test_sighting_one_photo_and_len():
    s = _sighting()
    assert len(s) == 1
    assert s.elephant_name == "Nellie"
    assert s.sighting_date == date(2024, 5, 10)
    assert s.sighting_id == "Nellie_2024-05-10"


def test_sighting_multiple_photos():
    s = _sighting(photos=(_photo(seq=1), _photo(seq=2)))
    assert len(s) == 2


def test_sighting_id_must_match_name_and_date():
    with pytest.raises(ValueError, match="does not match"):
        _sighting(sighting_id="Wrong_2024-05-10")


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


def test_duplicate_photo_filenames_raise():
    p = _photo()
    with pytest.raises(ValueError, match="duplicated"):
        _sighting(photos=(p, p))


def test_photo_sighting_id_must_match_sighting():
    other = _photo(name="Other", d="2024-06-01", subdir="other")
    with pytest.raises(ValueError, match="sighting_id"):
        _sighting(photos=(other,))