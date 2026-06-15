from datetime import date

import pytest

from elephant_id.domain import Sighting


def test_sighting_one_photo_and_len(make_sighting):
    s = make_sighting()
    assert len(s) == 1
    assert s.elephant_name == "Devin"
    assert s.sighting_date == date(2015, 11, 5)
    assert s.sighting_id == "Devin_2015-11-05"
    assert str(s) == "Sighting('Devin_2015-11-05', num_photos=1)"


def test_sighting_multiple_photos(make_photo, make_sighting):
    s = make_sighting(photos=(make_photo(sequence=1), make_photo(sequence=2)))
    assert len(s) == 2


def test_sighting_id_must_match_name_and_date(make_sighting):
    with pytest.raises(ValueError, match="does not match"):
        make_sighting(sighting_id="Wrong_2015-11-05")


def test_sighting_id_uses_zero_padded_iso_date(make_sighting):
    s = make_sighting(sighting_date=date(2015, 1, 5))
    assert s.sighting_id == "Devin_2015-01-05"


@pytest.mark.parametrize(
    "field, value",
    [
        ("elephant_name", ""),
        ("sighting_id", ""),
    ],
)
def test_sighting_rejects_empty_identity_fields(make_sighting, field, value):
    with pytest.raises(ValueError):
        make_sighting(**{field: value})


def test_sighting_rejects_missing_date(make_photo):
    with pytest.raises(ValueError, match="Sighting date"):
        Sighting(
            elephant_name="Devin",
            sighting_date=None,
            sighting_id="Devin_2015-11-05",
            photos=(make_photo(),),
        )


def test_requires_at_least_one_photo(make_sighting):
    with pytest.raises(ValueError, match="At least one photo"):
        make_sighting(photos=())


def test_duplicate_photo_identifiers_raise(make_photo, make_sighting):
    p = make_photo()
    with pytest.raises(ValueError, match="duplicated"):
        make_sighting(photos=(p, p))


def test_photo_sighting_id_must_match_sighting(make_photo, make_sighting):
    other = make_photo(name="Aaron", sighting_date="2008-11-24")
    with pytest.raises(ValueError, match="sighting_id"):
        make_sighting(photos=(other,))


def test_photo_elephant_name_must_match_sighting(make_sighting):
    class MismatchedPhoto:
        identifier = "Devin_2015-11-05_99"
        sighting_id = "Devin_2015-11-05"
        elephant_name = "Aaron"

    with pytest.raises(ValueError, match="elephant name"):
        make_sighting(photos=(MismatchedPhoto(),))
