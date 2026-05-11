from pathlib import Path

import pytest

from elephant_id.models import Photo


def _valid_photo(**overrides) -> Photo:
    base = dict(
        identifier="Devin_2015-11-05_09",
        image_path=Path("elephants-alive/Devin/2015-11-05/Devin_2015-11-05_09.jpg"),
        elephant_name="Devin",
        sighting_id="Devin_2015-11-05",
    )
    return Photo(**(base | overrides))


def test_photo_constructed_and_str():
    p = _valid_photo()
    assert p.identifier == "Devin_2015-11-05_09"
    assert p.image_path == Path("elephants-alive/Devin/2015-11-05/Devin_2015-11-05_09.jpg")
    assert str(p) == "Devin_2015-11-05_09"


@pytest.mark.parametrize(
    "field, value",
    [
        ("identifier", ""),
        ("elephant_name", ""),
        ("sighting_id", ""),
    ],
)
def test_empty_string_fields_raise(field, value):
    kwargs = {}
    if field == "identifier":
        kwargs["identifier"] = value
        kwargs["image_path"] = Path("")
    else:
        kwargs[field] = value
    with pytest.raises(ValueError):
        _valid_photo(**kwargs)


def test_absolute_image_path_raises():
    with pytest.raises(ValueError, match="relative"):
        _valid_photo(image_path=Path("/abs/Devin_2015-11-05_09.jpg"))


def test_identifier_must_match_image_path_stem():
    with pytest.raises(ValueError, match="does not match"):
        _valid_photo(
            identifier="wrong",
            image_path=Path("elephants-alive/Devin/2015-11-05/Devin_2015-11-05_09.jpg"),
        )


def test_identifier_must_start_with_sighting_id_prefix():
    with pytest.raises(ValueError, match="does not start with"):
        _valid_photo(
            identifier="Other_2024-05-10_1",
            image_path=Path("elephants-alive/Other/2024-05-10/Other_2024-05-10_1.jpg"),
        )


def test_sighting_id_must_start_with_elephant_name():
    with pytest.raises(ValueError, match="Sighting id does not start"):
        _valid_photo(
            elephant_name="Devin",
            sighting_id="Other_2015-11-05",
            identifier="Other_2015-11-05_09",
            image_path=Path("sightings/Other_2015-11-05_09.jpg"),
        )
