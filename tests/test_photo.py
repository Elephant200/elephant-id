from pathlib import Path

import pytest

from elephant_id.models import Photo


def _valid_photo(**overrides) -> Photo:
    base = dict(
        filename="Nellie_2024-05-10_1.jpg",
        image_path=Path("elephants-alive/Nellie/2024-05-10/Nellie_2024-05-10_1.jpg"),
        elephant_name="Nellie",
        sighting_id="Nellie_2024-05-10",
    )
    return Photo(**(base | overrides))


def test_photo_constructed_and_str():
    p = _valid_photo()
    assert p.filename == "Nellie_2024-05-10_1.jpg"
    assert p.image_path == Path("elephants-alive/Nellie/2024-05-10/Nellie_2024-05-10_1.jpg")
    assert str(p) == "Nellie_2024-05-10_1.jpg"


@pytest.mark.parametrize(
    "field, value",
    [
        ("filename", ""),
        ("elephant_name", ""),
        ("sighting_id", ""),
    ],
)
def test_empty_string_fields_raise(field, value):
    kwargs = {}
    if field == "filename":
        kwargs["filename"] = value
        kwargs["image_path"] = Path("")
    else:
        kwargs[field] = value
    with pytest.raises(ValueError):
        _valid_photo(**kwargs)


def test_absolute_image_path_raises():
    with pytest.raises(ValueError, match="relative"):
        _valid_photo(image_path=Path("/abs/Nellie_2024-05-10_1.jpg"))


def test_filename_must_match_image_path_name():
    with pytest.raises(ValueError, match="does not match"):
        _valid_photo(
            filename="wrong.jpg",
            image_path=Path("elephants-alive/Nellie/2024-05-10/Nellie_2024-05-10_1.jpg"),
        )


def test_filename_must_start_with_sighting_id_prefix():
    with pytest.raises(ValueError, match="does not start with"):
        _valid_photo(
            filename="Other_2024-05-10_1.jpg",
            image_path=Path("elephants-alive/Other/2024-05-10/Other_2024-05-10_1.jpg"),
        )


def test_sighting_id_must_start_with_elephant_name():
    with pytest.raises(ValueError, match="Sighting id does not start"):
        _valid_photo(
            sighting_id="xNellie_2024-05-10",
            filename="xNellie_2024-05-10_1.jpg",
            image_path=Path("sightings/xNellie_2024-05-10_1.jpg"),
        )
