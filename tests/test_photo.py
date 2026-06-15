import dataclasses
from pathlib import Path

import pytest


def test_photo_constructed_and_str(make_photo):
    p = make_photo(sequence=9)
    assert p.identifier == "Devin_2015-11-05_09"
    assert p.image_path == Path("Devin/2015-11-05/Devin_2015-11-05_09.jpg")
    assert str(p) == "Photo('Devin_2015-11-05_09', image_path=Path('Devin/2015-11-05/Devin_2015-11-05_09.jpg'))"


@pytest.mark.parametrize(
    "field, value",
    [
        ("identifier", ""),
        ("elephant_name", ""),
        ("sighting_id", ""),
    ],
)
def test_photo_rejects_empty_identity_fields(make_photo, field, value):
    kwargs = {field: value}
    if field == "identifier":
        kwargs["image_path"] = Path("")
    with pytest.raises(ValueError):
        make_photo(**kwargs)


def test_photo_is_immutable(make_photo):
    p = make_photo()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.elephant_name = "Other"


def test_photos_with_equal_fields_are_equal_and_hashable(make_photo):
    a = make_photo(sequence=1)
    b = make_photo(sequence=1)
    c = make_photo(sequence=2)
    assert a == b and hash(a) == hash(b)
    assert a != c
    assert {a, b, c} == {a, c}


def test_absolute_image_path_raises(make_photo):
    with pytest.raises(ValueError, match="relative"):
        make_photo(sequence=9, image_path=Path("/abs/Devin_2015-11-05_09.jpg"))


def test_image_path_with_parent_traversal_raises(make_photo):
    with pytest.raises(ValueError, match=r"\.\."):
        make_photo(sequence=9, image_path=Path("../Devin_2015-11-05_09.jpg"))


def test_identifier_must_match_image_path_stem(make_photo):
    with pytest.raises(ValueError, match="does not match"):
        make_photo(
            identifier="wrong",
            image_path=Path("Devin/2015-11-05/Devin_2015-11-05_09.jpg"),
        )


def test_identifier_must_start_with_sighting_id_prefix(make_photo):
    with pytest.raises(ValueError, match="does not start with"):
        make_photo(
            identifier="Other_2024-05-10_1",
            image_path=Path("Other/2024-05-10/Other_2024-05-10_1.jpg"),
        )


def test_identifier_equal_to_sighting_id_without_suffix_raises(make_photo):
    with pytest.raises(ValueError, match="does not start with"):
        make_photo(
            identifier="Devin_2015-11-05",
            sighting_id="Devin_2015-11-05",
            image_path=Path("Devin/2015-11-05/Devin_2015-11-05.jpg"),
        )


def test_identifier_sharing_prefix_without_separator_raises(make_photo):
    with pytest.raises(ValueError, match="does not start with"):
        make_photo(
            identifier="Devin_2015-11-05extra",
            sighting_id="Devin_2015-11-05",
            image_path=Path("Devin/2015-11-05/Devin_2015-11-05extra.jpg"),
        )


def test_sighting_id_must_start_with_elephant_name(make_photo):
    with pytest.raises(ValueError, match="Sighting id does not start"):
        make_photo(
            name="Devin",
            sighting_id="Other_2015-11-05",
            identifier="Other_2015-11-05_09",
            image_path=Path("sightings/Other_2015-11-05_09.jpg"),
        )
