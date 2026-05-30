from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from elephant_id.dataset import Dataset
from elephant_id.domain import Photo, SeekCode, Sighting

# Real subset of dataset/elephants-alive/images.csv: two elephants, three
# sightings. Order matters for iter_sightings (preserves CSV row order).
# Tuple: (identifier, name, date_str, color, seek_code). image_path is derived.
AARON_CODE = "B80T11E____-3_3_X_0S_01"
DEVIN_CODE = "B__T11E____-4_3_X00S___"
ROWS = [
    ("Aaron_2008-11-24_01", "Aaron", "2008-11-24", (10, 20, 30), AARON_CODE),
    ("Aaron_2008-11-24_02", "Aaron", "2008-11-24", (40, 50, 60), AARON_CODE),
    ("Aaron_2008-11-24_03", "Aaron", "2008-11-24", (70, 80, 90), AARON_CODE),
    ("Devin_2015-11-05_01", "Devin", "2015-11-05", (100, 110, 120), DEVIN_CODE),
    ("Devin_2015-11-05_02", "Devin", "2015-11-05", (130, 140, 150), DEVIN_CODE),
    ("Devin_2017-06-27_01", "Devin", "2017-06-27", (160, 170, 180), ""),
]


def _image_path(identifier: str, name: str, date_str: str) -> str:
    return f"{name}/{date_str}/{identifier}.jpg"


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    root = tmp_path / "coded"
    root.mkdir()
    pd.DataFrame(
        [
            {
                "identifier": identifier,
                "date": date_str,
                "name": name,
                "image_path": _image_path(identifier, name, date_str),
                "seek_code": seek_code,
            }
            for identifier, name, date_str, _, seek_code in ROWS
        ]
    ).to_csv(tmp_path / "images.csv", index=False)
    for identifier, name, date_str, color, _ in ROWS:
        path = root / _image_path(identifier, name, date_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color).save(path)
    return Dataset(dataset_root=root, metadata_path=tmp_path / "images.csv")


def test_init_rejects_missing_root(tmp_path, dataset):
    with pytest.raises(FileNotFoundError, match="Dataset root"):
        Dataset(dataset_root=tmp_path / "missing", metadata_path=dataset.metadata_path)


def test_init_rejects_file_as_root(tmp_path, dataset):
    file_path = tmp_path / "not-a-dir"
    file_path.write_text("x")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        Dataset(dataset_root=file_path, metadata_path=dataset.metadata_path)


def test_init_rejects_missing_metadata(dataset):
    with pytest.raises(FileNotFoundError, match="Metadata path"):
        Dataset(dataset_root=dataset.dataset_root, metadata_path=dataset.dataset_root / "missing.csv")


def test_init_rejects_non_csv_metadata(dataset):
    bad = dataset.dataset_root / "metadata.json"
    bad.write_text("{}")
    with pytest.raises(ValueError, match="must be a CSV"):
        Dataset(dataset_root=dataset.dataset_root, metadata_path=bad)


def test_metadata_is_lazy_and_dates_parsed(dataset: Dataset):
    assert dataset.metadata is None
    dataset._ensure_loaded()
    assert len(dataset.metadata) == len(ROWS)
    assert isinstance(dataset.metadata["date"].iloc[0], date)


def test_path_for_resolves_to_existing_file(dataset: Dataset):
    photo = dataset.get_photo("Aaron_2008-11-24_01")
    assert dataset.path_for(photo) == dataset.dataset_root / photo.image_path
    assert dataset.path_for(photo).exists()


def test_get_photo_returns_fully_populated_photo(dataset: Dataset):
    assert dataset.get_photo("Devin_2017-06-27_01") == Photo(
        identifier="Devin_2017-06-27_01",
        image_path=Path("Devin/2017-06-27/Devin_2017-06-27_01.jpg"),
        elephant_name="Devin",
        sighting_id="Devin_2017-06-27",
    )


def test_get_photo_unknown_identifier_raises(dataset: Dataset):
    with pytest.raises(KeyError, match="No photo with identifier"):
        dataset.get_photo("Ghost_2099-01-01_01")


def test_iter_photos_yields_all_rows_in_order(dataset: Dataset):
    assert [p.identifier for p in dataset.iter_photos()] == [r[0] for r in ROWS]


def test_get_sighting_aggregates_all_photos_for_day(dataset: Dataset):
    sighting = dataset.get_sighting("Aaron", date(2008, 11, 24))
    assert isinstance(sighting, Sighting)
    assert sighting.sighting_id == "Aaron_2008-11-24"
    assert [p.identifier for p in sighting.photos] == [
        "Aaron_2008-11-24_01", "Aaron_2008-11-24_02", "Aaron_2008-11-24_03",
    ]


def test_get_sighting_separates_same_elephant_by_date(dataset: Dataset):
    a = dataset.get_sighting("Devin", date(2015, 11, 5))
    b = dataset.get_sighting("Devin", date(2017, 6, 27))
    assert (len(a), len(b)) == (2, 1)
    assert {p.identifier for p in a.photos}.isdisjoint(p.identifier for p in b.photos)


def test_get_sighting_unknown_elephant_raises(dataset: Dataset):
    with pytest.raises(KeyError, match="No sighting"):
        dataset.get_sighting("Ghost", date(2008, 11, 24))


def test_get_sighting_unknown_date_raises(dataset: Dataset):
    with pytest.raises(KeyError, match="2099-01-01"):
        dataset.get_sighting("Aaron", date(2099, 1, 1))


def test_iter_sightings_preserves_csv_order_and_groups(dataset: Dataset):
    assert [(s.elephant_name, s.sighting_date, len(s)) for s in dataset.iter_sightings()] == [
        ("Aaron", date(2008, 11, 24), 3),
        ("Devin", date(2015, 11, 5), 2),
        ("Devin", date(2017, 6, 27), 1),
    ]


def test_get_ground_truth_returns_parsed_seek_code(dataset: Dataset):
    sighting = dataset.get_sighting("Aaron", date(2008, 11, 24))
    code = dataset.get_ground_truth(sighting)
    assert isinstance(code, SeekCode)
    assert str(code) == AARON_CODE


def test_get_ground_truth_unknown_sighting_raises(dataset: Dataset):
    ghost = Sighting(
        elephant_name="Aaron",
        sighting_date=date(2099, 1, 1),
        sighting_id="Aaron_2099-01-01",
        photos=(Photo(
            identifier="Aaron_2099-01-01_01",
            image_path=Path("Aaron/2099-01-01/Aaron_2099-01-01_01.jpg"),
            elephant_name="Aaron",
            sighting_id="Aaron_2099-01-01",
        ),),
    )
    with pytest.raises(KeyError, match="Sighting not found"):
        dataset.get_ground_truth(ghost)


def test_get_ground_truth_missing_code_raises(dataset: Dataset):
    sighting = dataset.get_sighting("Devin", date(2017, 6, 27))
    with pytest.raises(ValueError, match="no seek code"):
        dataset.get_ground_truth(sighting)


def test_read_image_returns_rgb_copy_and_caches(dataset: Dataset):
    photo = dataset.get_photo("Aaron_2008-11-24_01")
    image = dataset.read_image(photo)
    assert image.mode == "RGB" and image.size == (4, 4)
    assert image.getpixel((0, 0)) == ROWS[0][3]
    assert image is not dataset._image_cache[photo.identifier]


def test_read_image_repeated_reads_return_distinct_copies(dataset: Dataset):
    photo = dataset.get_photo("Aaron_2008-11-24_01")
    first, second = dataset.read_image(photo), dataset.read_image(photo)
    assert first is not second
    assert first.getpixel((0, 0)) == second.getpixel((0, 0))


def test_read_image_lru_eviction(dataset: Dataset):
    dataset.image_cache_size = 2
    photos = list(dataset.iter_photos())[:3]
    for p in photos:
        dataset.read_image(p)
    assert list(dataset._image_cache) == [photos[1].identifier, photos[2].identifier]


def test_read_image_cache_hit_refreshes_recency(dataset: Dataset):
    dataset.image_cache_size = 2
    a, b, c = list(dataset.iter_photos())[:3]

    dataset.read_image(a)
    dataset.read_image(b)
    dataset.read_image(a)  # cache hit: touch a so b becomes least-recent
    dataset.read_image(c)  # evicts b, not a

    assert list(dataset._image_cache) == [a.identifier, c.identifier]


def test_clear_image_cache(dataset: Dataset):
    dataset.read_image(dataset.get_photo("Aaron_2008-11-24_01"))
    dataset.clear_image_cache()
    assert not dataset._image_cache
