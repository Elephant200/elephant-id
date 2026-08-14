"""Tests for the research Dataset and image-only PhotoStore seam."""

import csv
from datetime import date
from pathlib import Path
from uuid import UUID

import pytest

from elephant_id.dataset import Dataset, PhotoStore
from elephant_id.domain import Photo

SIGHTING_A = UUID("919bc2ca-0817-45d5-b81e-780deb7bfbf8")
SIGHTING_B = UUID("36ca67fc-d693-41ec-a7d5-1ae88e36cfa0")
PHOTO_A1 = UUID("2ba8e9c1-b6e4-4f66-9f3d-66bc5700ef49")
PHOTO_A2 = UUID("0042cc08-5de2-4fc9-bb1b-82fb7478b25d")
PHOTO_B1 = UUID("04f5cf3c-dcaa-4855-9043-cb2b3f26b297")

METADATA_ROWS = (
    {
        "photo_id": str(PHOTO_A1),
        "sighting_id": str(SIGHTING_A),
        "date": "2020-01-02",
        "name": "Ada",
        "image_path": "Ada/2020-01-02/one.jpg",
    },
    {
        "photo_id": str(PHOTO_A2),
        "sighting_id": str(SIGHTING_A),
        "date": "2020-01-02",
        "name": "Ada",
        "image_path": "Ada/2020-01-02/two.jpg",
    },
    {
        "photo_id": str(PHOTO_B1),
        "sighting_id": str(SIGHTING_B),
        "date": "2021-03-04",
        "name": "Bea",
        "image_path": "Bea/2021-03-04/one.jpg",
    },
)


def _write_metadata(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    """Write canonical assigned metadata for a synthetic Dataset."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_ROWS[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Create a Dataset backed by synthetic metadata and encoded bytes."""
    image_root = tmp_path / "coded"
    image_root.mkdir()
    metadata_path = tmp_path / "images.csv"
    _write_metadata(metadata_path, METADATA_ROWS)
    for row in METADATA_ROWS:
        image_path = image_root / row["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"original:{row['photo_id']}".encode())
    return Dataset(dataset_root=image_root, metadata_path=metadata_path)


def test_dataset_constructs_and_resolves_neutral_domain_values(
    dataset: Dataset,
) -> None:
    """Dataset resolves UUIDs, global values, and private identity metadata."""
    photo = dataset.photo(PHOTO_A1)
    sighting = dataset.sighting(SIGHTING_A)

    assert photo == Photo(photo_id=PHOTO_A1, sighting_id=SIGHTING_A)
    assert sighting.sighting_date == date(2020, 1, 2)
    assert {item.photo_id for item in sighting.photos} == {PHOTO_A1, PHOTO_A2}
    assert dataset.known_elephant_name(SIGHTING_A) == "Ada"
    assert {item.photo_id for item in dataset.iter_photos()} == {
        PHOTO_A1,
        PHOTO_A2,
        PHOTO_B1,
    }
    assert {item.sighting_id for item in dataset.iter_sightings()} == {
        SIGHTING_A,
        SIGHTING_B,
    }


def test_photo_store_reads_original_encoded_bytes(dataset: Dataset) -> None:
    """PhotoStore resolves immutable bytes solely through photo identity."""
    photo = dataset.photo(PHOTO_A1)
    store: PhotoStore = dataset.photo_store

    encoded = store.read(photo)

    assert encoded == f"original:{PHOTO_A1}".encode()


def test_photo_store_reports_missing_mapped_bytes(
    dataset: Dataset,
    tmp_path: Path,
) -> None:
    """A mapped Photo with no file raises a clear storage error."""
    photo = dataset.photo(PHOTO_A1)
    stored_path = tmp_path / "coded" / METADATA_ROWS[0]["image_path"]
    stored_path.unlink()

    with pytest.raises(FileNotFoundError, match=str(PHOTO_A1)):
        dataset.photo_store.read(photo)


def test_dataset_rejects_inconsistent_sighting_metadata(tmp_path: Path) -> None:
    """One sighting UUID cannot resolve to multiple known elephants."""
    image_root = tmp_path / "coded"
    image_root.mkdir()
    metadata_path = tmp_path / "images.csv"
    inconsistent = (
        METADATA_ROWS[0],
        METADATA_ROWS[1] | {"name": "Someone else"},
    )
    _write_metadata(metadata_path, inconsistent)

    with pytest.raises(ValueError, match="inconsistent metadata"):
        Dataset(dataset_root=image_root, metadata_path=metadata_path)
