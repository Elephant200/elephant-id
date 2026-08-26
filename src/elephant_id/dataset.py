"""Research metadata and image-only storage for the assigned dataset."""

import csv
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Protocol
from uuid import UUID

from elephant_id.domain import Photo, Sighting

_METADATA_COLUMNS = ("photo_id", "sighting_id", "date", "name", "image_path")


class PhotoStore(Protocol):
    """Retrieve immutable original encoded bytes for neutral Photos."""

    def read(self, photo: Photo) -> bytes:
        """Return the original encoded bytes associated with a Photo."""
        ...


class _FilesystemPhotoStore:
    """Resolve original photo bytes from assigned filesystem metadata."""

    def __init__(self, paths_by_photo_id: dict[UUID, Path]) -> None:
        self._paths_by_photo_id = paths_by_photo_id

    def read(self, photo: Photo) -> bytes:
        """Return original bytes for a mapped Photo.

        Raises:
            KeyError: If the photo ID has no storage mapping.
            FileNotFoundError: If the mapped bytes are missing.
        """
        try:
            path = self._paths_by_photo_id[photo.photo_id]
        except KeyError:
            raise KeyError(f"No storage mapping for photo {photo.photo_id}") from None
        try:
            return path.read_bytes()
        except FileNotFoundError:
            raise FileNotFoundError(f"Stored bytes are missing for photo {photo.photo_id}") from None


class Dataset:
    """Identity-aware research metadata with an image-only PhotoStore."""

    def __init__(self, dataset_root: Path, metadata_path: Path) -> None:
        """Load assigned metadata and build the indices.

        Args:
            dataset_root: Root directory containing paths from the metadata.
            metadata_path: Assigned canonical image metadata CSV.

        Raises:
            FileNotFoundError: If the root or metadata file is missing.
            NotADirectoryError: If the dataset root is not a directory.
            ValueError: If metadata cannot construct unambiguous domain values.
        """
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
        if not dataset_root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

        root = dataset_root.resolve()
        photos_by_id: dict[UUID, Photo] = {}
        paths_by_photo_id: dict[UUID, Path] = {}
        photos_by_sighting_id: dict[UUID, list[Photo]] = {}
        dates_by_sighting_id: dict[UUID, date] = {}
        names_by_sighting_id: dict[UUID, str] = {}

        with metadata_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if tuple(reader.fieldnames or ()) != _METADATA_COLUMNS:
                raise ValueError(f"Metadata columns must be {_METADATA_COLUMNS}, got {reader.fieldnames}")

            for row_number, row in enumerate(reader, start=2):
                photo_id = self._parse_uuid4(row["photo_id"], "photo_id", row_number)
                sighting_id = self._parse_uuid4(
                    row["sighting_id"], "sighting_id", row_number
                )
                try:
                    sighting_date = date.fromisoformat(row["date"])
                except ValueError:
                    raise ValueError(f"Invalid date at metadata row {row_number}: {row['date']!r}") from None
                name = row["name"]
                if not name:
                    raise ValueError(f"Missing name at metadata row {row_number}")
                image_path = Path(row["image_path"])
                if image_path.is_absolute() or ".." in image_path.parts:
                    raise ValueError(f"Unsafe image_path at metadata row {row_number}: {image_path}")
                if photo_id in photos_by_id:
                    raise ValueError(f"Duplicate photo_id in metadata: {photo_id}")

                existing_date = dates_by_sighting_id.setdefault(
                    sighting_id, sighting_date
                )
                existing_name = names_by_sighting_id.setdefault(sighting_id, name)
                if existing_date != sighting_date or existing_name != name:
                    raise ValueError(f"Sighting {sighting_id} has inconsistent metadata")

                photo = Photo(photo_id=photo_id, sighting_id=sighting_id)
                photos_by_id[photo_id] = photo
                paths_by_photo_id[photo_id] = root / image_path
                photos_by_sighting_id.setdefault(sighting_id, []).append(photo)

        self._photos_by_id = photos_by_id
        self._sightings_by_id = {
            sighting_id: Sighting(
                sighting_id=sighting_id,
                sighting_date=dates_by_sighting_id[sighting_id],
                photos=tuple(photos),
            )
            for sighting_id, photos in photos_by_sighting_id.items()
        }
        self._names_by_sighting_id = names_by_sighting_id
        self.photo_store: PhotoStore = _FilesystemPhotoStore(paths_by_photo_id)

    @staticmethod
    def _parse_uuid4(value: str, field_name: str, row_number: int) -> UUID:
        """Parse a canonical UUIDv4 metadata value."""
        try:
            parsed = UUID(value)
        except ValueError:
            raise ValueError(f"Invalid {field_name} at metadata row {row_number}: {value!r}") from None
        if parsed.version != 4:
            raise ValueError(f"Invalid {field_name} at metadata row {row_number}: not UUIDv4")
        return parsed

    def photo(self, photo_id: UUID) -> Photo:
        """Resolve a neutral Photo by permanent photo ID."""
        try:
            return self._photos_by_id[photo_id]
        except KeyError:
            raise KeyError(f"Unknown photo_id: {photo_id}") from None

    def sighting(self, sighting_id: UUID) -> Sighting:
        """Resolve a neutral Sighting by permanent sighting ID."""
        try:
            return self._sightings_by_id[sighting_id]
        except KeyError:
            raise KeyError(f"Unknown sighting_id: {sighting_id}") from None

    def known_elephant_name(self, sighting_id: UUID) -> str:
        """Resolve the private known-elephant name for a sighting ID."""
        try:
            return self._names_by_sighting_id[sighting_id]
        except KeyError:
            raise KeyError(f"Unknown sighting_id: {sighting_id}") from None

    def iter_photos(self) -> Iterator[Photo]:
        """Iterate over every Photo."""
        yield from self._photos_by_id.values()

    def iter_sightings(self) -> Iterator[Sighting]:
        """Iterate over every Sighting."""
        yield from self._sightings_by_id.values()
