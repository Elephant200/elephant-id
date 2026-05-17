from collections import OrderedDict
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from elephant_id.domain import Photo, SeekCode, Sighting


class Dataset:
    """Interface to the SEEK elephant ID dataset on disk."""

    def __init__(
        self,
        dataset_root: Path,
        metadata_path: Path,
        image_cache_size: int = 32,
    ) -> None:
        """Validate paths and prepare lazy metadata loading.

        Args:
            dataset_root: Directory containing the image files.
            metadata_path: Path to the metadata CSV.
            image_cache_size: Max number of decoded images to keep in memory.
        """
        self.dataset_root: Path = dataset_root
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")
        if not self.dataset_root.is_dir():
            raise NotADirectoryError(f"Dataset root is not a directory: {self.dataset_root}")

        self.metadata_path: Path = metadata_path
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata path does not exist: {self.metadata_path}")
        if self.metadata_path.suffix != ".csv":
            raise ValueError(f"Metadata path must be a CSV file: {self.metadata_path}")

        self.metadata: pd.DataFrame | None = None # Lazily loaded
        self._image_cache: OrderedDict[str, Image.Image] = OrderedDict()
        self.image_cache_size: int = image_cache_size

    def path_for(self, photo: Photo) -> Path:
        """Resolve a photo's image path against the dataset root.

        Args:
            photo: The photo whose path to resolve.

        Returns:
            Absolute path to the photo's image file.
        """
        return self.dataset_root / photo.image_path

    def get_photo(self, identifier: str) -> Photo:
        """Look up a single photo by identifier.

        Args:
            identifier: The photo's identifier (filename stem).

        Returns:
            The matching Photo.
        """
        self._ensure_loaded()

        rows = self.metadata[self.metadata["identifier"] == identifier]
        if rows.empty:
            raise KeyError(f"No photo with identifier: {identifier}")
        row = rows.iloc[0]
        return Photo(
            identifier=row["identifier"],
            image_path=Path(row["image_path"]),
            elephant_name=row["name"],
            sighting_id=f"{row['name']}_{row['date'].isoformat()}",
        )

    def iter_photos(self) -> Iterator[Photo]:
        """Yield every photo in CSV row order.

        Yields:
            Each Photo in the dataset.
        """
        self._ensure_loaded()
        for _, row in self.metadata.iterrows():
            yield Photo(
                identifier=row["identifier"],
                image_path=Path(row["image_path"]),
                elephant_name=row["name"],
                sighting_id=f"{row['name']}_{row['date'].isoformat()}",
            )

    def get_sighting(self, elephant_name: str, sighting_date: date) -> Sighting:
        """Look up the sighting for an elephant on a given date.

        Args:
            elephant_name: The elephant's name.
            sighting_date: The date of the sighting.

        Returns:
            The Sighting containing every photo for that elephant on that date.
        """
        self._ensure_loaded()

        mask = (
            (self.metadata["name"] == elephant_name)
            & (self.metadata["date"] == sighting_date)
        )
        rows = self.metadata[mask]
        if rows.empty:
            raise KeyError(
                f"No sighting for {elephant_name} on {sighting_date.isoformat()}"
            )

        sighting_id = f"{elephant_name}_{sighting_date.isoformat()}"
        photos = tuple(
            Photo(
                identifier=row["identifier"],
                image_path=Path(row["image_path"]),
                elephant_name=elephant_name,
                sighting_id=sighting_id,
            )
            for _, row in rows.iterrows()
        )
        return Sighting(
            elephant_name=elephant_name,
            sighting_date=sighting_date,
            sighting_id=sighting_id,
            photos=photos,
        )

    def iter_sightings(self) -> Iterator[Sighting]:
        """Yield one Sighting per unique (elephant, date), in CSV row order.

        Yields:
            Each Sighting in the dataset.
        """
        self._ensure_loaded()
        for (name, sighting_date), rows in self.metadata.groupby(
            ["name", "date"], sort=False
        ):
            sighting_id = f"{name}_{sighting_date.isoformat()}"
            photos = tuple(
                Photo(
                    identifier=row["identifier"],
                    image_path=Path(row["image_path"]),
                    elephant_name=name,
                    sighting_id=sighting_id,
                )
                for _, row in rows.iterrows()
            )
            yield Sighting(
                elephant_name=name,
                sighting_date=sighting_date,
                sighting_id=sighting_id,
                photos=photos,
            )

    def get_ground_truth(self, sighting: Sighting) -> SeekCode:
        """Return the SEEK code recorded for a sighting.

        Assumes all rows for the sighting share one code.

        Args:
            sighting: The sighting to look up.

        Returns:
            The parsed SeekCode for the sighting.
        """
        self._ensure_loaded()
        rows = self.metadata[(self.metadata["name"] == sighting.elephant_name) & (self.metadata["date"] == sighting.sighting_date)]
        if rows.empty:
            raise KeyError(
                f"Sighting not found: {sighting.elephant_name} on {sighting.sighting_date.isoformat()}"
            )
        code = rows.iloc[0]["seek_code"]
        if pd.isna(code) or code == "":
            raise ValueError(
                f"Sighting has no seek code: {sighting.elephant_name} on {sighting.sighting_date.isoformat()}"
            )
        return SeekCode.from_str(code)

    def read_image(
        self,
        photo: Photo,
        crop: tuple[int, int, int, int] | None = None,
    ) -> Image.Image:
        """Load a photo's image as RGB, using an LRU cache.

        Args:
            photo: The photo to load.
            crop: The crop to apply to the image, in xywh coordinates. Defaults to None.

        Returns:
            A fresh RGB copy of the image
        """
        key = f"{photo.identifier}_{crop}" if crop else photo.identifier

        if key in self._image_cache:
            image = self._image_cache.pop(key)
            self._image_cache[key] = image
            return image.copy()

        with Image.open(self.path_for(photo)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if crop:
                image = image.crop(crop)
            loaded = image.copy()

        self._image_cache[key] = loaded

        if len(self._image_cache) > self.image_cache_size:
            self._image_cache.popitem(last=False)

        return loaded.copy()

    def clear_image_cache(self) -> None:
        """Empty the image cache."""
        self._image_cache.clear()

    def _ensure_loaded(self) -> None:
        """Lazily load the metadata CSV on first access."""
        if self.metadata is not None:
            return
        self.metadata = pd.read_csv(self.metadata_path, parse_dates=["date"])
        self.metadata["date"] = self.metadata["date"].dt.date
