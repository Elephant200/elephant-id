"""Load the known-elephant gallery from precomputed tear profiles."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class GalleryData:
    """Row-aligned tear profiles and metadata for the known-elephant catalog."""

    profiles: np.ndarray
    photo_ids: tuple[str, ...]
    identities: tuple[str, ...]
    sides: tuple[str, ...]
    dates: tuple[str, ...]
    crop_paths: tuple[str | None, ...]

    def __post_init__(self) -> None:
        """Validate that every metadata column matches the profile rows.

        Raises:
            ValueError: If any column length differs from the profile count.
        """
        row_count = len(self.profiles)
        columns = (self.photo_ids, self.identities, self.sides, self.dates, self.crop_paths)
        if any(len(column) != row_count for column in columns):
            raise ValueError("Gallery metadata columns must match the profile row count")


def load_gallery(profiles_path: Path, manifest_path: Path) -> GalleryData:
    """Load gallery profiles and join ear-crop image paths from the manifest.

    Args:
        profiles_path: Profile cache ``.npz`` written by ``scripts/evaluation.py``.
        manifest_path: High-quality export manifest CSV; crop paths resolve
            relative to its parent directory.

    Returns:
        Gallery rows aligned with the profile array.

    Raises:
        FileNotFoundError: If the profile cache is missing.
    """
    if not profiles_path.exists():
        raise FileNotFoundError(f"Gallery profile cache not found: {profiles_path}")
    cached = np.load(profiles_path, allow_pickle=False)
    crops = _crop_paths_by_photo_side(manifest_path)

    photo_ids = tuple(str(photo_id) for photo_id in cached["photo_ids"])
    sides = tuple(str(side) for side in cached["sides"])
    gallery = GalleryData(
        profiles=np.asarray(cached["profiles"], dtype=np.float64),
        photo_ids=photo_ids,
        identities=tuple(str(identity) for identity in cached["identities"]),
        sides=sides,
        dates=tuple(str(date) for date in cached["dates"]),
        crop_paths=tuple(
            crops.get((photo_id, side))
            for photo_id, side in zip(photo_ids, sides, strict=True)
        ),
    )
    logger.info(
        f"Loaded gallery: {len(gallery.photo_ids)} profiles, "
        f"{len(set(gallery.identities))} elephants"
    )
    return gallery


def _crop_paths_by_photo_side(manifest_path: Path) -> dict[tuple[str, str], str]:
    """Map (photo identifier, ear side) to the exported ear-crop absolute path."""
    if not manifest_path.exists():
        logger.warning(f"High-quality manifest not found: {manifest_path}")
        return {}
    crops: dict[tuple[str, str], str] = {}
    manifest_root = manifest_path.parent
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            crop_path = manifest_root / row["exported_path"]
            if crop_path.exists():
                crops[(row["photo_identifier"], row["side"])] = str(crop_path)
    return crops
