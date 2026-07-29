"""Load the known-elephant gallery from precomputed tear profiles."""

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from elephant_id.image.transforms import apply_crop

GALLERY_DISPLAY_CROP_PAD = 0.15


@dataclass(frozen=True)
class GalleryData:
    """Row-aligned tear profiles and metadata for the known-elephant catalog."""

    profiles: np.ndarray
    photo_ids: tuple[str, ...]
    identities: tuple[str, ...]
    sides: tuple[str, ...]
    dates: tuple[str, ...]
    crop_paths: tuple[str | None, ...]
    display_crop_paths: tuple[str | None, ...]
    source_paths: tuple[str | None, ...]

    def __post_init__(self) -> None:
        """Validate that every metadata column matches the profile rows.

        Raises:
            ValueError: If any column length differs from the profile count.
        """
        row_count = len(self.profiles)
        columns = (
            self.photo_ids,
            self.identities,
            self.sides,
            self.dates,
            self.crop_paths,
            self.display_crop_paths,
            self.source_paths,
        )
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
    manifest_entries = _manifest_entries_by_photo_side(manifest_path)

    photo_ids = tuple(str(photo_id) for photo_id in cached["photo_ids"])
    sides = tuple(str(side) for side in cached["sides"])
    gallery = GalleryData(
        profiles=np.asarray(cached["profiles"], dtype=np.float64),
        photo_ids=photo_ids,
        identities=tuple(str(identity) for identity in cached["identities"]),
        sides=sides,
        dates=tuple(str(date) for date in cached["dates"]),
        crop_paths=tuple(
            (manifest_entries.get((photo_id, side)) or {}).get("crop_path")
            for photo_id, side in zip(photo_ids, sides, strict=True)
        ),
        display_crop_paths=tuple(
            (manifest_entries.get((photo_id, side)) or {}).get("display_crop_path")
            for photo_id, side in zip(photo_ids, sides, strict=True)
        ),
        source_paths=tuple(
            (manifest_entries.get((photo_id, side)) or {}).get("source_path")
            for photo_id, side in zip(photo_ids, sides, strict=True)
        ),
    )
    logger.info(
        f"Loaded gallery: {len(gallery.photo_ids)} profiles, "
        f"{len(set(gallery.identities))} elephants"
    )
    return gallery


def _manifest_entries_by_photo_side(manifest_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Map (photo identifier, ear side) to display image paths from the manifest."""
    if not manifest_path.exists():
        logger.warning(f"High-quality manifest not found: {manifest_path}")
        return {}
    entries: dict[tuple[str, str], dict[str, str]] = {}
    manifest_root = manifest_path.parent
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            crop_path = _resolve_manifest_path(manifest_root, row.get("exported_abs_path") or row.get("exported_path"))
            source_path = _resolve_manifest_path(manifest_root, row.get("source_abs_path") or row.get("source_image_path"))
            if crop_path is None and source_path is None:
                continue
            key = (row["photo_identifier"], row["side"])
            entry = {
                "crop_path": str(crop_path) if crop_path is not None else "",
                "display_crop_path": str(crop_path) if crop_path is not None else "",
                "source_path": str(source_path) if source_path is not None else "",
            }
            padded_path = _write_display_crop(manifest_root, row, source_path)
            if padded_path is not None:
                entry["display_crop_path"] = str(padded_path)
            entries[key] = {name: value for name, value in entry.items() if value}
    return entries


def _resolve_manifest_path(manifest_root: Path, value: str | None) -> Path | None:
    """Resolve a manifest path value if it points at an existing file."""
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = manifest_root / path
    return path if path.exists() else None


def _write_display_crop(
    manifest_root: Path,
    row: dict[str, str],
    source_path: Path | None,
) -> Path | None:
    """Write a 15%-padded catalog display crop when source geometry is available."""
    if source_path is None:
        return None
    try:
        crop_xyxy = tuple(
            float(row[key]) for key in ("crop_x1", "crop_y1", "crop_x2", "crop_y2")
        )
    except (KeyError, TypeError, ValueError):
        return None
    image = cv2.imread(str(source_path))
    if image is None:
        return None
    display_path = (
        manifest_root
        / "images_padded"
        / row["side"]
        / row["identity"]
        / f"{row['photo_identifier']}.jpg"
    )
    if display_path.exists():
        return display_path
    try:
        display_path.parent.mkdir(parents=True, exist_ok=True)
        crop = apply_crop(image, crop_xyxy, pad=GALLERY_DISPLAY_CROP_PAD)
        if cv2.imwrite(str(display_path), crop):
            return display_path
    except ValueError as error:
        logger.warning(f"Could not write padded catalog crop for {row.get('photo_identifier')}: {error}")
    return None
