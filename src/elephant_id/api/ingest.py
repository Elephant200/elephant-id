"""Turn a folder of sighting photos into tear profiles for matching.

Photos are indexed in place (never copied or modified). Analysis prefers the
real ``PhotoAnalyzer`` pipeline, which is served from the on-disk model cache
for dataset photos; when the analyzer is unavailable or fails on a photo, the
precomputed gallery profile for that photo identifier is used instead so the
demo path stays fully offline.

Besides the tear profiles used for matching, ingest exports the review
evidence described in docs/pipeline.md §4.2: an annotated overlay of every
model detection, per-photo view/age/gender/tusk suggestions, and the per-ear
tear-depth profile (the matching embedding) for plotting.
"""

import csv
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from elephant_id.api import overlays
from elephant_id.api.gallery import GalleryData
from elephant_id.api.profiles import plot_profile
from elephant_id.constants import TEAR_PROFILE_BINS
from elephant_id.dataset import Dataset
from elephant_id.matching import tear_mass

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PHOTO_STEM_PATTERN = re.compile(r"^(?P<name>.+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<seq>\d+)$")


@dataclass(frozen=True)
class EarResult:
    """One usable ear extracted from a query photo."""

    side: str
    crop_path: str | None
    profile: tuple[float, ...] = field(default=())
    mass: float = 0.0


@dataclass(frozen=True)
class PhotoResult:
    """Ingest outcome and review evidence for one file in the folder."""

    file_name: str
    photo_id: str | None
    status: str  # "analyzed" | "precomputed" | "skipped"
    detail: str
    photo_path: str | None = None
    overlay_path: str | None = None
    view: str | None = None
    age: dict | None = None
    gender: dict | None = None
    tusks: tuple[dict, ...] = field(default=())
    ears: tuple[EarResult, ...] = field(default=())

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class SightingProfiles:
    """Extracted profiles plus per-photo outcomes for one ingested folder."""

    profiles: np.ndarray
    sides: tuple[str, ...]
    photo_ids: tuple[str, ...]
    crop_paths: tuple[str | None, ...]
    photos: tuple[PhotoResult, ...]


def list_photo_files(folder: Path) -> list[Path]:
    """Return the folder's image files in name order.

    Raises:
        NotADirectoryError: If the folder does not exist or is not a directory.
    """
    if not folder.is_dir():
        raise NotADirectoryError(f"Sighting folder not found: {folder}")
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def ingest_sighting(
    folder: Path,
    work_dir: Path,
    gallery: GalleryData,
    cache_root: Path,
    progress: Callable[[int, int], None] | None = None,
) -> SightingProfiles:
    """Extract tear profiles and review evidence for every photo in a folder.

    Args:
        folder: Folder of photos for one sighting of one elephant.
        work_dir: Writable directory for this sighting's metadata and crops.
        gallery: Precomputed profiles used as the per-photo fallback.
        cache_root: Model cache root for the analyzer services.
        progress: Optional callback receiving (processed, total) after each photo.

    Returns:
        Row-aligned profiles and the per-photo ingest report.

    Raises:
        NotADirectoryError: If the folder does not exist.
    """
    files = list_photo_files(folder)
    parsed = {path: PHOTO_STEM_PATTERN.match(path.stem) for path in files}
    dataset = _folder_dataset(folder, work_dir, parsed)
    analyzer = _make_analyzer(dataset, cache_root)
    fallback = _fallback_rows(gallery)

    profiles: list[np.ndarray] = []
    sides: list[str] = []
    photo_ids: list[str] = []
    crop_paths: list[str | None] = []
    photos: list[PhotoResult] = []
    for index, path in enumerate(files):
        result, rows = _ingest_photo(path, parsed[path], dataset, analyzer, fallback, work_dir)
        photos.append(result)
        for side, profile, crop_path in rows:
            profiles.append(profile)
            sides.append(side)
            photo_ids.append(result.photo_id or path.stem)
            crop_paths.append(crop_path)
        if progress is not None:
            progress(index + 1, len(files))

    profile_array = (
        np.vstack(profiles) if profiles else np.zeros((0, TEAR_PROFILE_BINS))
    )
    logger.info(
        f"Ingested {folder}: {len(files)} files, {len(profiles)} ear profiles"
    )
    return SightingProfiles(
        profiles=profile_array,
        sides=tuple(sides),
        photo_ids=tuple(photo_ids),
        crop_paths=tuple(crop_paths),
        photos=tuple(photos),
    )


def analyze_single_photo(
    path: Path,
    work_dir: Path,
    cache_root: Path,
) -> PhotoResult:
    """Run the full analyzer on one standalone photo (development Lab page).

    The photo must already be stored at ``path`` with a
    ``{Name}_{YYYY-MM-DD}_{seq}`` stem so cache keys stay stable.

    Raises:
        ValueError: If the filename stem does not parse or analysis fails.
    """
    stem_match = PHOTO_STEM_PATTERN.match(path.stem)
    if stem_match is None:
        raise ValueError(f"Photo stem must parse as name_date_seq: {path.stem}")
    dataset = _folder_dataset(path.parent, work_dir, {path: stem_match})
    analyzer = _make_analyzer(dataset, cache_root)
    if analyzer is None:
        raise ValueError("Photo analyzer is unavailable (missing local AI setup)")
    try:
        analysis = analyzer.analyze(dataset.get_photo(path.stem))
    except Exception as error:
        logger.warning(f"Lab analysis failed for {path.stem}: {error}")
        raise ValueError(
            "Could not analyze this photo offline. Photos outside the local "
            "model cache need the live segmentation service — for the offline "
            "demo, use a photo from the dataset."
        ) from error
    if analysis is None:
        raise ValueError("No usable elephant evidence found in the photo")
    result, _ = _photo_result_from_analysis(path, path.stem, analysis, dataset, work_dir)
    return result


def _ingest_photo(
    path: Path,
    stem_match: re.Match[str] | None,
    dataset: Dataset | None,
    analyzer: object | None,
    fallback: dict[str, list[tuple[str, np.ndarray, str | None]]],
    work_dir: Path,
) -> tuple[PhotoResult, list[tuple[str, np.ndarray, str | None]]]:
    """Extract (side, profile, crop path) rows for one photo file."""
    if stem_match is None or dataset is None:
        return (
            PhotoResult(
                file_name=path.name,
                photo_id=None,
                status="skipped",
                detail="Filename is not in {Name}_{YYYY-MM-DD}_{seq} format",
                photo_path=str(path),
            ),
            [],
        )

    photo_id = path.stem
    if analyzer is not None:
        try:
            analysis = analyzer.analyze(dataset.get_photo(photo_id))
        except Exception as error:
            logger.warning(f"Analysis failed for {photo_id}: {error}")
            analysis = None
        if analysis is not None and analysis["ears"]:
            return _photo_result_from_analysis(path, photo_id, analysis, dataset, work_dir)

    precomputed = fallback.get(photo_id, [])
    if precomputed:
        ears = tuple(
            EarResult(
                side=side,
                crop_path=crop,
                profile=plot_profile(profile),
                mass=float(tear_mass(profile)[0]),
            )
            for side, profile, crop in precomputed
        )
        return (
            PhotoResult(
                file_name=path.name,
                photo_id=photo_id,
                status="precomputed",
                detail=f"{_plural(len(precomputed), 'profile')} from the gallery cache",
                photo_path=str(path),
                ears=ears,
            ),
            precomputed,
        )

    return (
        PhotoResult(
            file_name=path.name,
            photo_id=photo_id,
            status="skipped",
            detail="No usable ear evidence found",
            photo_path=str(path),
        ),
        [],
    )


def _photo_result_from_analysis(
    path: Path,
    photo_id: str,
    analysis: dict,
    dataset: Dataset,
    work_dir: Path,
) -> tuple[PhotoResult, list[tuple[str, np.ndarray, str | None]]]:
    """Build the full review evidence package for one analyzed photo."""
    image = dataset.read_image(dataset.get_photo(photo_id))
    overlay_path = _export_image(
        overlays.annotate_photo(image, analysis), work_dir / "overlays", f"{photo_id}.jpg"
    )

    rows: list[tuple[str, np.ndarray, str | None]] = []
    ears: list[EarResult] = []
    for ear_data in analysis["ears"]:
        profile = np.asarray(ear_data["tear_profile"].profile, dtype=np.float64)
        if profile.shape != (TEAR_PROFILE_BINS,):
            logger.warning(f"Unexpected profile shape for {photo_id}: {profile.shape}")
            continue
        ear = ear_data["ear"]
        crop_path = _export_image(
            overlays.annotate_ear_crop(image, ear),
            work_dir / "crops",
            f"{photo_id}_{ear.side}.jpg",
        )
        rows.append((str(ear.side), profile, crop_path))
        ears.append(
            EarResult(
                side=str(ear.side),
                crop_path=crop_path,
                profile=plot_profile(profile),
                mass=float(tear_mass(profile)[0]),
            )
        )

    age = analysis.get("age")
    gender = analysis.get("gender")
    result = PhotoResult(
        file_name=path.name,
        photo_id=photo_id,
        status="analyzed",
        detail=f"{_plural(len(rows), 'usable ear')}",
        photo_path=str(path),
        overlay_path=overlay_path,
        view=str(analysis["view"]),
        age=_rounded(age) if isinstance(age, dict) else None,
        gender=_rounded(gender) if isinstance(gender, dict) else None,
        tusks=tuple(
            {"side": tusk["side"], "confidence": round(float(tusk["confidence"]), 3)}
            for tusk in analysis["tusks"]
        ),
        ears=tuple(ears),
    )
    return result, rows


def _plural(count: int, noun: str) -> str:
    """Format a count with a naturally pluralized noun."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _rounded(values: dict) -> dict:
    """Round numeric leaves for compact JSON storage."""
    rounded: dict = {}
    for key, value in values.items():
        if isinstance(value, int | float):
            rounded[key] = round(float(value), 3)
        elif isinstance(value, list) and all(isinstance(v, int | float) for v in value):
            rounded[key] = [round(float(v), 3) for v in value]
        else:
            rounded[key] = value
    return rounded


def _export_image(image: np.ndarray, directory: Path, file_name: str) -> str | None:
    """Write an evidence image, returning its path (or None on failure)."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / file_name
        cv2.imwrite(str(output_path), image)
        return str(output_path)
    except Exception as error:
        logger.warning(f"Could not export evidence image {file_name}: {error}")
        return None


def _folder_dataset(
    folder: Path,
    work_dir: Path,
    parsed: dict[Path, re.Match[str] | None],
) -> Dataset | None:
    """Index the folder as a Dataset via a generated metadata CSV."""
    rows = [
        {
            "identifier": path.stem,
            "image_path": path.name,
            "name": match["name"],
            "date": match["date"],
        }
        for path, match in parsed.items()
        if match is not None
    ]
    if not rows:
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = work_dir / "metadata.csv"
    with metadata_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["identifier", "image_path", "name", "date"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["identifier"]))
    return Dataset(dataset_root=folder, metadata_path=metadata_path)


def _make_analyzer(dataset: Dataset | None, cache_root: Path) -> object | None:
    """Create a PhotoAnalyzer, or None when local AI services are unavailable."""
    if dataset is None:
        return None
    try:
        from elephant_id.coding.photo_analyzer import PhotoAnalyzer

        return PhotoAnalyzer(dataset=dataset, cache_root=cache_root)
    except Exception as error:
        logger.warning(
            f"Photo analyzer unavailable ({error}); using precomputed profiles only"
        )
        return None


def _fallback_rows(
    gallery: GalleryData,
) -> dict[str, list[tuple[str, np.ndarray, str | None]]]:
    """Group precomputed gallery rows by photo identifier."""
    rows: dict[str, list[tuple[str, np.ndarray, str | None]]] = {}
    for index, photo_id in enumerate(gallery.photo_ids):
        rows.setdefault(photo_id, []).append(
            (gallery.sides[index], gallery.profiles[index], gallery.crop_paths[index])
        )
    return rows
