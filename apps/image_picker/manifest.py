"""Manifest read/write and crop export for the matching image picker.

The manifest is the single source of truth for picks. A pick is an upsert
keyed by ``(side, identity, sighting_date)``: recording one writes its row and
exports two images -- a full-frame copy under ``images_full`` and a tight ear
crop under ``images_crop`` -- and replacing it deletes the stale exports first.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from elephant_id.dataset import Dataset

from .analysis import EarCandidate, encode_crop_jpeg
from .config import CODED_ROOT, HIGH_QUALITY_ROOT

MANIFEST_FIELDS = [
    "side",
    "identity",
    "photo_identifier",
    "sighting_date",
    "source_image_path",
    "source_abs_path",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "crop_confidence",
    "exported_path",
    "exported_abs_path",
    "exported_at",
]


def safe_path_component(value: str) -> str:
    """Return a conservative single-path-segment name for an identity."""
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip(" .")
    return cleaned or "identity"


class ManifestStore:
    """Own manifest rows and their exported crop/full images on disk."""

    def __init__(
        self,
        dataset: Dataset,
        root: Path = HIGH_QUALITY_ROOT,
    ) -> None:
        """Derive the manifest and export paths under ``root``; touch nothing yet."""
        self.dataset = dataset
        self.root = root
        self.manifest_path = root / "manifest.csv"
        self.full_root = root / "images_full"
        self.crop_root = root / "images_crop"
        self._lock = threading.Lock()

    def picks_for_identity(self, identity: str) -> dict[tuple[str, str], str]:
        """Return ``{(side, sighting_date): photo_identifier}`` for one elephant."""
        return self.picks_by_identity().get(identity, {})

    def picks_by_identity(self) -> dict[str, dict[tuple[str, str], str]]:
        """Return every elephant's picks in a single manifest read.

        Reads the manifest once and groups rows by identity so callers that need
        every elephant's picks (for example the eligible-list progress badges)
        avoid re-reading the whole file per elephant.
        """
        by_identity: dict[str, dict[tuple[str, str], str]] = {}
        for row in self._read_rows():
            identity = row.get("identity")
            if not identity:
                continue
            picks = by_identity.setdefault(identity, {})
            picks[(row["side"], row["sighting_date"])] = row["photo_identifier"]
        return by_identity

    def record_pick(self, candidate: EarCandidate) -> None:
        """Upsert a candidate as the canonical pick for its side and sighting."""
        key = (candidate.side, candidate.identity, candidate.sighting_date)
        with self._lock:
            rows = self._read_rows()
            # Delete any prior exports for this key before writing the new ones,
            # so re-picking the same photo does not delete its fresh export.
            for row in rows:
                if (row["side"], row["identity"], row["sighting_date"]) == key:
                    self._delete_exports(row)
            new_row = self._export(candidate)
            kept: list[dict] = []
            replaced = False
            for row in rows:
                if (row["side"], row["identity"], row["sighting_date"]) == key:
                    if not replaced:
                        kept.append(new_row)
                        replaced = True
                else:
                    kept.append(row)
            if not replaced:
                kept.append(new_row)
            self._write_rows(kept)
        logger.info(
            f"Recorded pick {candidate.side} {candidate.identity} "
            f"{candidate.sighting_date}: {candidate.photo_identifier}"
        )

    def _export(self, candidate: EarCandidate) -> dict:
        """Write the full-frame and crop images and build the manifest row."""
        photo = self.dataset.get_photo(candidate.photo_identifier)
        source = self.dataset.path_for(photo)
        image = self.dataset.read_image(photo)

        safe_identity = safe_path_component(candidate.identity)
        full_dir = self.full_root / candidate.side / safe_identity
        crop_dir = self.crop_root / candidate.side / safe_identity
        full_dir.mkdir(parents=True, exist_ok=True)
        crop_dir.mkdir(parents=True, exist_ok=True)

        full_path = full_dir / f"{candidate.photo_identifier}{source.suffix}"
        crop_path = crop_dir / f"{candidate.photo_identifier}.jpg"

        shutil.copy2(source, full_path)
        crop_path.write_bytes(encode_crop_jpeg(image, candidate.crop_xyxy))

        x1, y1, x2, y2 = candidate.crop_xyxy
        return {
            "side": candidate.side,
            "identity": candidate.identity,
            "photo_identifier": candidate.photo_identifier,
            "sighting_date": candidate.sighting_date,
            "source_image_path": candidate.image_path,
            "source_abs_path": str((CODED_ROOT / candidate.image_path).resolve()),
            "crop_x1": f"{x1:.3f}",
            "crop_y1": f"{y1:.3f}",
            "crop_x2": f"{x2:.3f}",
            "crop_y2": f"{y2:.3f}",
            "crop_confidence": f"{candidate.confidence:.6f}",
            "exported_path": crop_path.relative_to(self.root).as_posix(),
            "exported_abs_path": str(crop_path.resolve()),
            "exported_at": datetime.now(UTC).isoformat(),
        }

    def _delete_exports(self, row: dict) -> None:
        """Delete the crop and full-frame files for a replaced manifest row."""
        safe_identity = safe_path_component(row["identity"])
        photo_identifier = row["photo_identifier"]
        crop = self.crop_root / row["side"] / safe_identity / f"{photo_identifier}.jpg"
        self._delete_file(crop)
        full_dir = self.full_root / row["side"] / safe_identity
        for full in full_dir.glob(f"{photo_identifier}.*"):
            self._delete_file(full)

    def _delete_file(self, path: Path) -> None:
        """Delete a file, refusing anything outside the export root."""
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root.resolve()):
            logger.warning(f"Refusing to delete export outside the export root: {resolved}")
            return
        if resolved.is_file():
            try:
                resolved.unlink()
            except OSError as error:
                logger.warning(f"Could not delete stale export {resolved}: {error}")

    def _read_rows(self) -> list[dict]:
        """Return current manifest rows, or an empty list when absent."""
        if not self.manifest_path.is_file():
            return []
        with self.manifest_path.open(newline="") as file:
            return list(csv.DictReader(file))

    def _write_rows(self, rows: list[dict]) -> None:
        """Atomically rewrite the manifest with the given rows."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.manifest_path.parent,
            prefix=f".{self.manifest_path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", newline="") as file:
                writer = csv.DictWriter(
                    file, fieldnames=MANIFEST_FIELDS, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(rows)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, self.manifest_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
