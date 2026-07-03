"""Persist Alphaphant sightings, match results, and review decisions."""

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger


class SightingStore:
    """JSON-backed store of ingested sightings under the sidecar data dir."""

    def __init__(self, data_dir: Path) -> None:
        """Create the store layout and load any existing records."""
        self.data_dir = data_dir
        self.sightings_dir = data_dir / "sightings"
        self.sightings_dir.mkdir(parents=True, exist_ok=True)
        self._store_path = data_dir / "store.json"
        self._lock = threading.Lock()
        self._records: dict[str, dict] = self._load()

    def create(self, folder: Path) -> dict:
        """Create a new sighting record for a folder ingest."""
        sighting_id = f"{folder.name}-{uuid.uuid4().hex[:8]}"
        record = {
            "sighting_id": sighting_id,
            "folder": str(folder),
            "folder_name": folder.name,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "analyzing",
            "progress": {"processed": 0, "total": 0},
            "photos": [],
            "profile_count": 0,
            "sides": [],
            "match": None,
            "decision": None,
        }
        with self._lock:
            self._records[sighting_id] = record
            self._save()
        logger.info(f"Created sighting {sighting_id} for {folder}")
        return dict(record)

    def update(self, sighting_id: str, **fields: object) -> dict:
        """Merge fields into a sighting record and persist.

        Raises:
            KeyError: If the sighting does not exist.
        """
        with self._lock:
            record = self._records[sighting_id]
            record.update(fields)
            self._save()
            return dict(record)

    def get(self, sighting_id: str) -> dict:
        """Return one sighting record.

        Raises:
            KeyError: If the sighting does not exist.
        """
        with self._lock:
            return dict(self._records[sighting_id])

    def list(self) -> list[dict]:
        """Return all sighting records, newest first."""
        with self._lock:
            return sorted(
                (dict(record) for record in self._records.values()),
                key=lambda record: record["created_at"],
                reverse=True,
            )

    def sighting_dir(self, sighting_id: str) -> Path:
        """Return the writable working directory for one sighting."""
        return self.sightings_dir / sighting_id

    def save_profiles(
        self,
        sighting_id: str,
        profiles: np.ndarray,
        sides: tuple[str, ...],
        photo_ids: tuple[str, ...],
        crop_paths: tuple[str | None, ...],
    ) -> None:
        """Persist a sighting's extracted profiles as an npz file."""
        directory = self.sighting_dir(sighting_id)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "profiles.npz",
            profiles=profiles,
            sides=np.asarray(sides),
            photo_ids=np.asarray(photo_ids),
            crop_paths=np.asarray([path or "" for path in crop_paths]),
        )

    def load_profiles(
        self, sighting_id: str
    ) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str | None, ...]]:
        """Load a sighting's extracted profiles.

        Raises:
            FileNotFoundError: If the sighting has no stored profiles.
        """
        path = self.sighting_dir(sighting_id) / "profiles.npz"
        if not path.exists():
            raise FileNotFoundError(f"No stored profiles for sighting {sighting_id}")
        data = np.load(path, allow_pickle=False)
        crop_paths = tuple(str(path) or None for path in data["crop_paths"])
        return (
            np.asarray(data["profiles"], dtype=np.float64),
            tuple(str(side) for side in data["sides"]),
            tuple(str(photo_id) for photo_id in data["photo_ids"]),
            crop_paths,
        )

    def _load(self) -> dict[str, dict]:
        """Read records from disk, tolerating a missing store file."""
        if not self._store_path.exists():
            return {}
        try:
            return json.loads(self._store_path.read_text())["sightings"]
        except (json.JSONDecodeError, KeyError) as error:
            logger.warning(f"Could not read {self._store_path}: {error}; starting empty")
            return {}

    def _save(self) -> None:
        """Atomically write records to disk. Caller must hold the lock."""
        payload = json.dumps({"sightings": self._records}, indent=2)
        temp_path = self._store_path.with_suffix(".json.tmp")
        temp_path.write_text(payload)
        temp_path.replace(self._store_path)
