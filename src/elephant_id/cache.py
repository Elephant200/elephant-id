"""Generic JSON cache for immutable named producers."""

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from elephant_id.constants import DEFAULT_CACHE_ROOT


def _validate_path_segment(value: str, label: str) -> None:
    """Validate one portable, non-empty cache path segment."""
    if value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be one non-empty path segment: {value!r}")


class CacheManager:
    """Persist JSON records from immutable named producers."""

    def __init__(
        self,
        cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    ) -> None:
        """Initialize cache persistence rooted at one directory.

        Args:
            cache_root: Root cache directory. Defaults to project cache.
        """
        self.cache_root = cache_root.resolve()

    def path_for(self, producer_id: str, key: str) -> Path:
        """Return the contained JSON path for a producer and input key.

        Raises:
            ValueError: If either identity is unsafe or escapes the cache root.
        """
        _validate_path_segment(producer_id, "cache producer ID")
        _validate_path_segment(key, "cache key")
        producer_dir = self.cache_root / producer_id
        if not producer_dir.resolve().is_relative_to(self.cache_root):
            raise ValueError(f"Cache producer ID escapes cache root: {producer_id!r}")
        path = producer_dir / f"{key}.json"
        if not path.resolve().is_relative_to(producer_dir.resolve()):
            raise ValueError(f"Cache key escapes producer directory: {key!r}")
        return path

    def exists(self, producer_id: str, key: str) -> bool:
        """Return whether a producer record is cached.

        Raises:
            ValueError: If either identity is unsafe.
        """
        return self.path_for(producer_id, key).exists()

    def load(self, producer_id: str, key: str) -> dict[str, object]:
        """Load one producer record.

        Raises:
            ValueError: If an identity is unsafe or the record is not a JSON object.
            UnicodeDecodeError: If the record is not valid UTF-8.
            FileNotFoundError: If the record does not exist.
        """
        path = self.path_for(producer_id, key)
        with path.open(encoding="utf-8") as file:
            record = json.load(file)
        if not isinstance(record, dict):
            raise ValueError(f"Cache record must be a JSON object: {producer_id}/{key}")
        return record

    def save(
        self,
        producer_id: str,
        key: str,
        value: dict[str, object],
    ) -> None:
        """Atomically save one producer record.

        Raises:
            ValueError: If an identity is unsafe or the value is circular.
            TypeError: If the value contains non-JSON-serializable data.
        """
        path = self.path_for(producer_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(value, file, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def get_or_compute(
        self,
        producer_id: str,
        key: str,
        compute_fn: Callable[[], dict[str, object]],
    ) -> dict[str, object]:
        """Load a producer record or compute and persist it on a miss.

        Args:
            producer_id: Stable identity of the deterministic processor.
            key: Caller-supplied opaque record key.
            compute_fn: Function that computes the record on a miss.

        Returns:
            The loaded or computed record.

        Raises:
            ValueError: If either identity is unsafe.
        """
        self.path_for(producer_id, key)
        if self.exists(producer_id, key):
            try:
                cached = self.load(producer_id, key)
            except (UnicodeDecodeError, ValueError):
                logger.warning(f"Ignoring corrupt cache record: {producer_id}/{key}")
            else:
                logger.debug(f"Cache hit: {producer_id}/{key}")
                return cached
        results = compute_fn()
        logger.debug(f"Cache miss: {producer_id}/{key}")
        self.save(producer_id, key, results)
        return results
