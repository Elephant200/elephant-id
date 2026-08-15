"""Generic JSON cache for immutable named producers."""

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from loguru import logger

from elephant_id.constants import DEFAULT_CACHE_ROOT

CacheMode = Literal["read_write", "read_only", "disabled"]


def _validate_path_segment(value: str, label: str) -> None:
    """Validate one portable, non-empty cache path segment."""
    if value in {"", ".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be one non-empty path segment: {value!r}")


class CacheManager:
    """Cache records from one immutable named producer."""

    def __init__(
        self,
        producer_name: str,
        cache_root: Path = Path(DEFAULT_CACHE_ROOT),
        mode: CacheMode | None = None,
    ) -> None:
        """Initialize the JSON cache manager for a producer.

        Args:
            producer_name: Immutable name of the record producer.
            cache_root: Root cache directory. Defaults to project cache.
            mode: Cache read and write behavior. Defaults to the
                ``ELEPHANT_ID_CACHE_MODE`` environment variable or ``read_write``.

        Raises:
            ValueError: If the producer name is unsafe, escapes the cache root,
                or the configured mode is unsupported.
        """
        _validate_path_segment(producer_name, "cache producer name")
        self.producer_name = producer_name
        configured_mode = (
            mode
            if mode is not None
            else os.getenv("ELEPHANT_ID_CACHE_MODE", "read_write")
        )
        if configured_mode not in ("read_write", "read_only", "disabled"):
            raise ValueError(f"Unsupported cache mode: {configured_mode!r}")
        self.mode = configured_mode
        cache_root_resolved = cache_root.resolve()
        self.cache_dir: Path = cache_root_resolved / producer_name
        if not self.cache_dir.resolve().is_relative_to(cache_root_resolved):
            raise ValueError(f"Cache producer name escapes cache root: {producer_name!r}")
        if self.mode == "read_write":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        """Return the contained JSON path for a caller-supplied key.

        Raises:
            ValueError: If the key is not one safe path segment.
        """
        _validate_path_segment(key, "cache key")
        path = self.cache_dir / f"{key}.json"
        if not path.resolve().is_relative_to(self.cache_dir.resolve()):
            raise ValueError(f"Cache key escapes producer directory: {key!r}")
        return path

    def exists(self, key: str) -> bool:
        """Return whether a key is cached, or false when caching is disabled.

        Raises:
            ValueError: If the key is not one safe path segment.
        """
        path = self.path_for(key)
        if self.mode == "disabled":
            return False
        return path.exists()

    def load(self, key: str) -> dict[str, object]:
        """Load the record for a key.

        Raises:
            ValueError: If the key is unsafe or the record is invalid JSON or
                not a JSON object.
            UnicodeDecodeError: If the record is not valid UTF-8.
            PermissionError: If cache reads are disabled.
            FileNotFoundError: If the record does not exist.
        """
        path = self.path_for(key)
        if self.mode == "disabled":
            raise PermissionError(f"Cache reads are disabled: {self.producer_name}/{key}")
        with path.open(encoding="utf-8") as file:
            record = json.load(file)
        if not isinstance(record, dict):
            raise ValueError(f"Cache record must be a JSON object: {self.producer_name}/{key}")
        return record

    def save(self, key: str, value: dict[str, object]) -> None:
        """Atomically save a record, or do nothing when caching is disabled.

        Raises:
            ValueError: If the key is unsafe or the value contains a circular
                reference.
            TypeError: If the value contains non-JSON-serializable data.
            PermissionError: If the cache is read-only.
        """
        path = self.path_for(key)
        if self.mode == "disabled":
            return
        if self.mode == "read_only":
            raise PermissionError(f"Cache is read-only: {self.producer_name}/{key}")
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
        key: str,
        compute_fn: Callable[[], dict[str, object]],
    ) -> dict[str, object]:
        """Load a record or compute it according to the configured mode.

        Args:
            key: Caller-supplied opaque record key.
            compute_fn: Function that computes the record on a writable miss.

        Returns:
            The loaded or computed record.

        Raises:
            FileNotFoundError: If a read-only record is missing.
            ValueError: If the key is unsafe or a read-only record is corrupt.
        """
        self.path_for(key)
        if self.mode == "disabled":
            return compute_fn()
        if self.exists(key):
            try:
                cached = self.load(key)
            except (UnicodeDecodeError, ValueError) as error:
                if self.mode == "read_only":
                    raise ValueError(f"Corrupt read-only cache record: {self.producer_name}/{key}") from error
                logger.warning(f"Ignoring corrupt cache record: {self.producer_name}/{key}")
            else:
                logger.debug(f"Cache hit: {self.producer_name}/{key}")
                return cached
        if self.mode == "read_only":
            raise FileNotFoundError(f"Read-only cache miss: {self.producer_name}/{key}")
        results = compute_fn()
        logger.debug(f"Cache miss: {self.producer_name}/{key}")
        self.save(key, results)
        return results
