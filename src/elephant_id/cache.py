"""General-purpose cache manager for AI models."""

import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from elephant_id.constants import DEFAULT_CACHE_ROOT


class CacheManager:
    """Cache AI model results as JSON files."""
    def __init__(
        self,
        namespace: str,
        cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    ) -> None:
        """Initialize the JSON cache manager for a namespace.

        Args:
            namespace: Cache namespace, such as the model name.
            cache_root: Root cache directory. Defaults to project cache.
        """
        self.namespace: str = namespace
        cache_root_resolved = cache_root.resolve()
        self.cache_dir: Path = cache_root_resolved / namespace
        if not self.cache_dir.resolve().is_relative_to(cache_root_resolved):
            raise ValueError(f"Cache namespace escapes cache root: {namespace!r}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        """Return the path for the given key.

        Raises:
            ValueError: If the key would escape the namespace directory.
        """
        path = self.cache_dir / f"{key}.json"
        if not path.resolve().is_relative_to(self.cache_dir.resolve()):
            raise ValueError(f"Cache key escapes namespace directory: {key!r}")
        return path

    def exists(self, key: str) -> bool:
        """Return whether the given key is cached."""
        return self.path_for(key).exists()

    def load(self, key: str) -> dict:
        """Load the results for the given key."""
        with open(self.path_for(key)) as f:
            return json.load(f)

    def save(self, key: str, value: dict) -> None:
        """Save the results for the given key.

        Uses a temporary file to avoid
        partially written files.
        """
        path = self.path_for(key)
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(value, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def delete(self, key: str) -> None:
        """Delete the results for the given key."""
        self.path_for(key).unlink()

    def get_or_compute(self, key: str, compute_fn: Callable[[], dict]) -> dict:
        """Get the results for a key, computing them on cache miss.

        Args:
            key: The key to get the results for
            compute_fn: A function that computes the results

        Returns:
            The results for the given key
        """
        if self.exists(key):
            logger.debug(f"Cache hit: {self.namespace}/{key}")
            return self.load(key)
        start = time.perf_counter()
        results = compute_fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.debug(f"Cache miss: {self.namespace}/{key} computed in {elapsed_ms:.0f}ms")
        self.save(key, results)
        return results
