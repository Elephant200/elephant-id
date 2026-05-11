"""
General-purpose cache manager for AI models.
"""

import json
from pathlib import Path
from typing import Callable

class CacheManager:
    """
    Class for caching results of AI models as json
    """
    def __init__(
        self, 
        namespace: str,
        cache_root: Path = Path(".cache"),
    ) -> None:
        """
        Initialize the cache manager for the given namespace.

        Args:
            namespace: The namespace to cache the results in, eg. name of the model.
            cache_root: The root cache directory. Defaults to Path(".cache").
        """
        self.namespace: str = namespace
        self.cache_dir: Path = cache_root.resolve() / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        """
        Return the path for the given key
        """
        return (self.cache_dir / key).with_suffix(".json")

    def exists(self, key: str) -> bool:
        """
        Check if the given key is cached
        """
        return self.path_for(key).exists()

    def load(self, key: str) -> dict:
        """
        Load the results for the given key
        """
        with open(self.path_for(key), "r") as f:
            return json.load(f)
    
    def save(self, key: str, value: dict) -> None:
        """
        Save the results for the given key
        """
        with open(self.path_for(key), "w") as f:
            json.dump(value, f)
    
    def delete(self, key: str) -> None:
        """
        Delete the results for the given key
        """
        self.path_for(key).unlink()
    
    def get_or_compute(self, key: str, compute_fn: Callable[[], dict]) -> dict:
        """
        Get the results for the given key, computing them if they are not cached.

        Args:
            key: The key to get the results for
            compute_fn: A function that computes the results

        Returns:
            The results for the given key
        """
        if self.exists(key):
            return self.load(key)
        results = compute_fn()
        self.save(key, results)
        return results
