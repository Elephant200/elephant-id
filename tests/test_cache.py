"""Tests for generic cache persistence across named producers."""

from pathlib import Path

import pytest
from loguru import logger

from elephant_id.cache import CacheManager

PHOTO_KEY = "8c47c36d-a75d-4ee4-a58a-3a08fca2c833"
CROP_KEY = f"{PHOTO_KEY}__crop_899_93_1260_512"


def test_get_or_compute_saves_and_reuses_result(tmp_path: Path) -> None:
    """A miss computes once and later calls reuse the saved record."""
    cache = CacheManager(cache_root=tmp_path)
    calls: list[str] = []

    def compute() -> dict[str, object]:
        """Record a computation and return one synthetic producer record."""
        calls.append("computed")
        return {"predictions": [{"class": "elephant", "confidence": 0.75}]}

    first = cache.get_or_compute("sam3-features", PHOTO_KEY, compute)
    second = cache.get_or_compute("sam3-features", PHOTO_KEY, compute)

    assert calls == ["computed"]
    assert first == second == {
        "predictions": [{"class": "elephant", "confidence": 0.75}]
    }
    assert cache.path_for("sam3-features", PHOTO_KEY).exists()


def test_producers_with_the_same_key_do_not_collide(tmp_path: Path) -> None:
    """Producer identity namespaces otherwise identical input keys."""
    cache = CacheManager(cache_root=tmp_path)

    cache.save("sam3-features", PHOTO_KEY, {"stage": "segmentation"})
    cache.save("alpha-profile-v3", PHOTO_KEY, {"stage": "profile"})

    assert cache.load("sam3-features", PHOTO_KEY) == {"stage": "segmentation"}
    assert cache.load("alpha-profile-v3", PHOTO_KEY) == {"stage": "profile"}


def test_get_or_compute_replaces_corrupt_json(tmp_path: Path) -> None:
    """Invalid JSON is treated as a miss and atomically replaced."""
    cache = CacheManager(cache_root=tmp_path)
    cache.path_for("sam3-features", PHOTO_KEY).parent.mkdir(parents=True)
    cache.path_for("sam3-features", PHOTO_KEY).write_text(
        "not json", encoding="utf-8"
    )

    assert cache.get_or_compute(
        "sam3-features", PHOTO_KEY, lambda: {"v": "recomputed"}
    ) == {"v": "recomputed"}
    assert cache.load("sam3-features", PHOTO_KEY) == {"v": "recomputed"}


def test_get_or_compute_replaces_non_object_json(tmp_path: Path) -> None:
    """Valid JSON that is not a record object is replaced."""
    cache = CacheManager(cache_root=tmp_path)
    cache.path_for("sam3-features", PHOTO_KEY).parent.mkdir(parents=True)
    cache.path_for("sam3-features", PHOTO_KEY).write_text("[]", encoding="utf-8")

    assert cache.get_or_compute(
        "sam3-features", PHOTO_KEY, lambda: {"v": "recomputed"}
    ) == {"v": "recomputed"}


def test_failed_atomic_save_preserves_existing_record(tmp_path: Path) -> None:
    """A failed replacement preserves the old record and removes its temporary file."""
    cache = CacheManager(cache_root=tmp_path)
    cache.save("sam3-features", PHOTO_KEY, {"v": "original"})

    with pytest.raises(TypeError):
        cache.save("sam3-features", PHOTO_KEY, {"v": object()})

    assert cache.load("sam3-features", PHOTO_KEY) == {"v": "original"}
    producer_dir = cache.path_for("sam3-features", PHOTO_KEY).parent
    assert list(producer_dir.glob(".*.tmp")) == []


def test_save_replaces_existing_record(tmp_path: Path) -> None:
    """A successful save replaces the complete existing JSON record."""
    cache = CacheManager(cache_root=tmp_path)
    cache.save("sam3-features", PHOTO_KEY, {"v": "original"})

    cache.save("sam3-features", PHOTO_KEY, {"v": "replacement"})

    assert cache.load("sam3-features", PHOTO_KEY) == {"v": "replacement"}


def test_get_or_compute_logs_logical_miss_then_hit(tmp_path: Path) -> None:
    """Cache logs identify records by producer and caller key."""
    cache = CacheManager(cache_root=tmp_path)
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        cache.get_or_compute("sam3-features", PHOTO_KEY, lambda: {"v": 1})
        cache.get_or_compute("sam3-features", PHOTO_KEY, lambda: {"v": 1})
    finally:
        logger.remove(sink_id)

    text = "\n".join(messages).lower()
    assert f"cache miss: sam3-features/{PHOTO_KEY}" in text
    assert f"cache hit: sam3-features/{PHOTO_KEY}" in text


def test_path_for_preserves_readable_dependent_inputs(tmp_path: Path) -> None:
    """Producer and integer crop coordinates remain readable in the path."""
    cache = CacheManager(cache_root=tmp_path)

    assert cache.path_for("yolo26n-keypoints-v1", CROP_KEY) == (
        tmp_path.resolve() / "yolo26n-keypoints-v1" / f"{CROP_KEY}.json"
    )


def test_missing_record_is_absent_and_cannot_be_loaded(tmp_path: Path) -> None:
    """Storage inspection distinguishes a missing record clearly."""
    cache = CacheManager(cache_root=tmp_path)

    assert cache.exists("sam3-features", PHOTO_KEY) is False
    with pytest.raises(FileNotFoundError):
        cache.load("sam3-features", PHOTO_KEY)


@pytest.mark.parametrize(
    "producer_id",
    ["", ".", "..", "sam3/features", "sam3\\features"],
)
def test_producer_id_must_be_one_safe_path_segment(
    tmp_path: Path,
    producer_id: str,
) -> None:
    """Producer identities cannot be empty, nested, or traversal segments."""
    cache = CacheManager(cache_root=tmp_path)

    with pytest.raises(ValueError, match="cache producer ID"):
        cache.path_for(producer_id, PHOTO_KEY)


@pytest.mark.parametrize(
    "key",
    ["", ".", "..", "nested/photo-id", "nested\\photo-id"],
)
def test_key_must_be_one_safe_path_segment(tmp_path: Path, key: str) -> None:
    """Caller keys cannot be empty, nested, or traversal segments."""
    cache = CacheManager(cache_root=tmp_path)

    with pytest.raises(ValueError, match="cache key"):
        cache.path_for("sam3-features", key)


def test_producer_symlink_cannot_escape_cache_root(tmp_path: Path) -> None:
    """An existing producer symlink cannot redirect records outside the root."""
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (cache_root / "sam3-features").symlink_to(outside, target_is_directory=True)
    cache = CacheManager(cache_root=cache_root)

    with pytest.raises(ValueError, match="producer ID escapes cache root"):
        cache.path_for("sam3-features", PHOTO_KEY)
