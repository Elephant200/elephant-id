"""Tests for the generic named-producer cache."""

from pathlib import Path

import pytest
from loguru import logger

from elephant_id.cache import CacheManager

PHOTO_KEY = "8c47c36d-a75d-4ee4-a58a-3a08fca2c833"
CROP_KEY = f"{PHOTO_KEY}__crop_899_93_1260_512"


def test_read_write_mode_computes_saves_and_reuses_result(tmp_path: Path) -> None:
    """Read-write mode computes one miss and reuses the saved record."""
    cache = CacheManager("sam3-features", cache_root=tmp_path, mode="read_write")
    calls: list[str] = []

    def compute() -> dict[str, object]:
        """Record a computation and return one synthetic producer record."""
        calls.append("computed")
        return {"predictions": [{"class": "elephant", "confidence": 0.75}]}

    first = cache.get_or_compute(PHOTO_KEY, compute)
    second = cache.get_or_compute(PHOTO_KEY, compute)

    assert calls == ["computed"]
    assert first == second == {
        "predictions": [{"class": "elephant", "confidence": 0.75}]
    }
    assert cache.path_for(PHOTO_KEY).exists()


def test_read_only_mode_loads_existing_record_without_computing(
    tmp_path: Path,
) -> None:
    """Read-only mode returns a valid existing record without computation."""
    writer = CacheManager("sam3-features", cache_root=tmp_path, mode="read_write")
    writer.save(PHOTO_KEY, {"v": "cached"})
    cache = CacheManager("sam3-features", cache_root=tmp_path, mode="read_only")

    def fail_if_called() -> dict[str, object]:
        """Fail if a frozen cache attempts computation."""
        raise AssertionError("read-only cache must not compute")

    assert cache.get_or_compute(PHOTO_KEY, fail_if_called) == {"v": "cached"}


def test_read_only_mode_fails_clearly_on_miss_without_computing(
    tmp_path: Path,
) -> None:
    """Read-only mode reports a logical record miss without writing."""
    cache = CacheManager("sam3-features", cache_root=tmp_path, mode="read_only")

    def fail_if_called() -> dict[str, object]:
        """Fail if a frozen cache attempts computation."""
        raise AssertionError("read-only cache must not compute")

    with pytest.raises(
        FileNotFoundError,
        match=f"Read-only cache miss: sam3-features/{PHOTO_KEY}",
    ):
        cache.get_or_compute(PHOTO_KEY, fail_if_called)

    assert not cache.cache_dir.exists()


def test_disabled_mode_bypasses_reads_and_writes(tmp_path: Path) -> None:
    """Disabled mode computes every call without changing stored records."""
    writer = CacheManager("sam3-features", cache_root=tmp_path, mode="read_write")
    writer.save(PHOTO_KEY, {"v": "cached"})
    cache = CacheManager("sam3-features", cache_root=tmp_path, mode="disabled")
    values = iter(({"v": "first"}, {"v": "second"}))

    assert cache.get_or_compute(PHOTO_KEY, lambda: next(values)) == {"v": "first"}
    assert cache.get_or_compute(PHOTO_KEY, lambda: next(values)) == {"v": "second"}
    assert writer.load(PHOTO_KEY) == {"v": "cached"}


def test_mode_controls_explicit_record_operations(tmp_path: Path) -> None:
    """Direct record operations cannot bypass disabled or read-only modes."""
    writer = CacheManager("sam3-features", cache_root=tmp_path, mode="read_write")
    writer.save(PHOTO_KEY, {"v": "cached"})
    disabled = CacheManager("sam3-features", cache_root=tmp_path, mode="disabled")
    read_only = CacheManager("sam3-features", cache_root=tmp_path, mode="read_only")

    assert disabled.exists(PHOTO_KEY) is False
    with pytest.raises(PermissionError, match="Cache reads are disabled"):
        disabled.load(PHOTO_KEY)
    disabled.save(PHOTO_KEY, {"v": "replacement"})
    with pytest.raises(PermissionError, match="Cache is read-only"):
        read_only.save(PHOTO_KEY, {"v": "replacement"})

    assert writer.load(PHOTO_KEY) == {"v": "cached"}


def test_environment_selects_cache_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The global environment setting controls managers without an override."""
    monkeypatch.setenv("ELEPHANT_ID_CACHE_MODE", "disabled")
    cache = CacheManager("sam3-features", cache_root=tmp_path)

    assert cache.get_or_compute(PHOTO_KEY, lambda: {"v": 1}) == {"v": 1}
    assert not cache.cache_dir.exists()


def test_read_write_is_the_default_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cache reads and writes by default when the setting is absent."""
    monkeypatch.delenv("ELEPHANT_ID_CACHE_MODE", raising=False)
    cache = CacheManager("sam3-features", cache_root=tmp_path)

    cache.get_or_compute(PHOTO_KEY, lambda: {"v": 1})

    assert cache.load(PHOTO_KEY) == {"v": 1}


def test_invalid_environment_mode_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid global mode fails during manager construction."""
    monkeypatch.setenv("ELEPHANT_ID_CACHE_MODE", "sometimes")

    with pytest.raises(ValueError, match="Unsupported cache mode: 'sometimes'"):
        CacheManager("sam3-features", cache_root=tmp_path)


def test_read_write_mode_replaces_corrupt_json(tmp_path: Path) -> None:
    """Read-write mode treats invalid JSON as a miss and replaces it."""
    cache = CacheManager("sam3-features", cache_root=tmp_path, mode="read_write")
    cache.path_for(PHOTO_KEY).write_text("not json", encoding="utf-8")

    assert cache.get_or_compute(PHOTO_KEY, lambda: {"v": "recomputed"}) == {
        "v": "recomputed"
    }
    assert cache.load(PHOTO_KEY) == {"v": "recomputed"}


def test_read_write_mode_replaces_non_object_json(tmp_path: Path) -> None:
    """Read-write mode rejects valid JSON that is not a record object."""
    cache = CacheManager("sam3-features", cache_root=tmp_path, mode="read_write")
    cache.path_for(PHOTO_KEY).write_text("[]", encoding="utf-8")

    assert cache.get_or_compute(PHOTO_KEY, lambda: {"v": "recomputed"}) == {
        "v": "recomputed"
    }


def test_read_only_mode_fails_clearly_on_corrupt_json(tmp_path: Path) -> None:
    """Read-only mode reports corruption without computing or replacing."""
    writer = CacheManager("sam3-features", cache_root=tmp_path, mode="read_write")
    writer.path_for(PHOTO_KEY).write_text("not json", encoding="utf-8")
    cache = CacheManager("sam3-features", cache_root=tmp_path, mode="read_only")

    def fail_if_called() -> dict[str, object]:
        """Fail if a frozen cache attempts computation."""
        raise AssertionError("read-only cache must not compute")

    with pytest.raises(
        ValueError,
        match=f"Corrupt read-only cache record: sam3-features/{PHOTO_KEY}",
    ):
        cache.get_or_compute(PHOTO_KEY, fail_if_called)

    assert writer.path_for(PHOTO_KEY).read_text(encoding="utf-8") == "not json"


def test_failed_atomic_save_preserves_existing_record(tmp_path: Path) -> None:
    """A failed replacement preserves the old record and removes its temporary file."""
    cache = CacheManager("sam3-features", cache_root=tmp_path, mode="read_write")
    cache.save(PHOTO_KEY, {"v": "original"})

    with pytest.raises(TypeError):
        cache.save(PHOTO_KEY, {"v": object()})

    assert cache.load(PHOTO_KEY) == {"v": "original"}
    assert list(cache.cache_dir.glob(".*.tmp")) == []


def test_save_replaces_existing_record(tmp_path: Path) -> None:
    """A successful save replaces the complete existing JSON record."""
    cache = CacheManager("sam3-features", cache_root=tmp_path, mode="read_write")
    cache.save(PHOTO_KEY, {"v": "original"})

    cache.save(PHOTO_KEY, {"v": "replacement"})

    assert cache.load(PHOTO_KEY) == {"v": "replacement"}


def test_get_or_compute_logs_logical_miss_then_hit(tmp_path: Path) -> None:
    """Cache logs identify records by producer and caller key."""
    cache = CacheManager("sam3-features", cache_root=tmp_path, mode="read_write")
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        cache.get_or_compute(PHOTO_KEY, lambda: {"v": 1})
        cache.get_or_compute(PHOTO_KEY, lambda: {"v": 1})
    finally:
        logger.remove(sink_id)

    text = "\n".join(messages).lower()
    assert f"cache miss: sam3-features/{PHOTO_KEY}" in text
    assert f"cache hit: sam3-features/{PHOTO_KEY}" in text


def test_path_for_preserves_readable_dependent_inputs(tmp_path: Path) -> None:
    """A UUID and integer crop coordinates remain readable in the file name."""
    cache = CacheManager(
        "yolo26n-keypoints-v1",
        cache_root=tmp_path,
        mode="read_write",
    )

    assert cache.path_for(CROP_KEY) == (
        tmp_path.resolve() / "yolo26n-keypoints-v1" / f"{CROP_KEY}.json"
    )


def test_missing_record_is_absent_and_cannot_be_loaded(tmp_path: Path) -> None:
    """Storage inspection distinguishes a missing record clearly."""
    cache = CacheManager("sam3-features", cache_root=tmp_path, mode="read_write")

    assert cache.exists(PHOTO_KEY) is False
    with pytest.raises(FileNotFoundError):
        cache.load(PHOTO_KEY)


@pytest.mark.parametrize(
    "producer_name",
    ["", ".", "..", "sam3/features", "sam3\\features"],
)
def test_producer_name_must_be_one_safe_path_segment(
    tmp_path: Path,
    producer_name: str,
) -> None:
    """Producer names cannot be empty, nested, or traversal segments."""
    with pytest.raises(ValueError, match="cache producer name"):
        CacheManager(producer_name, cache_root=tmp_path, mode="read_write")


@pytest.mark.parametrize(
    "key",
    ["", ".", "..", "nested/photo-id", "nested\\photo-id"],
)
def test_key_must_be_one_safe_path_segment(tmp_path: Path, key: str) -> None:
    """Caller keys cannot be empty, nested, or traversal segments."""
    cache = CacheManager("sam3-features", cache_root=tmp_path, mode="read_write")

    with pytest.raises(ValueError, match="cache key"):
        cache.path_for(key)


def test_disabled_mode_still_validates_keys(tmp_path: Path) -> None:
    """Disabling storage does not permit malformed caller keys."""
    cache = CacheManager("sam3-features", cache_root=tmp_path, mode="disabled")

    with pytest.raises(ValueError, match="cache key"):
        cache.get_or_compute("../photo-id", lambda: {"v": 1})


def test_producer_symlink_cannot_escape_cache_root(tmp_path: Path) -> None:
    """An existing producer symlink cannot redirect records outside the root."""
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (cache_root / "sam3-features").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="producer name escapes cache root"):
        CacheManager("sam3-features", cache_root=cache_root, mode="read_write")
