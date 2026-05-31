import pytest

from elephant_id.cache import CacheManager


def test_get_or_compute_writes_json_and_reuses_cached_result(tmp_path):
    cache = CacheManager("sam3/body", cache_root=tmp_path)
    key = "Adam_2011-03-31_02__conf-0.50__nms-True__iou-0"
    calls = []

    def compute():
        calls.append("computed")
        return {"predictions": [{"class": "elephant", "confidence": 0.75}]}

    first = cache.get_or_compute(key, compute)
    second = cache.get_or_compute(key, compute)

    assert calls == ["computed"]
    assert first == second == {
        "predictions": [{"class": "elephant", "confidence": 0.75}]
    }
    assert cache.path_for(key).exists()


def test_path_for_preserves_decimal_cache_key_segments(tmp_path):
    cache = CacheManager("sam3/body", cache_root=tmp_path)
    key = "Adam_2011-03-31_02__conf-0.50__nms-True__iou-0.20"

    assert cache.path_for(key) == (
        tmp_path.resolve()
        / "sam3/body/Adam_2011-03-31_02__conf-0.50__nms-True__iou-0.20.json"
    )


def test_cache_delete_removes_saved_result(tmp_path):
    cache = CacheManager("anchor", cache_root=tmp_path)
    key = "Adam_2011-03-31_03__crop_899_93"

    cache.save(key, {"predictions": {"class": "anchor"}})
    cache.delete(key)

    assert not cache.exists(key)


def test_exists_is_false_for_unknown_key(tmp_path):
    cache = CacheManager("sam3/body", cache_root=tmp_path)

    assert cache.exists("never_saved") is False


def test_load_missing_key_raises(tmp_path):
    cache = CacheManager("sam3/body", cache_root=tmp_path)

    with pytest.raises(FileNotFoundError):
        cache.load("never_saved")


def test_save_overwrites_existing_value(tmp_path):
    cache = CacheManager("anchor", cache_root=tmp_path)
    key = "Adam_2011-03-31_03"

    cache.save(key, {"v": 1})
    cache.save(key, {"v": 2})

    assert cache.load(key) == {"v": 2}


def test_init_creates_namespace_directory(tmp_path):
    cache = CacheManager("nested/space", cache_root=tmp_path)

    assert cache.cache_dir.is_dir()
    assert cache.cache_dir == tmp_path.resolve() / "nested/space"


def test_init_rejects_namespace_escaping_cache_root(tmp_path):
    with pytest.raises(ValueError, match="namespace escapes"):
        CacheManager("../evil", cache_root=tmp_path)


def test_path_for_rejects_key_escaping_namespace(tmp_path):
    cache = CacheManager("sam3/body", cache_root=tmp_path)

    with pytest.raises(ValueError, match="key escapes"):
        cache.path_for("../../escape")
