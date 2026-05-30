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


def test_cache_delete_removes_saved_result(tmp_path):
    cache = CacheManager("anchor", cache_root=tmp_path)
    key = "Adam_2011-03-31_03__crop_899_93"

    cache.save(key, {"predictions": {"class": "anchor"}})
    cache.delete(key)

    assert not cache.exists(key)
