from pathlib import Path

from elephant_id.ai.anchor import AnchorService
from elephant_id.domain import Photo


class _RecordingCache:
    def __init__(self, result):
        self.key = None
        self._result = result

    def get_or_compute(self, key, compute_fn):
        self.key = key
        return self._result


def _photo() -> Photo:
    return Photo(
        identifier="Adam_2011-03-31_03",
        image_path=Path("Adam/2011-03-31/Adam_2011-03-31_03.jpg"),
        elephant_name="Adam",
        sighting_id="Adam_2011-03-31",
    )


def _service(result: dict) -> AnchorService:
    service = AnchorService.__new__(AnchorService)
    service.cache_manager = _RecordingCache(result)
    service.dataset = object()
    service.runner = object()
    return service


def test_anchor_cache_key_includes_full_crop_coordinates():
    service = _service({"predictions": []})

    service.run(_photo(), crop_xyxy=(899.2, 93.8, 1260.1, 512.9))

    assert service.cache_manager.key == "Adam_2011-03-31_03__crop_899_93_1260_512"


def test_anchor_translates_crop_relative_coords_to_absolute():
    result = {
        "predictions": [
            {
                "x1": 1.0,
                "y1": 2.0,
                "x2": 3.0,
                "y2": 4.0,
                "keypoints": [[5.0, 6.0], [7.0, 8.0]],
            }
        ]
    }
    service = _service(result)

    translated = service.run(_photo(), crop_xyxy=(100.0, 200.0, 460.0, 600.0))

    pred = translated["predictions"][0]
    assert (pred["x1"], pred["y1"]) == (101.0, 202.0)
    assert (pred["x2"], pred["y2"]) == (103.0, 204.0)
    assert pred["keypoints"] == [[105.0, 206.0], [107.0, 208.0]]


def test_anchor_translates_every_prediction():
    result = {
        "predictions": [
            {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "keypoints": [[0, 0], [1, 1]]},
            {
                "x1": 10,
                "y1": 10,
                "x2": 11,
                "y2": 11,
                "keypoints": [[10, 10], [11, 11]],
            },
        ]
    }
    service = _service(result)

    translated = service.run(_photo(), crop_xyxy=(5.0, 7.0, 100.0, 100.0))

    assert translated["predictions"][0]["x1"] == 5.0
    assert translated["predictions"][1]["x1"] == 15.0
    assert translated["predictions"][1]["keypoints"][1] == [16.0, 18.0]
