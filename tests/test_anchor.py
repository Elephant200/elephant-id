from pathlib import Path

from elephant_id.ai.anchor import AnchorService
from elephant_id.ai.detection import Detection
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


def _detection(xyxy, keypoints) -> Detection:
    return Detection(
        xyxy=xyxy,
        class_name="anchor",
        class_id=0,
        confidence=0.9,
        keypoints=keypoints,
    )


def test_anchor_cache_key_includes_full_crop_coordinates():
    service = _service({"detections": []})

    service.run(_photo(), crop_xyxy=(899.2, 93.8, 1260.1, 512.9))

    assert service.cache_manager.key == "Adam_2011-03-31_03__crop_899_93_1260_512"


def test_anchor_translates_crop_relative_coords_to_absolute():
    detection = _detection((1.0, 2.0, 3.0, 4.0), ((5.0, 6.0), (7.0, 8.0)))
    service = _service({"detections": [detection.to_dict()]})

    translated = service.run(_photo(), crop_xyxy=(100.0, 200.0, 460.0, 600.0))

    result = translated[0]
    assert result.xyxy == (101.0, 202.0, 103.0, 204.0)
    assert result.keypoints == ((105.0, 206.0), (107.0, 208.0))


def test_anchor_translates_every_prediction():
    service = _service(
        {
            "detections": [
                _detection((0, 0, 1, 1), ((0, 0), (1, 1))).to_dict(),
                _detection((10, 10, 11, 11), ((10, 10), (11, 11))).to_dict(),
            ]
        }
    )

    translated = service.run(_photo(), crop_xyxy=(5.0, 7.0, 100.0, 100.0))

    assert translated[0].xyxy[0] == 5.0
    assert translated[1].xyxy[0] == 15.0
    assert translated[1].keypoints[1] == (16.0, 18.0)
