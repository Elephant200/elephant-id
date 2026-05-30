import pytest

from elephant_id.ai.sam3 import Sam3Service, prediction_center_to_xyxy


def test_prediction_center_to_xyxy_preserves_non_bbox_fields():
    prediction = {
        "class": "elephant",
        "class_id": 0,
        "confidence": 0.75,
        "x": 1072.5,
        "y": 1319.5,
        "width": 315.0,
        "height": 357.0,
    }

    converted = prediction_center_to_xyxy(prediction)

    assert converted == {
        "class": "elephant",
        "class_id": 0,
        "confidence": 0.75,
        "x1": 915.0,
        "y1": 1141.0,
        "x2": 1230.0,
        "y2": 1498.0,
    }
    assert prediction["x"] == 1072.5


class _RecordingDataset:
    def __init__(self):
        self.read_photos = []

    def read_image(self, photo):
        self.read_photos.append(photo)
        return object()


class _RecordingRunner:
    confidence_threshold = 0.5
    nms = True
    nms_iou_threshold = 0.2

    def __init__(self):
        self.calls = []

    def run(self, image, query_preset):
        self.calls.append((image, query_preset))
        return {"predictions": []}


class _RecordingCache:
    def __init__(self):
        self.key = None

    def get_or_compute(self, key, compute_fn):
        self.key = key
        return compute_fn()


def test_sam3_service_uses_preset_cache_and_reads_photo(make_photo):
    service = Sam3Service.__new__(Sam3Service)
    service.runner = _RecordingRunner()
    service.dataset = _RecordingDataset()
    cache = _RecordingCache()
    service.cache_managers = {"body": cache}
    photo = make_photo(name="Adam", sighting_date="2011-03-31", sequence=2)

    result = service.run(photo, "body")

    assert result == {"predictions": []}
    assert cache.key == "Adam_2011-03-31_02__conf-0.50__nms-True__iou-0.20"
    assert service.dataset.read_photos == [photo]
    assert len(service.runner.calls) == 1
    assert service.runner.calls[0][1] == "body"


def test_sam3_service_rejects_unknown_query_preset(make_photo):
    service = Sam3Service.__new__(Sam3Service)
    service.runner = _RecordingRunner()
    service.dataset = _RecordingDataset()
    service.cache_managers = {}

    with pytest.raises(ValueError, match="Unknown SAM3 query preset"):
        service.run(make_photo(), "unknown")
