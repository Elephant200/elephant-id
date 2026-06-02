import json
from pathlib import Path

import pytest

from elephant_id.ai.detection import Detection
from elephant_id.ai.sam3 import Sam3Runner, Sam3Service, detection_from_prediction
from elephant_id.constants import SAM3_QUERY_PRESETS

_SAMPLE_RESPONSE = Path(__file__).resolve().parents[1] / "docs" / "sam3_sample_response.json"


def test_detection_from_prediction_converts_center_box_and_strips_class():
    detection = detection_from_prediction(
        {
            "class": " ear",  # leading whitespace from SAM3
            "class_id": 2,
            "confidence": 0.75,
            "x": 100.0,
            "y": 200.0,
            "width": 20.0,
            "height": 40.0,
            "rle_mask": {"size": [4, 4], "counts": "abc"},
        }
    )

    assert detection.xyxy == (90.0, 180.0, 110.0, 220.0)
    assert detection.class_name == "ear"
    assert detection.class_id == 2
    assert detection.confidence == 0.75
    assert detection.rle_mask == {"size": [4, 4], "counts": "abc"}
    assert detection.keypoints == ()


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
        return []


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

    assert result == []
    assert cache.key == "Adam_2011-03-31_02__conf-0.50__nms-True__iou-0.20"
    assert service.dataset.read_photos == [photo]
    assert len(service.runner.calls) == 1
    assert service.runner.calls[0][1] == "body"


def test_sam3_service_returns_detections_from_cached_envelope(make_photo):
    detection = Detection(
        xyxy=(1.0, 2.0, 3.0, 4.0), class_name="ear", class_id=2, confidence=0.8
    )

    class _CannedCache:
        def get_or_compute(self, key, compute_fn):
            return {"detections": [detection.to_dict()]}

    service = Sam3Service.__new__(Sam3Service)
    service.runner = _RecordingRunner()
    service.dataset = _RecordingDataset()
    service.cache_managers = {"features": _CannedCache()}

    result = service.run(make_photo(), "features")

    assert result == [detection]


def test_sam3_service_compute_builds_envelope(make_photo):
    service = Sam3Service.__new__(Sam3Service)
    service.runner = _RecordingRunner()
    service.dataset = _RecordingDataset()

    envelope = service._compute(make_photo(), "features")

    assert envelope["queries"] == list(SAM3_QUERY_PRESETS["features"])
    assert envelope["confidence_threshold"] == 0.5
    assert envelope["nms"] is True
    assert envelope["nms_iou_threshold"] == 0.2
    assert envelope["detections"] == []


def test_sam3_service_rejects_unknown_query_preset(make_photo):
    service = Sam3Service.__new__(Sam3Service)
    service.runner = _RecordingRunner()
    service.dataset = _RecordingDataset()
    service.cache_managers = {}

    with pytest.raises(ValueError, match="Unknown SAM3 query preset"):
        service.run(make_photo(), "unknown")


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def run_workflow(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


def _runner(response) -> Sam3Runner:
    runner = Sam3Runner.__new__(Sam3Runner)
    runner.client = _FakeClient(response)
    runner.workspace_name = "ws"
    runner.workflow_id = "wf"
    runner.confidence_threshold = 0.5
    runner.nms = True
    runner.nms_iou_threshold = 0.2
    return runner


def test_sam3_runner_parses_sample_response_into_detections():
    response = json.loads(_SAMPLE_RESPONSE.read_text())
    runner = _runner(response)

    detections = runner.run(image=object(), query_preset="features")

    assert len(detections) == 3
    # First detection: center (x=1194.5, y=686), size (w=345, h=392).
    first = detections[0]
    assert first.xyxy == (
        1194.5 - 345 / 2,
        686 - 392 / 2,
        1194.5 + 345 / 2,
        686 + 392 / 2,
    )
    assert first.class_name == "trunk"
    assert first.rle_mask["size"] == [1080, 1920]
    # Class names are normalized (the sample emits " ear" with a leading space).
    assert detections[1].class_name == "ear"


def test_sam3_runner_passes_preset_queries_and_params_to_client():
    response = json.loads(_SAMPLE_RESPONSE.read_text())
    runner = _runner(response)

    runner.run(image=object(), query_preset="features")

    call = runner.client.calls[0]
    assert call["workspace_name"] == "ws"
    assert call["workflow_id"] == "wf"
    assert call["parameters"]["queries"] == "elephant trunk,tusk,ear,tail"
    assert call["parameters"]["confidence_threshold"] == 0.5
    assert call["parameters"]["nms"] is True
    assert call["parameters"]["nms_iou_threshold"] == 0.2


def test_sam3_runner_rejects_empty_response():
    runner = _runner([{"predictions": {}}])

    with pytest.raises(ValueError, match="Unexpected response from SAM3"):
        runner.run(image=object(), query_preset="body")


def test_sam3_runner_rejects_unknown_query_preset():
    runner = _runner([{"predictions": {"predictions": []}}])

    with pytest.raises(ValueError, match="Unknown SAM3 query preset"):
        runner.run(image=object(), query_preset="nope")
