import json
from pathlib import Path

import pytest

from elephant_id.ai.sam3 import Sam3Runner, Sam3Service, prediction_center_to_xyxy
from elephant_id.constants import SAM3_QUERY_PRESETS

_SAMPLE_RESPONSE = Path(__file__).resolve().parents[1] / "docs" / "sam3_sample_response.json"


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


def test_sam3_runner_parses_sample_response_and_converts_bbox():
    response = json.loads(_SAMPLE_RESPONSE.read_text())
    runner = _runner(response)

    result = runner.run(image=object(), query_preset="features")

    assert result["queries"] == list(SAM3_QUERY_PRESETS["features"])
    assert result["confidence_threshold"] == 0.5
    assert result["nms"] is True
    assert result["nms_iou_threshold"] == 0.2

    preds = result["predictions"]
    assert len(preds) == 3

    # First detection: center (x=1194.5, y=686), size (w=345, h=392).
    first = preds[0]
    assert first["x1"] == 1194.5 - 345 / 2
    assert first["y1"] == 686 - 392 / 2
    assert first["x2"] == 1194.5 + 345 / 2
    assert first["y2"] == 686 + 392 / 2
    # Center keys are dropped; non-bbox fields are preserved.
    assert "x" not in first and "width" not in first
    assert first["class"] == "trunk"
    assert first["rle_mask"]["size"] == [1080, 1920]


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
