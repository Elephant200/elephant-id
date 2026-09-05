"""Uncached YOLO ear-landmark detection."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from elephant_id.domain import Photo
from elephant_id.image import BgrImage
from elephant_id.image.boxes import BoundingBox
from elephant_id.inference.detection import Detection


@dataclass(frozen=True, slots=True)
class YoloLandmarkConfig:
    """Intentional parameters of the settled landmark model."""

    weights: Path
    confidence_threshold: float
    device: str
    output_decimals: int


DEFAULT_CONFIG = YoloLandmarkConfig(
    weights=Path("model_weights/anchor_extraction_yolo26_v2/weights.pt"),
    confidence_threshold=0.25,
    device="mps", # Change on non-macOS devices
    output_decimals=1,
)
PRODUCER_SLUG = "yolo26n-keypoints-v1"


class _LandmarkPredictor(Protocol):
    """Internal crop-relative landmark prediction seam."""

    def predict(self, image: BgrImage) -> tuple[Detection, ...]:
        """Return crop-relative landmark detections."""
        ...


def _detection_from_prediction(prediction: dict[str, Any]) -> Detection:
    """Build a crop-relative Detection from one YOLO prediction."""
    box = prediction["box"]
    keypoints = prediction["keypoints"]
    return Detection(
        xyxy=(
            float(box["x1"]),
            float(box["y1"]),
            float(box["x2"]),
            float(box["y2"]),
        ),
        class_name=str(prediction["name"]).strip(),
        class_id=int(prediction["class"]),
        confidence=float(prediction["confidence"]),
        keypoints=tuple(
            (float(x), float(y))
            for x, y in zip(keypoints["x"], keypoints["y"], strict=True)
        ),
    )


class _YoloPredictor:
    """Lazy local adapter around the settled Ultralytics model."""

    def __init__(self, config: YoloLandmarkConfig) -> None:
        """Load the configured local weights."""
        from ultralytics import YOLO

        self._config = config
        self._model = YOLO(config.weights)

    def predict(self, image: BgrImage) -> tuple[Detection, ...]:
        """Return crop-relative landmark detections."""
        results = self._model.predict(
            source=image,
            device=self._config.device,
            conf=self._config.confidence_threshold,
            verbose=False,
        )
        predictions = json.loads(
            results[0].to_json(decimals=self._config.output_decimals)
        )
        return tuple(
            _detection_from_prediction(prediction) for prediction in predictions
        )


class YoloEarLandmarkDetector:
    """Detect the strongest ear landmark pair with the settled YOLO model."""

    producer_slug = PRODUCER_SLUG
    config = DEFAULT_CONFIG

    def __init__(self, *, predictor: _LandmarkPredictor | None = None) -> None:
        """Configure lazy local-model prediction."""
        self._predictor = predictor

    def _model_predictor(self) -> _LandmarkPredictor:
        """Return the predictor, loading local weights on first use."""
        if self._predictor is None:
            self._predictor = _YoloPredictor(self.config)
        return self._predictor

    def detect(
        self,
        photo: Photo,
        image: BgrImage,
        ear_box: BoundingBox,
    ) -> Detection | None:
        """Return the strongest landmark detection in full-image coordinates."""
        crop = image[ear_box.y1 : ear_box.y2, ear_box.x1 : ear_box.x2].copy()
        relative = self._model_predictor().predict(crop)
        if not relative:
            logger.debug(f"Detected no ear landmarks for photo {photo.photo_id}")
            return None
        selected = max(relative, key=lambda detection: detection.confidence)
        result = selected.translate(float(ear_box.x1), float(ear_box.y1))
        logger.debug(f"Detected ear landmarks for photo {photo.photo_id}")
        return result
