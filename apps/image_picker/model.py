"""Cache-first model helpers for side-specific ear candidates."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from elephant_id.ai.detection import Detection
from elephant_id.cache import CacheManager
from elephant_id.coding.ears.anchored_ear import AnchoredEar
from elephant_id.constants import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
    DEFAULT_SAM3_NMS,
    DEFAULT_SAM3_NMS_IOU_THRESHOLD,
    MIN_FEATURE_BODY_OVERLAP,
    MIN_MULTIPLE_BODY_AREA_RATIO,
    MIN_MULTIPLE_EAR_AREA_RATIO,
    ROBOFLOW_WORKSPACE,
    SAM3_QUERY_PRESETS,
)
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.image.transforms import apply_crop

from .config import (
    MAX_EAR_BOX_HEIGHT_WIDTH,
    MIN_EAR_BOX_AREA,
    MIN_EAR_BOX_HEIGHT_WIDTH,
    SIDES,
)


class PickerModelUnavailableError(RuntimeError):
    """Raised when a missing model result cannot be computed."""


@dataclass(frozen=True)
class EarCandidate:
    """One accepted side-specific ear crop candidate."""

    candidate_id: str
    side: str
    photo_identifier: str
    identity: str
    date: str
    image_path: str
    seek_code: str
    crop_xyxy: tuple[float, float, float, float]
    confidence: float
    box_height_width: float
    box_area: float

    def to_json(self, selected: bool = False) -> dict:
        """Return the JSON representation consumed by the frontend."""
        return {
            "candidateId": self.candidate_id,
            "side": self.side,
            "photoIdentifier": self.photo_identifier,
            "identity": self.identity,
            "date": self.date,
            "imagePath": self.image_path,
            "seekCode": self.seek_code,
            "cropXyxy": list(self.crop_xyxy),
            "confidence": self.confidence,
            "boxHeightWidth": self.box_height_width,
            "boxArea": self.box_area,
            "selected": selected,
        }


class CacheFirstSam3:
    """Run SAM3 with cache reads before optional remote computation."""

    def __init__(self, dataset: Dataset, cache_root: Path = Path(DEFAULT_CACHE_ROOT)) -> None:
        """Create per-preset cache managers without initializing the runner."""
        self.dataset = dataset
        self.cache_managers = {
            preset: CacheManager(f"sam3/{preset}", cache_root=cache_root)
            for preset in SAM3_QUERY_PRESETS
        }
        self._runner = None
        self._runner_lock = threading.Lock()

    def run(self, photo: Photo, query_preset: str) -> list[Detection]:
        """Return cached detections or compute them on cache miss."""
        if query_preset not in SAM3_QUERY_PRESETS:
            raise ValueError(f"Unknown SAM3 query preset: {query_preset}")
        key = _sam3_cache_key(photo)
        cache = self.cache_managers[query_preset]
        if cache.exists(key):
            envelope = cache.load(key)
        else:
            envelope = self._compute(photo, query_preset)
            cache.save(key, envelope)
        detections = [Detection.from_dict(detection) for detection in envelope["detections"]]
        logger.info(f"Picker SAM3 {query_preset} for {photo.identifier}: {len(detections)} detections")
        return detections

    def _compute(self, photo: Photo, query_preset: str) -> dict:
        """Run SAM3 remotely for one uncached photo/preset."""
        runner = self._get_runner()
        detections = runner.run(self.dataset.read_image(photo), query_preset)
        return {
            "queries": list(SAM3_QUERY_PRESETS[query_preset]),
            "confidence_threshold": DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
            "nms": DEFAULT_SAM3_NMS,
            "nms_iou_threshold": DEFAULT_SAM3_NMS_IOU_THRESHOLD,
            "detections": [detection.to_dict() for detection in detections],
        }

    def _get_runner(self):
        """Construct the SAM3 runner only when a cache miss needs it."""
        with self._runner_lock:
            if self._runner is not None:
                return self._runner
            _load_dotenv_if_available()
            if not os.environ.get("ROBOFLOW_API_KEY", "").strip():
                raise PickerModelUnavailableError(
                    "SAM3 cache miss and ROBOFLOW_API_KEY is not set."
                )
            try:
                from elephant_id.ai.sam3 import Sam3Runner
            except Exception as error:
                raise PickerModelUnavailableError(f"SAM3 is unavailable: {error}") from error
            self._runner = Sam3Runner(
                confidence_threshold=DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
                nms=DEFAULT_SAM3_NMS,
                nms_iou_threshold=DEFAULT_SAM3_NMS_IOU_THRESHOLD,
                workspace_name=ROBOFLOW_WORKSPACE,
            )
            return self._runner


class CacheFirstAnchor:
    """Run the anchor model with cache reads before local computation."""

    def __init__(self, dataset: Dataset, cache_root: Path = Path(DEFAULT_CACHE_ROOT)) -> None:
        """Create the cache manager without loading YOLO weights."""
        self.dataset = dataset
        self.cache_manager = CacheManager("anchor", cache_root=cache_root)
        self._runner = None
        self._runner_lock = threading.Lock()

    def run(self, photo: Photo, crop_xyxy: tuple[float, float, float, float]) -> list[Detection]:
        """Return cached anchor detections or compute them on cache miss."""
        ox1, oy1, ox2, oy2 = (int(coord) for coord in crop_xyxy)
        key = f"{photo.identifier}__crop_{ox1}_{oy1}_{ox2}_{oy2}"
        if self.cache_manager.exists(key):
            envelope = self.cache_manager.load(key)
        else:
            envelope = self._compute(photo, crop_xyxy)
            self.cache_manager.save(key, envelope)
        detections = [
            Detection.from_dict(detection).translate(float(ox1), float(oy1))
            for detection in envelope["detections"]
        ]
        logger.info(f"Picker Anchor for {photo.identifier}: {len(detections)} detections")
        return detections

    def _compute(self, photo: Photo, crop_xyxy: tuple[float, float, float, float]) -> dict:
        """Run the local anchor model for one uncached ear crop."""
        runner = self._get_runner()
        detections = runner.run(apply_crop(self.dataset.read_image(photo), crop_xyxy))
        return {"detections": [detection.to_dict() for detection in detections]}

    def _get_runner(self):
        """Construct the anchor runner only when a cache miss needs it."""
        with self._runner_lock:
            if self._runner is not None:
                return self._runner
            try:
                from elephant_id.ai.anchor import AnchorRunner
            except Exception as error:
                raise PickerModelUnavailableError(f"Anchor model is unavailable: {error}") from error
            self._runner = AnchorRunner()
            return self._runner


class CandidateAnalyzer:
    """Find accepted side-specific ear candidates for photos."""

    def __init__(self, dataset: Dataset, cache_root: Path = Path(DEFAULT_CACHE_ROOT)) -> None:
        """Create cache-first model services for the picker."""
        self.dataset = dataset
        self.sam3 = CacheFirstSam3(dataset=dataset, cache_root=cache_root)
        self.anchor = CacheFirstAnchor(dataset=dataset, cache_root=cache_root)

    def candidates_for_photo(
        self,
        photo: Photo,
        *,
        side: str,
        identity: str,
        date: str,
        image_path: str,
        seek_code: str,
    ) -> list[EarCandidate]:
        """Return accepted crop candidates for one photo and side."""
        if side not in SIDES:
            raise ValueError(f"Invalid side: {side}")

        body_detections = self.sam3.run(photo, "body")
        feature_detections = self.sam3.run(photo, "features")
        body = _choose_body(body_detections)
        if body is None:
            return []

        features_on_body = _features_on_body(body, feature_detections)
        ears = _choose_usable_ears([
            feature for feature in features_on_body if feature.class_name == "ear"
        ])

        accepted: list[EarCandidate] = []
        for ear in ears:
            if not _raw_ear_box_passes(ear):
                continue
            anchors = self.anchor.run(photo, ear.xyxy)
            if not anchors:
                continue
            anchor = max(anchors, key=lambda detection: detection.confidence)
            anchored = AnchoredEar(ear, anchor)
            if anchored.side != side:
                continue
            x1, y1, x2, y2 = ear.xyxy
            width = x2 - x1
            height = y2 - y1
            accepted.append(
                EarCandidate(
                    candidate_id=f"{photo.identifier}__{side}__{len(accepted)}",
                    side=side,
                    photo_identifier=photo.identifier,
                    identity=identity,
                    date=date,
                    image_path=image_path,
                    seek_code=seek_code,
                    crop_xyxy=tuple(float(value) for value in ear.xyxy),
                    confidence=ear.confidence,
                    box_height_width=height / width,
                    box_area=width * height,
                )
            )
        return accepted


def _sam3_cache_key(photo: Photo) -> str:
    """Return the cache key used by ``Sam3Service``."""
    return (
        f"{photo.identifier}__"
        f"conf-{DEFAULT_SAM3_CONFIDENCE_THRESHOLD:.2f}__"
        f"nms-{DEFAULT_SAM3_NMS}__"
        f"iou-{DEFAULT_SAM3_NMS_IOU_THRESHOLD:.2f}"
    )


def _raw_ear_box_passes(ear: Detection) -> bool:
    """Return whether a SAM3 ear box passes the picker geometry gate."""
    x1, y1, x2, y2 = ear.xyxy
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        return False
    ratio = height / width
    return (
        width * height >= MIN_EAR_BOX_AREA
        and MIN_EAR_BOX_HEIGHT_WIDTH <= ratio <= MAX_EAR_BOX_HEIGHT_WIDTH
    )


def _choose_body(body_detections: list[Detection]) -> Detection | None:
    """Choose one body detection using the production analyzer rule."""
    if len(body_detections) == 1:
        return body_detections[0]
    if len(body_detections) == 0:
        return None
    bodies_by_area = sorted(body_detections, key=lambda detection: detection.area(), reverse=True)
    if len(bodies_by_area) == 1:
        return bodies_by_area[0]
    if bodies_by_area[0].area() / bodies_by_area[1].area() > MIN_MULTIPLE_BODY_AREA_RATIO:
        return bodies_by_area[0]
    return None


def _features_on_body(body: Detection, feature_detections: list[Detection]) -> list[Detection]:
    """Keep feature detections mostly inside the selected body."""
    output = []
    for feature in feature_detections:
        feature_area = feature.area()
        if feature_area == 0.0:
            continue
        if feature.intersection_area(body) / feature_area > MIN_FEATURE_BODY_OVERLAP:
            output.append(feature)
    return output


def _choose_usable_ears(ears: list[Detection]) -> list[Detection]:
    """Mirror the production analyzer's multiple-ear filtering."""
    if len(ears) > 2:
        ears = sorted(ears, key=lambda detection: detection.area(), reverse=True)[:2]
    if len(ears) != 2:
        return ears
    first_area = ears[0].area()
    second_area = ears[1].area()
    if second_area == 0.0:
        return [ears[0]]
    if first_area == 0.0:
        return [ears[1]]
    if first_area / second_area > MIN_MULTIPLE_EAR_AREA_RATIO:
        return [ears[0]]
    if second_area / first_area > MIN_MULTIPLE_EAR_AREA_RATIO:
        return [ears[1]]
    return ears


def _load_dotenv_if_available() -> None:
    """Load ``.env`` when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
