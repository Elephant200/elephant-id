"""Behavior tests for semantic inference processors and persistence."""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from elephant_id.cache import CacheManager
from elephant_id.domain import Photo
from elephant_id.image.boxes import BoundingBox
from elephant_id.inference import Detection
from elephant_id.inference.landmarks import (
    CachedEarLandmarkDetector,
    YoloEarLandmarkDetector,
)
from elephant_id.inference.segmentation.sam3 import (
    CachedSam3FeatureSegmenter,
    Sam3EarSegmenter,
    Sam3FeatureSegmenter,
)

PHOTO = Photo(
    photo_id=UUID("8c47c36d-a75d-4ee4-a58a-3a08fca2c833"),
    sighting_id=UUID("c44ac5bd-eb07-493d-b296-f69f3844e463"),
)
IMAGE = np.zeros((20, 30, 3), dtype=np.uint8)


def _detection(
    *,
    class_name: str,
    xyxy: tuple[float, float, float, float],
    confidence: float = 0.8,
    rle_mask: dict[str, object] | None = None,
    keypoints: tuple[tuple[float, float], ...] | None = None,
) -> Detection:
    """Build one synthetic inference detection."""
    return Detection(
        xyxy=xyxy,
        class_name=class_name,
        class_id=0,
        confidence=confidence,
        rle_mask=rle_mask,
        keypoints=keypoints,
    )


@dataclass
class _FeatureSegmenter:
    """Record complete SAM3 feature computations."""

    detections: tuple[Detection, ...]
    producer_slug: str = "sam3-features"
    calls: list[Photo] = field(default_factory=list)

    def segment_features(
        self,
        photo: Photo,
        image: np.ndarray,
    ) -> tuple[Detection, ...]:
        """Record and return every configured feature."""
        self.calls.append(photo)
        return self.detections


def test_sam3_cache_preserves_complete_features_before_ear_adaptation(
    tmp_path: Path,
) -> None:
    """SAM3 misses persist every feature and ears are filtered afterward."""
    ear = _detection(class_name="ear", xyxy=(2.0, 3.0, 12.0, 15.0))
    tail = _detection(class_name="tail", xyxy=(15.0, 5.0, 20.0, 10.0))
    inner = _FeatureSegmenter((tail, ear))
    cache = CacheManager(cache_root=tmp_path)
    cached = CachedSam3FeatureSegmenter(inner, cache)
    segmenter = Sam3EarSegmenter(cached)

    assert segmenter.segment(PHOTO, IMAGE) == (ear,)
    assert segmenter.segment(PHOTO, IMAGE) == (ear,)
    assert inner.calls == [PHOTO]
    assert cache.load(inner.producer_slug, str(PHOTO.photo_id))["detections"] == [
        tail.to_dict(),
        ear.to_dict(),
    ]
    assert segmenter.producer_slug == inner.producer_slug


def test_sam3_cache_reads_existing_complete_multi_feature_record(
    tmp_path: Path,
) -> None:
    """Historical complete records remain valid without model computation."""
    ear = _detection(class_name="ear", xyxy=(2.0, 3.0, 12.0, 15.0))
    tusk = _detection(class_name="tusk", xyxy=(15.0, 5.0, 20.0, 10.0))
    cache = CacheManager(cache_root=tmp_path)
    cache.save(
        "sam3-features",
        str(PHOTO.photo_id),
        {"detections": [tusk.to_dict(), ear.to_dict()]},
    )
    inner = _FeatureSegmenter(())
    cached = CachedSam3FeatureSegmenter(inner, cache)

    assert cached.segment_features(PHOTO, IMAGE) == (tusk, ear)
    assert inner.calls == []


@dataclass
class _LandmarkDetector:
    """Record full-image landmark detections."""

    detection: Detection | None
    producer_slug: str = "yolo26n-keypoints-v1"
    calls: list[BoundingBox] = field(default_factory=list)

    def detect(
        self,
        photo: Photo,
        image: np.ndarray,
        ear_box: BoundingBox,
    ) -> Detection | None:
        """Record and return one optional full-image detection."""
        self.calls.append(ear_box)
        return self.detection


def test_landmark_cache_stores_full_image_coordinates(tmp_path: Path) -> None:
    """Landmark cache payloads use the public full-image coordinate system."""
    box = BoundingBox(10, 6, 23, 20)
    detection = _detection(
        class_name="ear",
        xyxy=(11.0, 8.0, 18.0, 17.0),
        keypoints=((12.0, 9.0), (16.0, 15.0)),
    )
    inner = _LandmarkDetector(detection)
    cache = CacheManager(cache_root=tmp_path)
    detector = CachedEarLandmarkDetector(inner, cache)

    assert detector.detect(PHOTO, IMAGE, box) == detection
    assert detector.detect(PHOTO, IMAGE, box) == detection
    assert inner.calls == [box]

    key = f"{PHOTO.photo_id}__crop_10_6_23_20"
    stored = cache.load(inner.producer_slug, key)
    assert Detection.from_dict(stored["detection"]) == detection
    assert detector.producer_slug == inner.producer_slug


def test_landmark_cache_preserves_ordinary_zero_detection(tmp_path: Path) -> None:
    """A cached null result remains an ordinary non-error outcome."""
    box = BoundingBox(10, 6, 23, 20)
    inner = _LandmarkDetector(None)
    detector = CachedEarLandmarkDetector(
        inner,
        CacheManager(cache_root=tmp_path),
    )

    assert detector.detect(PHOTO, IMAGE, box) is None
    assert detector.detect(PHOTO, IMAGE, box) is None
    assert inner.calls == [box]


def test_landmark_cache_rejects_obsolete_multi_detection_schema(
    tmp_path: Path,
) -> None:
    """A stale crop-relative record cannot masquerade as a null detection."""
    box = BoundingBox(10, 6, 23, 20)
    cache = CacheManager(cache_root=tmp_path)
    key = f"{PHOTO.photo_id}__crop_10_6_23_20"
    cache.save("yolo26n-keypoints-v1", key, {"detections": []})
    detector = CachedEarLandmarkDetector(_LandmarkDetector(None), cache)

    with pytest.raises(ValueError, match="must contain detection"):
        detector.detect(PHOTO, IMAGE, box)


def test_yolo_selects_strongest_detection_and_returns_full_image_geometry() -> None:
    """YOLO hides extra candidates and translates the winner once."""
    weaker = _detection(
        class_name="ear",
        xyxy=(1.0, 2.0, 8.0, 11.0),
        confidence=0.2,
        keypoints=((2.0, 3.0), (6.0, 9.0)),
    )
    stronger = _detection(
        class_name="ear",
        xyxy=(2.0, 3.0, 9.0, 12.0),
        confidence=0.9,
        keypoints=((3.0, 4.0), (7.0, 10.0)),
    )

    class _Predictor:
        """Return two crop-relative detections."""

        def __init__(self) -> None:
            self.images: list[np.ndarray] = []

        def predict(self, image: np.ndarray) -> tuple[Detection, ...]:
            """Record the crop and return both detections."""
            self.images.append(image)
            return weaker, stronger

    predictor = _Predictor()
    detector = YoloEarLandmarkDetector(predictor=predictor)
    result = detector.detect(PHOTO, IMAGE, BoundingBox(10, 6, 23, 20))

    assert result is not None
    assert result.xyxy == (12.0, 9.0, 19.0, 18.0)
    assert result.keypoints == ((13.0, 10.0), (17.0, 16.0))
    assert predictor.images[0].shape == (14, 13, 3)


def test_processors_construct_without_loading_external_models() -> None:
    """Constructing raw processors does not initialize external clients."""
    features = Sam3FeatureSegmenter(api_key="unused")
    landmarks = YoloEarLandmarkDetector()

    assert features._client is None
    assert landmarks._predictor is None
