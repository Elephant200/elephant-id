"""Persistence decorator for ear-landmark detection."""

from math import isfinite

from elephant_id.cache import CacheManager
from elephant_id.domain import Photo
from elephant_id.image import BgrImage
from elephant_id.image.boxes import BoundingBox
from elephant_id.inference.detection import Detection
from elephant_id.inference.landmarks.protocol import EarLandmarkDetector


class CachedEarLandmarkDetector:
    """Persist full-image output from an ear-landmark detector."""

    def __init__(self, inner: EarLandmarkDetector, cache: CacheManager) -> None:
        """Wrap one landmark detector with shared JSON persistence."""
        self._inner = inner
        self._cache = cache

    @property
    def producer_slug(self) -> str:
        """Return the wrapped detector's stable slug."""
        return self._inner.producer_slug

    def detect(
        self,
        photo: Photo,
        image: BgrImage,
        ear_box: BoundingBox,
    ) -> Detection | None:
        """Return cached or newly detected full-image landmarks."""
        key = f"{photo.photo_id}__crop_{'_'.join(map(str, ear_box.as_tuple()))}"

        def compute() -> dict[str, object]:
            """Serialize one optional full-image landmark detection."""
            detection = self._inner.detect(photo, image, ear_box)
            return {"detection": None if detection is None else detection.to_dict()}

        record = self._cache.get_or_compute(self.producer_slug, key, compute)
        if "detection" not in record:
            raise ValueError("Landmark cache record must contain detection")
        value = record.get("detection")
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("Landmark cache detection must be an object or null")
        try:
            detection = Detection.from_dict(value)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Landmark cache detection is invalid") from error
        if (
            detection.keypoints is None
            or len(detection.keypoints) != 2
            or not all(isfinite(value) for value in detection.xyxy)
            or not all(
                isfinite(coordinate)
                for point in detection.keypoints
                for coordinate in point
            )
        ):
            raise ValueError("Landmark cache detection is invalid")
        return detection
