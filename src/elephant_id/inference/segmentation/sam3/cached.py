"""Persistence decorator for SAM3 multi-feature segmentation."""

from elephant_id.cache import CacheManager
from elephant_id.domain import Photo
from elephant_id.image import BgrImage
from elephant_id.inference.detection import Detection
from elephant_id.inference.segmentation.sam3.features import _FeatureSegmenter


def _parse_detections(record: dict[str, object]) -> tuple[Detection, ...]:
    """Parse the complete detection list from a SAM3 cache record."""
    values = record.get("detections")
    if not isinstance(values, list):
        raise ValueError("SAM3 cache record must contain a detections list")
    try:
        return tuple(Detection.from_dict(value) for value in values)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("SAM3 cache record contains an invalid detection") from error


class CachedSam3FeatureSegmenter:
    """Persist the complete output of a SAM3 feature segmenter."""

    def __init__(self, inner: _FeatureSegmenter, cache: CacheManager) -> None:
        """Wrap one settled SAM3 feature processor."""
        self._inner = inner
        self._cache = cache

    @property
    def producer_slug(self) -> str:
        """Return the wrapped processor's stable slug."""
        return self._inner.producer_slug

    def segment_features(
        self,
        photo: Photo,
        image: BgrImage,
    ) -> tuple[Detection, ...]:
        """Return cached or newly computed complete feature detections."""
        def compute() -> dict[str, object]:
            """Serialize one complete SAM3 feature result."""
            detections = self._inner.segment_features(photo, image)
            return {"detections": [value.to_dict() for value in detections]}

        record = self._cache.get_or_compute(
            self.producer_slug,
            str(photo.photo_id),
            compute,
        )
        return _parse_detections(record)
