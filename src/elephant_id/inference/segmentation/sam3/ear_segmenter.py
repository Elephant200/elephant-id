"""Ear-only semantic adapter for SAM3 feature segmentation."""

from loguru import logger

from elephant_id.domain import Photo
from elephant_id.image import BgrImage
from elephant_id.inference.detection import Detection
from elephant_id.inference.segmentation.sam3.features import _FeatureSegmenter


class Sam3EarSegmenter:
    """Adapt complete SAM3 feature output to semantic ear segmentation."""

    def __init__(self, features: _FeatureSegmenter) -> None:
        """Wrap one raw or cached SAM3 feature processor."""
        self._features = features

    @property
    def producer_slug(self) -> str:
        """Return the underlying deterministic feature slug."""
        return self._features.producer_slug

    def segment(self, photo: Photo, image: BgrImage) -> tuple[Detection, ...]:
        """Return only ear detections in full-image coordinates."""
        ears = tuple(
            detection
            for detection in self._features.segment_features(photo, image)
            if detection.class_name == "ear"
        )
        logger.info(
            f"Segmented ears for photo {photo.photo_id}: {len(ears)} detections"
        )
        return ears
