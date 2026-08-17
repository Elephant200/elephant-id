"""Semantic ear-segmentation protocol."""

from typing import Protocol

from elephant_id.domain import Photo
from elephant_id.image import BgrImage
from elephant_id.inference.detection import Detection


class EarSegmenter(Protocol):
    """Locate and segment ears in one source photo."""

    @property
    def producer_slug(self) -> str:
        """Return the stable identity of this deterministic processor."""
        ...

    def segment(self, photo: Photo, image: BgrImage) -> tuple[Detection, ...]:
        """Return only ear detections in full-image coordinates."""
        ...
