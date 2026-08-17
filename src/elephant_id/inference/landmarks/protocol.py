"""Semantic ear-landmark detection protocol."""

from typing import Protocol

from elephant_id.domain import Photo
from elephant_id.image import BgrImage
from elephant_id.image.boxes import BoundingBox
from elephant_id.inference.detection import Detection


class EarLandmarkDetector(Protocol):
    """Detect the strongest anatomical landmark pair for one ear crop."""

    @property
    def producer_slug(self) -> str:
        """Return the stable identity of this deterministic processor."""
        ...

    def detect(
        self,
        photo: Photo,
        image: BgrImage,
        ear_box: BoundingBox,
    ) -> Detection | None:
        """Return full-image landmarks, or none for an ordinary zero detection."""
        ...
