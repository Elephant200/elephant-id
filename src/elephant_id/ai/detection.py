"""Typed model-output detection shared across the AI services."""

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
from pycocotools import mask as coco_mask

from elephant_id.image.boxes import clip_xyxy
from elephant_id.image.masks import RleMask, decode_rle_mask


@dataclass(frozen=True, slots=True)
class Detection:
    """One model detection: a box, optional mask/keypoints, and a label.

    ``xyxy`` is image-space and half-open (x2/y2 exclusive), matching
    ``image.boxes``. Immutable; ``translate`` and ``clip`` return new
    instances. Not hashable (holds a dict).
    """

    xyxy: tuple[float, float, float, float]
    class_name: str
    class_id: int
    confidence: float
    rle_mask: RleMask | None = None
    keypoints: tuple[tuple[float, float], ...] = ()

    # --- geometry ---
    @property
    def x1(self) -> float:
        """Left edge of the detection box."""
        return self.xyxy[0]

    @property
    def y1(self) -> float:
        """Top edge of the detection box."""
        return self.xyxy[1]

    @property
    def x2(self) -> float:
        """Right edge of the detection box."""
        return self.xyxy[2]

    @property
    def y2(self) -> float:
        """Bottom edge of the detection box."""
        return self.xyxy[3]

    def area(self) -> float:
        """Mask area (exact pixel count) if masked, else box area."""
        if self.rle_mask is not None:
            return float(coco_mask.area([self.rle_mask])[0])
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def mask(self) -> np.ndarray:
        """Decoded boolean mask. Recomputed each access."""
        if self.rle_mask is None:
            raise ValueError("Detection has no mask")
        return decode_rle_mask(self.rle_mask)

    def intersection_area(self, other: "Detection") -> float:
        """Intersection area of this detection's mask and another detection's mask."""
        if self.rle_mask is None or other.rle_mask is None:
            raise ValueError("Both detections need a mask")
        return float(coco_mask.area([coco_mask.merge([self.rle_mask, other.rle_mask], intersect=True)])[0])

    def union_area(self, other: "Detection") -> float:
        """Union area of this detection's mask and another detection's mask."""
        if self.rle_mask is None or other.rle_mask is None:
            raise ValueError("Both detections need a mask")
        return float(coco_mask.area([coco_mask.merge([self.rle_mask, other.rle_mask], intersect=False)])[0])

    def iou(self, other: "Detection") -> float:
        """Intersection over union of this detection's mask and another detection's mask."""
        if self.rle_mask is None or other.rle_mask is None:
            raise ValueError("Both detections need a mask")
        union_area = self.union_area(other)
        if union_area == 0.0:
            return 0.0
        return self.intersection_area(other) / union_area

    # --- transforms (return new instances) ---
    def translate(self, dx: float, dy: float) -> "Detection":
        """Return a copy shifted by (dx, dy), moving box and keypoints."""
        return dataclasses.replace(
            self,
            xyxy=(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy),
            keypoints=tuple((kx + dx, ky + dy) for kx, ky in self.keypoints),
        )

    def clip(self, image_width: int, image_height: int) -> "Detection":
        """Return a copy with the box clipped to image bounds.

        Keypoints are left untouched.
        """
        return dataclasses.replace(
            self, xyxy=tuple(float(coord) for coord in clip_xyxy(*self.xyxy, image_width, image_height))
        )

    # --- cache serialization ---
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for the cache."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "rle_mask": self.rle_mask,
            "keypoints": [list(keypoint) for keypoint in self.keypoints],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        return cls(
            xyxy=(data["x1"], data["y1"], data["x2"], data["y2"]),
            class_name=data["class_name"],
            class_id=data["class_id"],
            confidence=data["confidence"],
            rle_mask=data.get("rle_mask"),
            keypoints=tuple(tuple(kp) for kp in data.get("keypoints", ()))
        )
