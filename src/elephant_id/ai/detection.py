"""Typed model-output detection shared across the AI services."""

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
from pycocotools import mask as coco_mask

from elephant_id.image.boxes import center_to_xyxy, clip_xyxy
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
    def area(self) -> float:
        """Mask area (exact pixel count) if masked, else box area."""
        if self.rle_mask is not None:
            return float(coco_mask.area([self.rle_mask])[0])
        x1, y1, x2, y2 = self.xyxy
        return (x2 - x1) * (y2 - y1)

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
        x1, y1, x2, y2 = self.xyxy
        return dataclasses.replace(
            self,
            xyxy=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
            keypoints=tuple((kx + dx, ky + dy) for kx, ky in self.keypoints),
        )

    def clip(self, image_width: int, image_height: int) -> "Detection":
        """Return a copy with the box clipped to image bounds.

        Keypoints are left untouched.
        """
        return dataclasses.replace(
            self, xyxy=clip_xyxy(*self.xyxy, image_width, image_height)
        )

    # --- adapters from raw model output ---
    @classmethod
    def from_sam3(cls, prediction: dict[str, Any]) -> "Detection":
        """Build from a raw Roboflow SAM3 prediction (center-format bbox)."""
        xyxy = center_to_xyxy(
            float(prediction["x"]),
            float(prediction["y"]),
            float(prediction["width"]),
            float(prediction["height"]),
        )
        return cls(
            xyxy=xyxy,
            class_name=str(prediction["class"]).strip(),  # SAM3 can emit leading whitespace
            class_id=int(prediction["class_id"]),
            confidence=float(prediction["confidence"]),
            rle_mask=prediction.get("rle_mask"),
        )

    @classmethod
    def from_anchor(cls, prediction: dict[str, Any]) -> "Detection":
        """Build from a raw ultralytics keypoint prediction entry."""
        box = prediction["box"]
        keypoints = prediction["keypoints"]
        return cls(
            xyxy=(box["x1"], box["y1"], box["x2"], box["y2"]),
            class_name=str(prediction["name"]).strip(),
            class_id=int(prediction["class"]),
            confidence=float(prediction["confidence"]),
            keypoints=tuple(zip(keypoints["x"], keypoints["y"], strict=True)),
        )

    # --- cache serialization ---
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for the cache."""
        x1, y1, x2, y2 = self.xyxy
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
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
