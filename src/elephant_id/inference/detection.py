"""Typed model-output detections shared by inference processors."""

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
from pycocotools import mask as coco_mask

from elephant_id.image.boxes import clip_xyxy
from elephant_id.image.masks import RleMask, decode_rle_mask


@dataclass(frozen=True, slots=True)
class Detection:
    """One immutable model detection.

    Semantic inference processors return full-image coordinates. Private model
    adapters may use this value transiently before translating crop output.
    """

    xyxy: tuple[float, float, float, float]
    class_name: str
    class_id: int
    confidence: float
    rle_mask: RleMask | None = None
    keypoints: tuple[tuple[float, float], ...] | None = None

    @property
    def x1(self) -> float:
        """Return the left edge."""
        return self.xyxy[0]

    @property
    def y1(self) -> float:
        """Return the top edge."""
        return self.xyxy[1]

    @property
    def x2(self) -> float:
        """Return the right edge."""
        return self.xyxy[2]

    @property
    def y2(self) -> float:
        """Return the bottom edge."""
        return self.xyxy[3]

    def area(self) -> float:
        """Return mask area when present, otherwise box area."""
        if self.rle_mask is not None:
            return float(coco_mask.area([self.rle_mask])[0])
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def get_mask(self) -> np.ndarray:
        """Return the decoded boolean mask."""
        if self.rle_mask is None:
            raise ValueError("Detection has no mask")
        return decode_rle_mask(self.rle_mask)

    def intersection_area(self, other: "Detection") -> float:
        """Return overlap area with another detection."""
        if self.rle_mask is not None and other.rle_mask is not None:
            intersection = coco_mask.merge(
                [self.rle_mask, other.rle_mask], intersect=True
            )
            return float(coco_mask.area([intersection])[0])
        if self.rle_mask is None and other.rle_mask is None:
            width = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
            height = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
            return width * height
        mask_detection, box_detection = (
            (self, other) if self.rle_mask is not None else (other, self)
        )
        mask = mask_detection.get_mask()
        x1, y1, x2, y2 = clip_xyxy(
            *box_detection.xyxy, mask.shape[1], mask.shape[0]
        )
        return float(np.sum(mask[y1:y2, x1:x2]))

    def union_area(self, other: "Detection") -> float:
        """Return combined coverage with another detection."""
        return self.area() + other.area() - self.intersection_area(other)

    def iou(self, other: "Detection") -> float:
        """Return intersection over union with another detection."""
        return self.intersection_area(other) / self.union_area(other)

    def translate(self, dx: float, dy: float) -> "Detection":
        """Return a copy shifted by ``dx`` and ``dy``.

        Raises:
            ValueError: If the detection contains a full-image mask.
        """
        if self.rle_mask is not None:
            raise ValueError("Cannot translate a detection with an RLE mask")
        return dataclasses.replace(
            self,
            xyxy=(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy),
            keypoints=(
                None
                if self.keypoints is None
                else tuple((x + dx, y + dy) for x, y in self.keypoints)
            ),
        )

    def clip(self, image_width: int, image_height: int) -> "Detection":
        """Return a copy with its half-open box clipped to image bounds."""
        return dataclasses.replace(
            self,
            xyxy=tuple(
                float(value)
                for value in clip_xyxy(
                    *self.xyxy,
                    image_width,
                    image_height,
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the detection to a JSON-compatible record."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "rle_mask": self.rle_mask,
            "keypoints": (
                None
                if self.keypoints is None
                else [list(keypoint) for keypoint in self.keypoints]
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        """Reconstruct a detection from its serialized record."""
        return cls(
            xyxy=(
                float(data["x1"]),
                float(data["y1"]),
                float(data["x2"]),
                float(data["y2"]),
            ),
            class_name=str(data["class_name"]),
            class_id=int(data["class_id"]),
            confidence=float(data["confidence"]),
            rle_mask=data.get("rle_mask"),
            keypoints=(
                None
                if data.get("keypoints") is None
                else tuple(
                    (float(point[0]), float(point[1]))
                    for point in data["keypoints"]
                )
            ),
        )
