"""Anchored ear contour preparation."""

from typing import Literal

import cv2
import numpy as np
from pycocotools import mask as coco_mask

from elephant_id.ai.detection import Detection
from elephant_id.coding.ears.geometry import resample2d
from elephant_id.image.masks import RleMask, decode_rle_mask


def _closed_path(points: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    """Return the closed-contour slice from start to end, wrapping if needed."""
    if start_idx <= end_idx:
        return points[start_idx : end_idx + 1]
    return np.concatenate([points[start_idx:], points[: end_idx + 1]])


def snap_points_to_contour(
    contour: np.ndarray,
    points: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Snap points to their nearest contour points."""
    contour_points = contour[:, 0, :]
    snapped = []
    for point in points:
        distances = np.sum((contour_points - np.array(point)) ** 2, axis=1)
        snapped_point = contour_points[int(np.argmin(distances))]
        snapped.append((float(snapped_point[0]), float(snapped_point[1])))
    return (snapped[0], snapped[1])


def cut_contour_by_anchors(
    contour: np.ndarray,
    anchor_points: tuple[tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    """Return the longer vertex path between two anchors on a closed contour."""
    points = contour[:, 0, :]

    # In image coordinates, smaller y is visually higher.
    start_point = min(anchor_points, key=lambda p: p[1])
    end_point = max(anchor_points, key=lambda p: p[1])

    start = np.array(start_point)
    end = np.array(end_point)

    start_idx = int(np.argmin(np.sum((points - start) ** 2, axis=1)))
    end_idx = int(np.argmin(np.sum((points - end) ** 2, axis=1)))

    forward_path = _closed_path(points, start_idx, end_idx)
    backward_path = _closed_path(points, end_idx, start_idx)[::-1]

    if len(forward_path) >= len(backward_path):
        return forward_path
    return backward_path


def largest_contour_from_rle(rle_mask: RleMask) -> np.ndarray:
    """Return the largest contour in an RLE mask."""
    mask_u8 = decode_rle_mask(rle_mask).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError("No contour found for ear mask")
    return max(contours, key=cv2.contourArea)


def encode_rle_mask(mask: np.ndarray) -> RleMask:
    """Encode a boolean mask as COCO RLE."""
    rle_mask = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle_mask["counts"]
    return {
        "size": list(mask.shape),
        "counts": counts.decode("utf-8") if isinstance(counts, bytes) else counts,
    }


class AnchoredEar:
    """An ear mask with anchor-to-anchor contour geometry."""

    def __init__(self, ear_detection: Detection, anchor_detection: Detection) -> None:
        """Initialize an ear from an ear detection and anchor keypoint detection."""
        if ear_detection.class_name != "ear":
            raise ValueError("Ear detection must be an ear")
        if anchor_detection.keypoints is None or len(anchor_detection.keypoints) != 2:
            raise ValueError("Anchor prediction must have exactly two keypoints")
        if ear_detection.rle_mask is None:
            raise ValueError("Ear detection must have an RLE mask")

        self.xyxy = ear_detection.xyxy
        self._mask_size = tuple(ear_detection.rle_mask["size"])
        self._source_contour = largest_contour_from_rle(ear_detection.rle_mask)
        self.anchor_points = snap_points_to_contour(
            self._source_contour,
            anchor_detection.keypoints,
        )

        # TODO: Implement lazy loading
        self._cut_contour = cut_contour_by_anchors(
            self._source_contour,
            self.anchor_points,
        )
        self._mask = self._build_mask()
        self._rle_mask = encode_rle_mask(self._mask)
        self.area = float(coco_mask.area([self.rle_mask])[0])

        anchor_center_x = (self.anchor_points[0][0] + self.anchor_points[1][0]) / 2
        box_center_x = (self.xyxy[0] + self.xyxy[2]) / 2
        self.side: Literal["left", "right"] = (
            "left" if anchor_center_x < box_center_x else "right"
        )

        # Placeholder until quality scoring combines area and aspect ratio.
        self.quality = 0.0

    @property
    def rle_mask(self) -> RleMask:
        """Cleaned ear mask encoded as COCO RLE."""
        return {
            "size": list(self._rle_mask["size"]),
            "counts": self._rle_mask["counts"],
        }

    @property
    def mask(self) -> np.ndarray:
        """Cleaned ear mask."""
        return self._mask.copy()

    @property
    def contour(self) -> np.ndarray:
        """Cleaned ear contour cut between the snapped anchor points."""
        return self._cut_contour.copy()

    def _build_mask(self) -> np.ndarray:
        """Build the cleaned ear mask from the cut contour."""
        height, width = self._mask_size
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = np.round(self._cut_contour).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [polygon], color=1)
        return mask.astype(bool)

    def resampled_contour(self, num_points: int = 1024) -> np.ndarray:
        """Return the cleaned ear contour resampled to a fixed point count."""
        return resample2d(self._cut_contour, num_points=num_points)

    def __repr__(self) -> str:
        """Return a compact debug description."""
        return f"AnchoredEar(side={self.side}, xyxy={self.xyxy}, anchor_points={self.anchor_points})"
