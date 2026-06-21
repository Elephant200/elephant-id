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

        self._ear_rle_mask = ear_detection.rle_mask
        self._mask_size = tuple(ear_detection.rle_mask["size"])
        self._keypoints = anchor_detection.keypoints
        self._original_anchor_points = tuple(
            sorted(anchor_detection.keypoints, key=lambda point: point[1])
        )

        # The cut contour is the source of truth for the cheap geometry.
        self._cut_contour: np.ndarray | None = None
        self._xyxy: tuple[float, float, float, float] | None = None
        self._anchor_points: (
            tuple[tuple[float, float], tuple[float, float]] | None
        ) = None
        self._side: Literal["left", "right"] | None = None
        self._mask: np.ndarray | None = None
        self._rle_mask: RleMask | None = None
        self._area: float | None = None

        # Placeholder until quality scoring combines area and aspect ratio.
        self.quality = 0.0

    def _ensure_geometry(self) -> None:
        """Compute the cut contour and its cheap derivatives in one pass."""
        if self._cut_contour is not None:
            return

        source_contour = largest_contour_from_rle(self._ear_rle_mask)
        snapped = snap_points_to_contour(source_contour, self._keypoints)
        contour = cut_contour_by_anchors(source_contour, snapped)

        poly = np.round(contour).astype(np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        xyxy = (float(x), float(y), float(x + w), float(y + h))

        anchor_points = (
            (float(contour[0][0]), float(contour[0][1])),
            (float(contour[-1][0]), float(contour[-1][1])),
        )

        anchor_center_x = (anchor_points[0][0] + anchor_points[1][0]) / 2
        box_center_x = (xyxy[0] + xyxy[2]) / 2
        self._side = "left" if anchor_center_x < box_center_x else "right"

        self._xyxy = xyxy
        self._anchor_points = anchor_points
        self._cut_contour = contour

    def _ensure_mask(self) -> None:
        """Rasterize the cut contour and calculate its pixel area."""
        if self._mask is not None:
            return

        self._ensure_geometry()
        height, width = self._mask_size
        mask = np.zeros((height, width), dtype=np.uint8)
        polygon = np.round(self._cut_contour).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [polygon], color=1)

        self._mask = mask.astype(bool)
        self._area = float(np.count_nonzero(self._mask))

    def _ensure_rle_mask(self) -> None:
        """Encode the cleaned mask only when a caller needs its RLE."""
        if self._rle_mask is not None:
            return
        self._ensure_mask()
        self._rle_mask = encode_rle_mask(self._mask)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Bounding box of the cut contour (half-open, x2/y2 exclusive)."""
        self._ensure_geometry()
        return self._xyxy

    @property
    def anchor_points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The two anchor points, snapped onto the cut contour."""
        self._ensure_geometry()
        return self._anchor_points

    @property
    def original_anchor_points(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Model anchor points ordered upper to lower before contour snapping."""
        return self._original_anchor_points

    @property
    def side(self) -> Literal["left", "right"]:
        """Whether this is the elephant's left or right ear."""
        self._ensure_geometry()
        return self._side

    @property
    def area(self) -> float:
        """Area of the cleaned ear mask in pixels."""
        self._ensure_mask()
        return self._area

    @property
    def rle_mask(self) -> RleMask:
        """Cleaned ear mask encoded as COCO RLE."""
        self._ensure_rle_mask()
        return {
            "size": list(self._rle_mask["size"]),
            "counts": self._rle_mask["counts"],
        }

    @property
    def mask(self) -> np.ndarray:
        """Cleaned ear mask."""
        self._ensure_mask()
        return self._mask.copy()

    @property
    def contour(self) -> np.ndarray:
        """Cleaned ear contour cut between the snapped anchor points."""
        self._ensure_geometry()
        return self._cut_contour.copy()

    def resampled_contour(self, num_points: int = 1024) -> np.ndarray:
        """Return the cleaned ear contour resampled to a fixed point count."""
        self._ensure_geometry()
        return resample2d(self._cut_contour, num_points=num_points)

    def __repr__(self) -> str:
        """Return a compact debug description."""
        return f"AnchoredEar(side={self.side}, xyxy={self.xyxy}, anchor_points={self.anchor_points})"
