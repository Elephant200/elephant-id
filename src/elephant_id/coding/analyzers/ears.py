"""Ear field analyzer."""

from loguru import logger
import cv2
import numpy as np

from elephant_id.ai import Detection
from elephant_id.coding.curvature import oriented_curvature
from elephant_id.constants import DEFAULT_CURVATURE_RADII, DEFAULT_CURVATURE_WEIGHTS
from elephant_id.domain import Photo
from elephant_id.image.masks import RleMask, decode_rle_mask


def _closed_path(points: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    if start_idx <= end_idx:
        return points[start_idx : end_idx + 1]
    return np.concatenate([points[start_idx:], points[: end_idx + 1]])


def _cut_contour_by_anchors(
    contour: np.ndarray,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
) -> np.ndarray:
    """Return the longer open path between two anchors on a closed contour."""
    points = contour[:, 0, :]
    start = np.array(start_point)
    end = np.array(end_point)

    start_idx = int(np.argmin(np.sum((points - start) ** 2, axis=1)))
    end_idx = int(np.argmin(np.sum((points - end) ** 2, axis=1)))

    forward_path = _closed_path(points, start_idx, end_idx)
    backward_path = _closed_path(points, end_idx, start_idx)[::-1]

    if len(forward_path) >= len(backward_path):
        return forward_path
    return backward_path


class Ear:
    """A single ear of an elephant."""
    def __init__(self, ear_detection: Detection, anchor_prediction: Detection) -> None:
        """
        Initialize an ear from a detection and an associated anchor
        prediction.

        Args:
            ear_detection: The detection of the ear.
            anchor_prediction: The anchor detection of the ear; must have keypoints
        """
        if ear_detection.class_name != "ear":
            raise ValueError("Ear detection must be an ear")
        if anchor_prediction.keypoints is None or len(anchor_prediction.keypoints) != 2:
            raise ValueError("Anchor prediction must have exactly two keypoints")

        self.xyxy = ear_detection.xyxy
        self.rle_mask = ear_detection.rle_mask
        self.anchor_points = anchor_prediction.keypoints

        # Lazy properties
        self._mask = None
        self._contour = None

    def mask(self) -> RleMask:
        if self._mask is None:
            self._mask = decode_rle_mask(self.rle_mask)
        return self._mask

    def contour(self) -> np.ndarray:
        if self._contour is None:
            mask_u8 = np.ascontiguousarray(self.mask().astype(np.uint8) * 255)
            contours, _ = cv2.findContours(
                mask_u8,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            if not contours:
                raise ValueError("No contour found for ear mask")
            closed_contour = max(contours, key=cv2.contourArea)
            start_point, end_point = self.anchor_points
            self._contour = _cut_contour_by_anchors(
                closed_contour,
                start_point,
                end_point,
            )
        return self._contour

    def curvature(self) -> np.ndarray:
        return oriented_curvature(
            self.contour(),
            radii=DEFAULT_CURVATURE_RADII,
            weights=DEFAULT_CURVATURE_WEIGHTS,
        )


class EarAnalyzer:
    """Analyze each anchored ear: geometry plus stubbed tear/hole evidence."""

    def __init__(self) -> None:
        ...

    def analyze(self, photo: Photo, prep: dict) -> dict:
        ...