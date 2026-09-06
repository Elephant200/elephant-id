"""Prepare segmented ears for AlphaTear extraction."""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from elephant_id.domain import Photo
from elephant_id.image.boxes import BoundingBox
from elephant_id.image.masks import RleMask, decode_rle_mask
from elephant_id.inference import Detection

EarSide = Literal["left", "right"]
LandmarkPair = tuple[tuple[float, float], tuple[float, float]]


def _closed_path(
    points: NDArray[np.float64], start: int, end: int
) -> NDArray[np.float64]:
    """Return a closed-contour slice, wrapping when necessary."""
    if start <= end:
        return points[start : end + 1]
    return np.concatenate([points[start:], points[: end + 1]])


def _largest_contour(rle_mask: RleMask) -> NDArray[np.float64]:
    """Return the largest external contour in an RLE mask."""
    mask = decode_rle_mask(rle_mask).view(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found for ear mask")
    return max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)


def _snap_landmarks(
    contour: NDArray[np.float64],
    landmarks: LandmarkPair,
) -> LandmarkPair:
    """Snap two landmarks independently to their nearest contour vertices."""
    snapped: list[tuple[float, float]] = []
    for landmark in landmarks:
        distances = np.sum((contour - np.asarray(landmark)) ** 2, axis=1)
        point = contour[int(np.argmin(distances))]
        snapped.append((float(point[0]), float(point[1])))
    return snapped[0], snapped[1]


def _cut_contour(
    contour: NDArray[np.float64],
    snapped_landmarks: LandmarkPair,
) -> NDArray[np.float64]:
    """Return the longer vertex path from upper to lower snapped landmark."""
    upper = min(snapped_landmarks, key=lambda point: point[1])
    lower = max(snapped_landmarks, key=lambda point: point[1])
    upper_index = int(np.argmin(np.sum((contour - np.asarray(upper)) ** 2, axis=1)))
    lower_index = int(np.argmin(np.sum((contour - np.asarray(lower)) ** 2, axis=1)))
    forward = _closed_path(contour, upper_index, lower_index)
    backward = _closed_path(contour, lower_index, upper_index)[::-1]
    return forward if len(forward) >= len(backward) else backward


@dataclass(frozen=True, slots=True, eq=False)
class PreparedEar:
    """Immutable full-image ear geometry ready for profile extraction."""

    source_photo: Photo
    source_box: BoundingBox
    contour: NDArray[np.float64]
    original_landmarks: LandmarkPair
    contour_anchors: LandmarkPair
    inferred_side: EarSide
    cleaned_area: float

    def __post_init__(self) -> None:
        """Copy and validate the prepared contour invariants."""
        contour = np.array(self.contour, dtype=np.float64, copy=True)
        if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 2:
            raise ValueError("Prepared ear contour must be a two-column polyline")
        if not np.isfinite(contour).all():
            raise ValueError("Prepared ear contour must contain finite coordinates")
        if self.cleaned_area <= 0:
            raise ValueError("Prepared ear cleaned area must be positive")
        if not np.array_equal(contour[0], self.contour_anchors[0]):
            raise ValueError("Prepared ear contour must start at its upper anchor")
        if not np.array_equal(contour[-1], self.contour_anchors[1]):
            raise ValueError("Prepared ear contour must end at its lower anchor")
        contour.setflags(write=False)
        object.__setattr__(self, "contour", contour)


def prepare_ear(
    ear_detection: Detection,
    landmark_detection: Detection,
    *,
    source_photo: Photo,
    source_box: BoundingBox,
) -> PreparedEar:
    """Prepare one segmented ear and its detected landmarks for AlphaTear."""
    if ear_detection.rle_mask is None:
        raise ValueError("Ear preparation requires a segmentation mask")
    if landmark_detection.keypoints is None or len(landmark_detection.keypoints) != 2:
        raise ValueError("Ear preparation requires exactly two landmarks")

    original_landmarks = tuple(
        sorted(landmark_detection.keypoints, key=lambda point: point[1])
    )
    source_contour = _largest_contour(ear_detection.rle_mask)
    snapped = _snap_landmarks(source_contour, original_landmarks)
    contour = _cut_contour(source_contour, snapped)
    contour_anchors: LandmarkPair = (
        (float(contour[0, 0]), float(contour[0, 1])),
        (float(contour[-1, 0]), float(contour[-1, 1])),
    )

    x, _, width, _ = cv2.boundingRect(np.round(contour).astype(np.int32))
    anchor_center_x = (contour_anchors[0][0] + contour_anchors[1][0]) / 2
    inferred_side: EarSide = "left" if anchor_center_x < x + width / 2 else "right"

    height, width = tuple(ear_detection.rle_mask["size"])
    cleaned_mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.round(contour).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(cleaned_mask, [polygon], color=1)

    return PreparedEar(
        source_photo=source_photo,
        source_box=source_box,
        contour=contour,
        original_landmarks=original_landmarks,
        contour_anchors=contour_anchors,
        inferred_side=inferred_side,
        cleaned_area=float(np.count_nonzero(cleaned_mask)),
    )
