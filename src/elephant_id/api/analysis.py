"""Build V1-preview analysis payloads from stored sighting profiles."""

from pathlib import Path

import cv2
import numpy as np

EAR_CANDIDATE_MIN_ASPECT = 0.7
EAR_CANDIDATE_MAX_ASPECT = 0.9
EAR_CANDIDATE_LIMIT = 3
EAR_SIDES = ("left", "right")


def candidate_id(row_index: int, side: str, photo_id: str) -> str:
    """Return a stable ear-candidate identifier for one profile row."""
    return f"profile:{row_index}:{side}:{photo_id}"


def workflow_status(record: dict) -> tuple[str, str]:
    """Return the preview queue status and next action for a sighting record."""
    decision = record.get("decision")
    if decision and decision.get("action") == "unresolved":
        return "Unresolved", "Review saved"
    if decision:
        return "Decided", "View confirmation"
    if record.get("status") == "failed":
        return "Failed", "Inspect error"
    if record.get("status") == "analyzing":
        return "Analyzing", "Wait for analysis"
    if not record.get("approved_evidence"):
        return "Needs evidence review", "Approve left and right ears"
    if not record.get("match"):
        return "Ready to match", "Rank known-elephant catalog"
    return "Needs decision", "Confirm identity decision"


def decorate_record(record: dict) -> dict:
    """Attach preview workflow labels to a stored sighting record."""
    decorated = dict(record)
    status, next_action = workflow_status(record)
    decorated["workflow_status"] = status
    decorated["next_action"] = next_action
    decorated.setdefault("approved_evidence", None)
    return decorated


def analysis_payload(
    record: dict,
    profiles: np.ndarray | None,
    sides: tuple[str, ...] = (),
    photo_ids: tuple[str, ...] = (),
    crop_paths: tuple[str | None, ...] = (),
    row_geometry: list[dict] | None = None,
) -> dict:
    """Return an analysis package summary with ranked ear candidates.

    ``row_geometry`` optionally supplies per-row clean crop paths and ear
    contours (crop-local coordinates) for the contour-correction editor.
    """
    payload = decorate_record(record)
    candidates = {"left": [], "right": []}
    geometry = row_geometry or []
    if profiles is not None and len(profiles) > 0:
        photo_lookup = _photo_lookup(record.get("photos", []))
        for row_index, (side, photo_id, crop_path) in enumerate(
            zip(sides, photo_ids, crop_paths, strict=True)
        ):
            if side not in candidates or not crop_path:
                continue
            dimensions = _image_dimensions(crop_path)
            if dimensions is None:
                continue
            width, height = dimensions
            aspect_ratio = width / height
            in_aspect_band = (
                EAR_CANDIDATE_MIN_ASPECT
                <= aspect_ratio
                <= EAR_CANDIDATE_MAX_ASPECT
            )
            photo = photo_lookup.get(photo_id, {})
            pixel_area = width * height
            row_geo = geometry[row_index] if row_index < len(geometry) else {}
            candidates[side].append(
                {
                    "candidate_id": candidate_id(row_index, side, photo_id),
                    "profile_row_index": row_index,
                    "side": side,
                    "photo_id": photo_id,
                    "file_name": photo.get("file_name") or photo_id,
                    "photo_path": photo.get("photo_path"),
                    "crop_path": crop_path,
                    "display_crop_path": row_geo.get("clean_crop_path") or crop_path,
                    "crop_width": width,
                    "crop_height": height,
                    "aspect_ratio": round(float(aspect_ratio), 3),
                    "in_aspect_band": in_aspect_band,
                    "pixel_area": pixel_area,
                    "clean_crop_path": row_geo.get("clean_crop_path"),
                    "contour": row_geo.get("contour"),
                    "ranking_note": (
                        "Temporary preview ranking: aspect ratio 0.7-0.9 "
                        "preferred, then larger crop area first."
                    ),
                }
            )
    for side in EAR_SIDES:
        candidates[side].sort(
            key=lambda item: (not item["in_aspect_band"], -item["pixel_area"])
        )
        candidates[side] = candidates[side][:EAR_CANDIDATE_LIMIT]
    payload["ear_candidates"] = candidates
    payload["approved_evidence"] = record.get("approved_evidence")
    payload["requires_both_sides"] = True
    payload["can_approve_evidence"] = all(candidates[side] for side in EAR_SIDES)
    return payload


def approved_profile_rows(
    record: dict,
    profiles: np.ndarray,
    sides: tuple[str, ...],
    photo_ids: tuple[str, ...],
    crop_paths: tuple[str | None, ...],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str | None, ...]]:
    """Return only the reviewer-approved left and right profile rows.

    Raises:
        ValueError: If the sighting does not have valid approved evidence.
    """
    approved = record.get("approved_evidence")
    if not isinstance(approved, dict):
        raise ValueError("Evidence review must approve one left and one right ear")
    rows: list[int] = []
    for side in EAR_SIDES:
        item = approved.get(side)
        if not isinstance(item, dict):
            raise ValueError("Evidence review must approve one left and one right ear")
        row_index = int(item.get("profile_row_index", -1))
        if row_index < 0 or row_index >= len(profiles):
            raise ValueError(f"Approved {side} evidence is no longer available")
        if sides[row_index] != side:
            raise ValueError(f"Approved {side} evidence points at a {sides[row_index]} row")
        rows.append(row_index)
    selected = np.asarray(rows, dtype=np.int64)
    return (
        np.asarray(profiles[selected], dtype=np.float64),
        tuple(sides[index] for index in rows),
        tuple(photo_ids[index] for index in rows),
        tuple(crop_paths[index] for index in rows),
    )


def _photo_lookup(photos: list[dict]) -> dict[str, dict]:
    """Map stored photo identifiers to photo records."""
    lookup = {}
    for photo in photos:
        photo_id = photo.get("photo_id")
        if photo_id and photo_id not in lookup:
            lookup[str(photo_id)] = photo
    return lookup


def _image_dimensions(path: str) -> tuple[int, int] | None:
    """Return image dimensions as ``(width, height)`` if the crop is readable."""
    image = cv2.imread(str(Path(path).expanduser()))
    if image is None:
        return None
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return None
    return width, height
