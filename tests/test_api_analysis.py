"""Tests for the V1-preview analysis payload builder."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from elephant_id.api.analysis import (
    EAR_CANDIDATE_LIMIT,
    analysis_payload,
)


def _write_crop(directory: Path, name: str, width: int, height: int) -> str:
    """Write a synthetic crop image and return its path."""
    path = directory / name
    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))
    return str(path)


def _record() -> dict:
    """Return a minimal ready sighting record."""
    return {"sighting_id": "s1", "status": "ready", "photos": []}


def _payload(crops: list[tuple[str, str]]) -> dict:
    """Build an analysis payload from (side, crop_path) rows."""
    profiles = np.zeros((len(crops), 4))
    sides = tuple(side for side, _ in crops)
    photo_ids = tuple(f"photo{i}" for i in range(len(crops)))
    crop_paths = tuple(path for _, path in crops)
    return analysis_payload(_record(), profiles, sides, photo_ids, crop_paths)


def test_out_of_band_crops_still_rank(tmp_path: Path) -> None:
    """A side with only out-of-band aspect ratios must not be empty."""
    crops = [
        ("left", _write_crop(tmp_path, "a.png", 200, 200)),  # aspect 1.0
        ("left", _write_crop(tmp_path, "b.png", 300, 100)),  # aspect 3.0
    ]
    payload = _payload(crops)
    left = payload["ear_candidates"]["left"]
    assert len(left) == 2
    assert left[0]["pixel_area"] >= left[1]["pixel_area"]
    assert all(candidate["in_aspect_band"] is False for candidate in left)


def test_in_band_preferred_over_larger_out_of_band(tmp_path: Path) -> None:
    """An in-band crop outranks a larger out-of-band crop."""
    crops = [
        ("left", _write_crop(tmp_path, "big.png", 500, 500)),  # out of band, huge
        ("left", _write_crop(tmp_path, "band.png", 120, 100)),  # aspect 1.2
    ]
    payload = _payload(crops)
    left = payload["ear_candidates"]["left"]
    assert left[0]["in_aspect_band"] is True
    assert left[1]["in_aspect_band"] is False


def test_candidate_limit_and_area_order(tmp_path: Path) -> None:
    """Candidates are capped at the limit, larger areas first within the band."""
    crops = [
        ("right", _write_crop(tmp_path, f"c{i}.png", 120 * (i + 1), 100 * (i + 1)))
        for i in range(EAR_CANDIDATE_LIMIT + 2)
    ]
    payload = _payload(crops)
    right = payload["ear_candidates"]["right"]
    assert len(right) == EAR_CANDIDATE_LIMIT
    areas = [candidate["pixel_area"] for candidate in right]
    assert areas == sorted(areas, reverse=True)


def test_unreadable_crops_are_skipped(tmp_path: Path) -> None:
    """Rows without a readable crop are dropped, not crashed on."""
    crops = [
        ("left", str(tmp_path / "missing.png")),
        ("left", _write_crop(tmp_path, "ok.png", 120, 100)),
    ]
    payload = _payload(crops)
    assert len(payload["ear_candidates"]["left"]) == 1


def test_can_approve_requires_both_sides(tmp_path: Path) -> None:
    """Approval readiness needs at least one candidate per side."""
    crops = [("left", _write_crop(tmp_path, "l.png", 120, 100))]
    payload = _payload(crops)
    assert payload["can_approve_evidence"] is False


@pytest.mark.parametrize("side", ["left", "right"])
def test_both_sides_populated(tmp_path: Path, side: str) -> None:
    """Each side receives its own candidates."""
    crops = [
        ("left", _write_crop(tmp_path, "l.png", 120, 100)),
        ("right", _write_crop(tmp_path, "r.png", 240, 200)),
    ]
    payload = _payload(crops)
    assert len(payload["ear_candidates"][side]) == 1
