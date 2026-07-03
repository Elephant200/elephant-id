"""Tests for the V1-preview sidecar workflow routes."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from elephant_id.api import app as api_app
from elephant_id.api.gallery import GalleryData
from elephant_id.api.store import SightingStore
from elephant_id.api.workflow import SightingWorkflow
from elephant_id.constants import TEAR_PROFILE_BINS


class FakeCandidate:
    """JSON-serializable fake match result."""

    def __init__(self, identity: str, query_photo_id: str) -> None:
        """Store minimal candidate evidence."""
        self.identity = identity
        self.query_photo_id = query_photo_id

    def to_dict(self) -> dict:
        """Return a fake ranked candidate."""
        return {
            "identity": self.identity,
            "score": 1.5,
            "confidence": 0.82,
            "evidence": [
                {
                    "side": "left",
                    "score": 1.5,
                    "query_photo_id": self.query_photo_id,
                    "gallery_photo_id": "Known_2020-01-01_01",
                    "gallery_date": "2020-01-01",
                    "gallery_crop_path": None,
                    "query_profile": (0.1, 0.2),
                    "gallery_profile": (0.1, 0.2),
                    "strength": "strong",
                }
            ],
        }


class FakeEngine:
    """Fake matching engine recording rank and extend calls."""

    elephant_count = 1
    profile_count = 2

    def __init__(self) -> None:
        """Initialize recorded calls."""
        self.rank_calls: list[tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]] = []
        self.extend_calls: list[tuple[np.ndarray, tuple[str, ...], str]] = []

    def rank(
        self,
        profiles: np.ndarray,
        sides: tuple[str, ...],
        photo_ids: tuple[str, ...],
        top_n: int = 12,
    ) -> list[FakeCandidate]:
        """Record the ranked query rows."""
        self.rank_calls.append((profiles.copy(), tuple(sides), tuple(photo_ids)))
        return [FakeCandidate("Known", photo_ids[0])][:top_n]

    def has_identity(self, identity: str) -> bool:
        """Return whether a fake known elephant exists."""
        return identity == "Known"

    def extend(
        self,
        profiles: np.ndarray,
        sides: tuple[str, ...],
        identity: str,
        date: str,
        photo_ids: tuple[str, ...],
        crop_paths: tuple[str | None, ...],
    ) -> None:
        """Record filed evidence."""
        self.extend_calls.append((profiles.copy(), tuple(sides), identity))


class FakeState:
    """AppState replacement with temp store and fake matching engine."""

    def __init__(self, data_dir: Path, engine: FakeEngine) -> None:
        """Create a fake state object for route tests."""
        self.data_dir = data_dir
        self.store = SightingStore(data_dir)
        self.gallery = GalleryData(
            profiles=np.zeros((0, TEAR_PROFILE_BINS)),
            photo_ids=(),
            identities=(),
            sides=(),
            dates=(),
            crop_paths=(),
        )
        self.engine = engine
        self.engine_error = None
        self.workflow = SightingWorkflow(self.store, self.gallery, data_dir)

    def allowed_image_roots(self) -> list[Path]:
        """Allow route tests to serve temp images."""
        return [self.data_dir]


@pytest.fixture
def preview_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[TestClient, SightingStore, FakeEngine, str]:
    """Return a sidecar client with one ready analyzed sighting."""
    engine = FakeEngine()
    state = FakeState(tmp_path, engine)
    monkeypatch.setattr(api_app, "AppState", lambda data_dir: state)
    client = TestClient(api_app.create_app(tmp_path))
    sighting_id = _ready_sighting(state.store, tmp_path)
    return client, state.store, engine, sighting_id


def test_analysis_groups_ranked_candidates_with_aspect_preference(
    preview_client: tuple[TestClient, SightingStore, FakeEngine, str],
) -> None:
    """Analysis ranks in-band candidates by area first, then out-of-band ones."""
    client, _, _, sighting_id = preview_client

    response = client.get(f"/sightings/{sighting_id}/analysis")

    assert response.status_code == 200
    payload = response.json()
    assert [item["profile_row_index"] for item in payload["ear_candidates"]["left"]] == [
        1,
        0,
        2,
    ]
    assert [item["profile_row_index"] for item in payload["ear_candidates"]["right"]] == [
        3,
        4,
    ]
    assert payload["can_approve_evidence"] is True


def test_evidence_approval_requires_one_left_and_one_right(
    preview_client: tuple[TestClient, SightingStore, FakeEngine, str],
) -> None:
    """Approval rejects missing or same-side candidate selections."""
    client, _, _, sighting_id = preview_client
    analysis = client.get(f"/sightings/{sighting_id}/analysis").json()
    left = analysis["ear_candidates"]["left"][0]["candidate_id"]

    response = client.post(
        f"/sightings/{sighting_id}/approve-evidence",
        json={"left_candidate_id": left, "right_candidate_id": left},
    )

    assert response.status_code == 400
    assert "one valid left and one valid right" in response.json()["detail"]


def test_matching_requires_approved_evidence_then_uses_approved_rows_only(
    preview_client: tuple[TestClient, SightingStore, FakeEngine, str],
) -> None:
    """Matching is gated on approval and sends only the two approved profile rows."""
    client, _, engine, sighting_id = preview_client

    blocked = client.post(f"/sightings/{sighting_id}/match", json={"top_n": 4})
    assert blocked.status_code == 409

    analysis = client.get(f"/sightings/{sighting_id}/analysis").json()
    left = analysis["ear_candidates"]["left"][0]["candidate_id"]
    right = analysis["ear_candidates"]["right"][0]["candidate_id"]
    approved = client.post(
        f"/sightings/{sighting_id}/approve-evidence",
        json={"left_candidate_id": left, "right_candidate_id": right},
    )
    assert approved.status_code == 200

    matched = client.post(f"/sightings/{sighting_id}/match", json={"top_n": 4})

    assert matched.status_code == 200
    assert matched.json()["match"]["candidates"][0]["identity"] == "Known"
    assert len(engine.rank_calls) == 1
    profiles, sides, photo_ids = engine.rank_calls[0]
    assert profiles.shape == (2, TEAR_PROFILE_BINS)
    assert sides == ("left", "right")
    assert photo_ids == ("PhotoA", "PhotoB")
    np.testing.assert_allclose(profiles[0], np.full(TEAR_PROFILE_BINS, 0.2))
    np.testing.assert_allclose(profiles[1], np.full(TEAR_PROFILE_BINS, 0.4))


def test_unresolved_decision_can_be_recorded_without_match(
    preview_client: tuple[TestClient, SightingStore, FakeEngine, str],
) -> None:
    """Unresolved identity decisions do not require match results."""
    client, _, _, sighting_id = preview_client

    response = client.post(
        f"/sightings/{sighting_id}/decision",
        json={"action": "unresolved"},
    )

    assert response.status_code == 200
    assert response.json()["decision"]["action"] == "unresolved"
    assert response.json()["workflow_status"] == "Unresolved"


def test_v1_decision_state_names_persist(
    preview_client: tuple[TestClient, SightingStore, FakeEngine, str],
) -> None:
    """New V1 decision state names are stored and filed with approved rows."""
    client, _, engine, sighting_id = preview_client
    analysis = client.get(f"/sightings/{sighting_id}/analysis").json()
    client.post(
        f"/sightings/{sighting_id}/approve-evidence",
        json={
            "left_candidate_id": analysis["ear_candidates"]["left"][0]["candidate_id"],
            "right_candidate_id": analysis["ear_candidates"]["right"][0]["candidate_id"],
        },
    )

    response = client.post(
        f"/sightings/{sighting_id}/decision",
        json={"action": "new_known_elephant", "elephant_name": "Newbie"},
    )

    assert response.status_code == 200
    assert response.json()["decision"]["action"] == "new_known_elephant"
    assert response.json()["decision"]["elephant_name"] == "Newbie"
    assert engine.extend_calls[0][1] == ("left", "right")
    assert engine.extend_calls[0][2] == "Newbie"


def _ready_sighting(store: SightingStore, tmp_path: Path) -> str:
    """Create a ready sighting with valid and invalid crop dimensions."""
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    crop_paths = (
        _crop(crop_dir / "left-small.jpg", 120, 100),
        _crop(crop_dir / "left-large.jpg", 240, 200),
        _crop(crop_dir / "left-invalid.jpg", 100, 100),
        _crop(crop_dir / "right.jpg", 180, 150),
        _crop(crop_dir / "right-invalid.jpg", 180, 100),
    )
    record = store.create(tmp_path / "Preview_2024-01-01")
    sighting_id = record["sighting_id"]
    store.save_profiles(
        sighting_id,
        np.vstack(
            [
                np.full(TEAR_PROFILE_BINS, 0.1),
                np.full(TEAR_PROFILE_BINS, 0.2),
                np.full(TEAR_PROFILE_BINS, 0.3),
                np.full(TEAR_PROFILE_BINS, 0.4),
                np.full(TEAR_PROFILE_BINS, 0.5),
            ]
        ),
        ("left", "left", "left", "right", "right"),
        ("PhotoA", "PhotoA", "PhotoA", "PhotoB", "PhotoB"),
        crop_paths,
    )
    store.update(
        sighting_id,
        status="ready",
        photos=[
            {
                "file_name": "PhotoA.jpg",
                "photo_id": "PhotoA",
                "status": "analyzed",
                "detail": "3 usable ears",
                "photo_path": str(tmp_path / "PhotoA.jpg"),
                "ears": [],
            },
            {
                "file_name": "PhotoB.jpg",
                "photo_id": "PhotoB",
                "status": "analyzed",
                "detail": "2 usable ears",
                "photo_path": str(tmp_path / "PhotoB.jpg"),
                "ears": [],
            },
        ],
        profile_count=5,
        sides=["left", "right"],
    )
    return sighting_id


def _crop(path: Path, width: int, height: int) -> str:
    """Write a synthetic crop image and return its path."""
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return str(path)
