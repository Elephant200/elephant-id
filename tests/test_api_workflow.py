"""Tests for :class:`elephant_id.api.workflow.SightingWorkflow`."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from elephant_id.api.gallery import GalleryData
from elephant_id.api.store import SightingStore
from elephant_id.api.workflow import (
    SightingWorkflow,
    WorkflowConflict,
    WorkflowInvalid,
)
from elephant_id.constants import TEAR_PROFILE_BINS


class FakeCandidate:
    """JSON-serializable fake match result."""

    def __init__(self, identity: str, query_photo_id: str) -> None:
        """Store minimal candidate evidence."""
        self.identity = identity
        self.query_photo_id = query_photo_id

    def to_dict(self) -> dict:
        """Return a fake ranked candidate."""
        return {"identity": self.identity, "score": 1.5, "confidence": 0.82}


class FakeEngine:
    """Fake matching engine recording rank and extend calls."""

    elephant_count = 1
    profile_count = 2

    def __init__(self) -> None:
        """Initialize recorded calls."""
        self.rank_calls: list[tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]] = []
        self.extend_calls: list[
            tuple[np.ndarray, tuple[str, ...], str, str, tuple[str, ...]]
        ] = []

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
        self.extend_calls.append(
            (profiles.copy(), tuple(sides), identity, date, tuple(photo_ids))
        )


def _crop(path: Path, width: int, height: int) -> str:
    """Write a synthetic crop image and return its path."""
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(path), image)
    return str(path)


def _make_workflow(tmp_path: Path) -> tuple[SightingWorkflow, SightingStore]:
    """Return a workflow wired to a real temp store and an empty gallery."""
    store = SightingStore(tmp_path)
    gallery = GalleryData(
        profiles=np.zeros((0, TEAR_PROFILE_BINS)),
        photo_ids=(),
        identities=(),
        sides=(),
        dates=(),
        crop_paths=(),
    )
    workflow = SightingWorkflow(store, gallery, tmp_path)
    return workflow, store


def _ready_sighting(store: SightingStore, tmp_path: Path) -> str:
    """Create a ready sighting with one valid left and one valid right crop."""
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir(exist_ok=True)
    crop_paths = (
        _crop(crop_dir / "left.jpg", 120, 100),
        _crop(crop_dir / "right.jpg", 180, 150),
    )
    record = store.create(tmp_path / "Sighting_2024-01-01")
    sighting_id = record["sighting_id"]
    store.save_profiles(
        sighting_id,
        np.vstack(
            [
                np.full(TEAR_PROFILE_BINS, 0.2),
                np.full(TEAR_PROFILE_BINS, 0.4),
            ]
        ),
        ("left", "right"),
        ("PhotoA", "PhotoB"),
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
                "detail": "1 usable ear",
                "date": "2024-01-01",
                "photo_path": str(tmp_path / "PhotoA.jpg"),
                "ears": [],
            },
            {
                "file_name": "PhotoB.jpg",
                "photo_id": "PhotoB",
                "status": "analyzed",
                "detail": "1 usable ear",
                "date": "2024-01-01",
                "photo_path": str(tmp_path / "PhotoB.jpg"),
                "ears": [],
            },
        ],
        profile_count=2,
        sides=["left", "right"],
    )
    return sighting_id


def _approve(
    workflow: SightingWorkflow, store: SightingStore, sighting_id: str
) -> dict:
    """Approve the single left and right candidate for a ready sighting."""
    package = workflow.analysis_package(sighting_id)
    left = package["ear_candidates"]["left"][0]["candidate_id"]
    right = package["ear_candidates"]["right"][0]["candidate_id"]
    return workflow.approve_evidence(sighting_id, left, right)


def _raising_supplier() -> FakeEngine:
    """Fail the test if the engine supplier is invoked."""
    raise AssertionError("engine_supplier must not be called")


# --- approve_evidence -------------------------------------------------------


def test_approve_evidence_happy_path(tmp_path: Path) -> None:
    """Approving valid left and right candidates records approved_evidence."""
    workflow, store = _make_workflow(tmp_path)
    sighting_id = _ready_sighting(store, tmp_path)

    record = _approve(workflow, store, sighting_id)

    assert record["approved_evidence"]["left"]["side"] == "left"
    assert record["approved_evidence"]["right"]["side"] == "right"
    assert record["approved_evidence"]["left"]["corrected"] is False


def test_approve_evidence_not_ready_raises_conflict(tmp_path: Path) -> None:
    """Approving evidence on a non-ready sighting raises WorkflowConflict."""
    workflow, store = _make_workflow(tmp_path)
    record = store.create(tmp_path / "Sighting_2024-01-01")
    sighting_id = record["sighting_id"]

    with pytest.raises(WorkflowConflict):
        workflow.approve_evidence(sighting_id, "left-id", "right-id")


def test_approve_evidence_missing_profiles_raises_conflict(tmp_path: Path) -> None:
    """A ready sighting with no stored profiles raises WorkflowConflict."""
    workflow, store = _make_workflow(tmp_path)
    record = store.create(tmp_path / "Sighting_2024-01-01")
    sighting_id = record["sighting_id"]
    store.update(sighting_id, status="ready")

    with pytest.raises(WorkflowConflict):
        workflow.approve_evidence(sighting_id, "left-id", "right-id")


def test_approve_evidence_invalid_candidate_ids_raise_invalid(tmp_path: Path) -> None:
    """Unknown candidate ids raise WorkflowInvalid."""
    workflow, store = _make_workflow(tmp_path)
    sighting_id = _ready_sighting(store, tmp_path)

    with pytest.raises(WorkflowInvalid):
        workflow.approve_evidence(sighting_id, "bogus-left", "bogus-right")


def test_approve_evidence_same_side_candidates_raise_invalid(tmp_path: Path) -> None:
    """Selecting the same-side candidate for both sides raises WorkflowInvalid."""
    workflow, store = _make_workflow(tmp_path)
    sighting_id = _ready_sighting(store, tmp_path)
    package = workflow.analysis_package(sighting_id)
    left = package["ear_candidates"]["left"][0]["candidate_id"]

    with pytest.raises(WorkflowInvalid):
        workflow.approve_evidence(sighting_id, left, left)


def test_approve_evidence_resets_match_and_decision(tmp_path: Path) -> None:
    """Re-approving evidence clears any prior match and decision."""
    workflow, store = _make_workflow(tmp_path)
    sighting_id = _ready_sighting(store, tmp_path)
    store.update(
        sighting_id,
        match={"matched_at": "x", "candidates": []},
        decision={"action": "unresolved", "elephant_name": None, "decided_at": "x"},
    )

    record = _approve(workflow, store, sighting_id)

    assert record["match"] is None
    assert record["decision"] is None


# --- match -------------------------------------------------------------


def test_match_happy_path_uses_only_approved_rows(tmp_path: Path) -> None:
    """Matching calls engine.rank with exactly the two approved profile rows."""
    workflow, store = _make_workflow(tmp_path)
    engine = FakeEngine()
    sighting_id = _ready_sighting(store, tmp_path)
    _approve(workflow, store, sighting_id)

    record = workflow.match(sighting_id, engine, top_n=4)

    assert record["match"]["candidates"][0]["identity"] == "Known"
    assert len(engine.rank_calls) == 1
    profiles, sides, photo_ids = engine.rank_calls[0]
    assert profiles.shape == (2, TEAR_PROFILE_BINS)
    assert sides == ("left", "right")
    assert photo_ids == ("PhotoA", "PhotoB")


def test_match_unapproved_sighting_raises_conflict(tmp_path: Path) -> None:
    """Matching before evidence approval raises WorkflowConflict."""
    workflow, store = _make_workflow(tmp_path)
    engine = FakeEngine()
    sighting_id = _ready_sighting(store, tmp_path)

    with pytest.raises(WorkflowConflict):
        workflow.match(sighting_id, engine, top_n=4)


# --- decide --------------------------------------------------------------


def test_decide_unresolved_does_not_call_engine_supplier(tmp_path: Path) -> None:
    """Unresolved decisions must not touch the engine supplier at all."""
    workflow, store = _make_workflow(tmp_path)
    sighting_id = _ready_sighting(store, tmp_path)

    record = workflow.decide(sighting_id, _raising_supplier, "unresolved", None)

    assert record["decision"]["action"] == "unresolved"
    assert record["decision"]["elephant_name"] is None


def test_decide_existing_known_elephant_unknown_name_raises_invalid(
    tmp_path: Path,
) -> None:
    """existing_known_elephant with an unrecognized name raises WorkflowInvalid."""
    workflow, store = _make_workflow(tmp_path)
    engine = FakeEngine()
    sighting_id = _ready_sighting(store, tmp_path)
    _approve(workflow, store, sighting_id)

    with pytest.raises(WorkflowInvalid):
        workflow.decide(
            sighting_id, lambda: engine, "existing_known_elephant", "Stranger"
        )


def test_decide_new_known_elephant_existing_name_raises_invalid(
    tmp_path: Path,
) -> None:
    """new_known_elephant with an already-known name raises WorkflowInvalid."""
    workflow, store = _make_workflow(tmp_path)
    engine = FakeEngine()
    sighting_id = _ready_sighting(store, tmp_path)
    _approve(workflow, store, sighting_id)

    with pytest.raises(WorkflowInvalid):
        workflow.decide(sighting_id, lambda: engine, "new_known_elephant", "Known")


def test_decide_missing_name_raises_invalid(tmp_path: Path) -> None:
    """A missing elephant_name for a naming action raises WorkflowInvalid."""
    workflow, store = _make_workflow(tmp_path)
    engine = FakeEngine()
    sighting_id = _ready_sighting(store, tmp_path)
    _approve(workflow, store, sighting_id)

    with pytest.raises(WorkflowInvalid):
        workflow.decide(sighting_id, lambda: engine, "new_known_elephant", "  ")


def test_decide_already_decided_raises_conflict(tmp_path: Path) -> None:
    """Deciding a sighting a second time raises WorkflowConflict."""
    workflow, store = _make_workflow(tmp_path)
    sighting_id = _ready_sighting(store, tmp_path)
    workflow.decide(sighting_id, _raising_supplier, "unresolved", None)

    with pytest.raises(WorkflowConflict):
        workflow.decide(sighting_id, _raising_supplier, "unresolved", None)


def test_decide_success_calls_engine_extend_with_approved_rows(tmp_path: Path) -> None:
    """A successful naming decision extends the engine with the approved rows."""
    workflow, store = _make_workflow(tmp_path)
    engine = FakeEngine()
    sighting_id = _ready_sighting(store, tmp_path)
    _approve(workflow, store, sighting_id)

    record = workflow.decide(
        sighting_id, lambda: engine, "new_known_elephant", "Newbie"
    )

    assert record["decision"]["action"] == "new_known_elephant"
    assert record["decision"]["elephant_name"] == "Newbie"
    assert len(engine.extend_calls) == 1
    profiles, sides, identity, date, photo_ids = engine.extend_calls[0]
    assert profiles.shape == (2, TEAR_PROFILE_BINS)
    assert sides == ("left", "right")
    assert identity == "Newbie"
    assert date == "2024-01-01"
    assert photo_ids == ("PhotoA", "PhotoB")


# --- refile_decisions ------------------------------------------------------


def test_refile_decisions_extends_engine_for_decided_sighting(tmp_path: Path) -> None:
    """Refiling calls engine.extend for a decided, approved sighting."""
    workflow, store = _make_workflow(tmp_path)
    engine = FakeEngine()
    sighting_id = _ready_sighting(store, tmp_path)
    _approve(workflow, store, sighting_id)
    workflow.decide(sighting_id, lambda: engine, "new_known_elephant", "Newbie")
    engine.extend_calls.clear()

    workflow.refile_decisions(engine)

    assert len(engine.extend_calls) == 1
    assert engine.extend_calls[0][2] == "Newbie"
    assert engine.extend_calls[0][1] == ("left", "right")


def test_refile_decisions_without_approved_evidence_uses_raw_profiles(
    tmp_path: Path,
) -> None:
    """A decided sighting with no approved_evidence still refiles, using raw rows."""
    workflow, store = _make_workflow(tmp_path)
    engine = FakeEngine()
    sighting_id = _ready_sighting(store, tmp_path)
    # Decide directly without approving evidence first (bypassing approve_evidence,
    # which the API layer would normally enforce before allowing a decision).
    store.update(
        sighting_id,
        decision={
            "action": "new_known_elephant",
            "elephant_name": "Rawhide",
            "decided_at": "2024-02-02T00:00:00+00:00",
        },
    )

    workflow.refile_decisions(engine)

    assert len(engine.extend_calls) == 1
    profiles, sides, identity, _date, photo_ids = engine.extend_calls[0]
    assert identity == "Rawhide"
    assert profiles.shape == (2, TEAR_PROFILE_BINS)
    assert sides == ("left", "right")
    assert photo_ids == ("PhotoA", "PhotoB")


def test_refile_decisions_skips_failing_record_with_warning(
    tmp_path: Path,
) -> None:
    """A record whose profiles cannot load is skipped, not raised."""
    workflow, store = _make_workflow(tmp_path)
    engine = FakeEngine()
    good_sighting_id = _ready_sighting(store, tmp_path)
    _approve(workflow, store, good_sighting_id)
    workflow.decide(good_sighting_id, lambda: engine, "new_known_elephant", "Newbie")
    engine.extend_calls.clear()

    broken_record = store.create(tmp_path / "Broken_2024-03-01")
    broken_id = broken_record["sighting_id"]
    store.update(
        broken_id,
        status="ready",
        decision={
            "action": "existing_known_elephant",
            "elephant_name": "Known",
            "decided_at": "2024-03-01T00:00:00+00:00",
        },
    )

    workflow.refile_decisions(engine)

    assert len(engine.extend_calls) == 1
    assert engine.extend_calls[0][2] == "Newbie"
