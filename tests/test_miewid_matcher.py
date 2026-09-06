"""Behavior tests for the MiewID embedding baseline matcher."""

import builtins
import math
from dataclasses import dataclass, field
from uuid import UUID

import cv2
import numpy as np
import pytest

from elephant_id.analysis import EarSide, PreparedEar
from elephant_id.domain import Photo, SightingEarPair
from elephant_id.image.boxes import BoundingBox
from elephant_id.matching.miewid import MiewIdEmbedder, MiewIdMatcher
from elephant_id.matching.protocol import CandidateKey


def _uuid(seed: int) -> UUID:
    """Return a deterministic UUIDv4 from a small integer seed."""
    return UUID(f"{seed:08x}-0000-4000-8000-000000000000")


def _pair(sighting_seed: int, left_seed: int, right_seed: int) -> SightingEarPair:
    """Build one sighting ear pair with deterministic identities."""
    sighting_id = _uuid(sighting_seed)
    return SightingEarPair(
        sighting_id,
        Photo(photo_id=_uuid(left_seed), sighting_id=sighting_id),
        Photo(photo_id=_uuid(right_seed), sighting_id=sighting_id),
    )


def _prepared_ear(photo: Photo, box: BoundingBox, side: EarSide) -> PreparedEar:
    """Build one small valid prepared ear for a photo and source box."""
    contour = np.asarray([[2.0, 3.0], [12.0, 8.0], [2.0, 15.0]])
    return PreparedEar(
        source_photo=photo,
        source_box=box,
        contour=contour,
        original_landmarks=((2.2, 3.1), (2.1, 15.2)),
        contour_anchors=((2.0, 3.0), (2.0, 15.0)),
        inferred_side=side,
        cleaned_area=80.0,
    )


def _prepared_pair(
    pair: SightingEarPair,
    left_box: BoundingBox,
    right_box: BoundingBox,
) -> tuple[PreparedEar, PreparedEar]:
    """Build the prepared left and right ears for one sighting ear pair."""
    return (
        _prepared_ear(pair.left_photo, left_box, "left"),
        _prepared_ear(pair.right_photo, right_box, "right"),
    )


@dataclass
class _FakePreparer:
    """Serve configured prepared-ear pairs by sighting ear pair."""

    by_pair: dict[SightingEarPair, tuple[PreparedEar, PreparedEar]]
    calls: list[SightingEarPair] = field(default_factory=list)

    def prepare(self, pair: SightingEarPair) -> tuple[PreparedEar, PreparedEar]:
        """Return the configured prepared ears for one pair."""
        self.calls.append(pair)
        return self.by_pair[pair]


@dataclass
class _FakePhotoStore:
    """Return one encoded synthetic image for every photo."""

    encoded: bytes

    def read(self, photo: Photo) -> bytes:
        """Return the shared encoded image bytes."""
        return self.encoded


@dataclass
class _ShapeEmbedder:
    """Return fixed unit vectors keyed by crop height and width."""

    vectors: dict[tuple[int, int], np.ndarray]
    producer_slug: str = "fake-embedder-v1"
    calls: list[tuple[int, ...]] = field(default_factory=list)

    def embed(self, crop: np.ndarray) -> np.ndarray:
        """Record the crop shape and return its configured vector."""
        self.calls.append(crop.shape)
        return self.vectors[(crop.shape[0], crop.shape[1])]


def _encoded_image() -> bytes:
    """Return one encoded 40-by-40 black image."""
    ok, encoded = cv2.imencode(".png", np.zeros((40, 40, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


QUERY = _pair(1, 11, 12)
CANDIDATE_A_PAIR = _pair(2, 21, 22)
CANDIDATE_B_PAIR_ONE = _pair(3, 31, 32)
CANDIDATE_B_PAIR_TWO = _pair(4, 41, 42)
CANDIDATE_A = CandidateKey(_uuid(100))
CANDIDATE_B = CandidateKey(_uuid(101))

QUERY_LEFT_BOX = BoundingBox(0, 0, 4, 3)
QUERY_RIGHT_BOX = BoundingBox(0, 0, 5, 3)
A_LEFT_BOX = BoundingBox(0, 0, 6, 3)
A_RIGHT_BOX = BoundingBox(0, 0, 7, 3)
B_ONE_LEFT_BOX = BoundingBox(0, 0, 8, 3)
B_ONE_RIGHT_BOX = BoundingBox(0, 0, 9, 3)
B_TWO_LEFT_BOX = BoundingBox(0, 0, 10, 3)
B_TWO_RIGHT_BOX = BoundingBox(0, 0, 11, 3)

_VECTORS = {
    (3, 4): np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    (3, 5): np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    (3, 6): np.asarray([0.8, 0.6, 0.0], dtype=np.float32),
    (3, 7): np.asarray([0.0, 0.5, math.sqrt(0.75)], dtype=np.float32),
    (3, 8): np.asarray([0.6, 0.8, 0.0], dtype=np.float32),
    (3, 9): np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    (3, 10): np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
    (3, 11): np.asarray([0.0, 0.9, math.sqrt(0.19)], dtype=np.float32),
}


def _build_matcher() -> tuple[MiewIdMatcher, _ShapeEmbedder]:
    """Build one matcher over the full synthetic query and catalog."""
    preparer = _FakePreparer(
        {
            QUERY: _prepared_pair(QUERY, QUERY_LEFT_BOX, QUERY_RIGHT_BOX),
            CANDIDATE_A_PAIR: _prepared_pair(
                CANDIDATE_A_PAIR, A_LEFT_BOX, A_RIGHT_BOX
            ),
            CANDIDATE_B_PAIR_ONE: _prepared_pair(
                CANDIDATE_B_PAIR_ONE, B_ONE_LEFT_BOX, B_ONE_RIGHT_BOX
            ),
            CANDIDATE_B_PAIR_TWO: _prepared_pair(
                CANDIDATE_B_PAIR_TWO, B_TWO_LEFT_BOX, B_TWO_RIGHT_BOX
            ),
        }
    )
    embedder = _ShapeEmbedder(_VECTORS)
    matcher = MiewIdMatcher(
        prepare_ears=preparer.prepare,
        photo_store=_FakePhotoStore(_encoded_image()),
        embedder=embedder,
    )
    return matcher, embedder


def _catalog() -> dict[CandidateKey, tuple[SightingEarPair, ...]]:
    """Return the two-candidate synthetic catalog."""
    return {
        CANDIDATE_A: (CANDIDATE_A_PAIR,),
        CANDIDATE_B: (CANDIDATE_B_PAIR_ONE, CANDIDATE_B_PAIR_TWO),
    }


def test_match_scores_every_candidate_with_finite_floats() -> None:
    """Scores cover exactly the catalog keys and stay finite."""
    matcher, _ = _build_matcher()

    scores = matcher.match(QUERY, _catalog())

    assert set(scores) == {CANDIDATE_A, CANDIDATE_B}
    assert all(math.isfinite(score) for score in scores.values())


def test_match_returns_empty_scores_for_empty_catalog() -> None:
    """An empty catalog produces an empty score mapping."""
    matcher, _ = _build_matcher()

    assert matcher.match(QUERY, {}) == {}


def test_match_rejects_candidate_without_evidence() -> None:
    """A candidate with an empty evidence tuple is a caller error."""
    matcher, _ = _build_matcher()

    with pytest.raises(RuntimeError, match="no catalog evidence"):
        matcher.match(QUERY, {CANDIDATE_A: ()})


def test_match_takes_side_maxima_then_means_the_sides() -> None:
    """Each side keeps its best cosine before averaging left and right."""
    matcher, _ = _build_matcher()

    scores = matcher.match(QUERY, _catalog())

    assert scores[CANDIDATE_A] == pytest.approx((0.8 + 0.5) / 2)
    assert scores[CANDIDATE_B] == pytest.approx((0.6 + 1.0) / 2)
    assert scores[CANDIDATE_B] > scores[CANDIDATE_A]


def test_match_memoizes_ear_embeddings_across_calls() -> None:
    """Repeating a match re-embeds no ear."""
    matcher, embedder = _build_matcher()

    matcher.match(QUERY, _catalog())
    first_call_count = len(embedder.calls)
    matcher.match(QUERY, _catalog())

    assert first_call_count == 8
    assert len(embedder.calls) == first_call_count


def test_match_crops_exactly_the_source_boxes() -> None:
    """The embedder receives crops sized by each ear's half-open box."""
    matcher, embedder = _build_matcher()

    matcher.match(QUERY, _catalog())

    expected = {(height, width, 3) for height, width in _VECTORS}
    assert set(embedder.calls) == expected


def test_match_rejects_out_of_image_ear_crop() -> None:
    """A source box outside the decoded image yields an empty-crop error."""
    pair = _pair(5, 51, 52)
    preparer = _FakePreparer(
        {pair: _prepared_pair(pair, BoundingBox(0, 50, 4, 60), QUERY_RIGHT_BOX)}
    )
    matcher = MiewIdMatcher(
        prepare_ears=preparer.prepare,
        photo_store=_FakePhotoStore(_encoded_image()),
        embedder=_ShapeEmbedder(_VECTORS),
    )

    with pytest.raises(ValueError, match="Empty ear crop"):
        matcher.match(pair, {})


def test_miewid_embedder_constructor_loads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the real embedder imports neither torch nor transformers."""
    real_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch" or name == "transformers" or name.startswith(
            ("torch.", "transformers.")
        ):
            raise AssertionError(f"Constructor imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    embedder = MiewIdEmbedder(device="cpu")

    assert embedder.producer_slug == "miewid-msv2-cosine-v1"


def test_miewid_pins_pretrained_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """A revision controls model loading and distinguishes the producer."""
    import sys
    from types import SimpleNamespace
    from typing import Self

    calls: list[tuple[str, dict[str, object]]] = []

    class Model:
        """Provide only the placement and evaluation operations used by loading."""

        def to(self, device: str) -> Self:
            """Accept the explicit CPU device without allocating tensors."""
            assert device == "cpu"
            return self

        def eval(self) -> Self:
            """Return this already fake evaluation model."""
            return self

    model = Model()

    def from_pretrained(model_id: str, **kwargs: object) -> Model:
        """Record the exact model source requested by the embedder."""
        calls.append((model_id, kwargs))
        return model

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=from_pretrained)),
    )
    embedder = MiewIdEmbedder("example/model", device="cpu", revision="fixed-revision")

    assert embedder._loaded_model() is model
    assert embedder._loaded_model() is model
    assert calls == [
        ("example/model", {"revision": "fixed-revision", "trust_remote_code": True})
    ]
    assert embedder.producer_slug.endswith("-fixed-revision")
