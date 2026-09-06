"""Tests for the CurvRank baseline catalog matcher."""

import math
from collections.abc import Mapping
from uuid import UUID

import numpy as np
import pytest
from numpy.typing import NDArray

from elephant_id.domain import Photo, SightingEarPair
from elephant_id.image.boxes import BoundingBox
from elephant_id.matching.curvrank import (
    CurvRankConfig,
    CurvRankMatcher,
    extract_descriptors,
)
from elephant_id.matching.protocol import CandidateKey
from elephant_id.preparation import EarSide, PreparedEar

_TEST_CONFIG = CurvRankConfig(
    curv_length=128,
    scales=(0.06, 0.10),
    num_keypoints=8,
    feat_dim=16,
    lnbnn_k=2,
)


def _uuid(value: int) -> UUID:
    """Return a deterministic UUIDv4-shaped value."""
    return UUID(f"00000000-0000-4000-8000-{value:012x}")


def _pair(value: int) -> SightingEarPair:
    """Return one neutral sighting ear pair."""
    sighting_id = _uuid(value)
    return SightingEarPair(
        sighting_id=sighting_id,
        left_photo=Photo(photo_id=_uuid(value + 1), sighting_id=sighting_id),
        right_photo=Photo(photo_id=_uuid(value + 2), sighting_id=sighting_id),
    )


def _jagged_contour(seed: int, *, amplitude: float = 10.0) -> NDArray[np.float64]:
    """Return a jagged top-to-bottom polyline with a seeded shape."""
    rng = np.random.default_rng(seed)
    num_points = 120
    y = np.linspace(0.0, 100.0, num_points)
    x = (
        40.0
        + amplitude * np.sin(np.linspace(0.0, 6.0 * np.pi, num_points))
        + rng.uniform(-3.0, 3.0, num_points)
    )
    return np.column_stack([x, y])


def _prepared_ear(
    contour: NDArray[np.float64],
    side: EarSide,
    photo: Photo,
) -> PreparedEar:
    """Return one synthetic prepared ear around the given contour."""
    anchors = (
        (float(contour[0, 0]), float(contour[0, 1])),
        (float(contour[-1, 0]), float(contour[-1, 1])),
    )
    return PreparedEar(
        source_photo=photo,
        source_box=BoundingBox(0, 0, 8, 8),
        contour=contour,
        original_landmarks=anchors,
        contour_anchors=anchors,
        inferred_side=side,
        cleaned_area=100.0,
    )


def _prepared_pair(
    pair: SightingEarPair,
    left_contour: NDArray[np.float64],
    right_contour: NDArray[np.float64],
) -> tuple[PreparedEar, PreparedEar]:
    """Return synthetic left and right prepared ears for one pair."""
    return (
        _prepared_ear(left_contour, "left", pair.left_photo),
        _prepared_ear(right_contour, "right", pair.right_photo),
    )


class RecordingPreparer:
    """Return controlled prepared ears while recording neutral pair inputs."""

    def __init__(
        self,
        prepared: Mapping[SightingEarPair, tuple[PreparedEar, PreparedEar]],
    ) -> None:
        """Initialize the preparer with one prepared-ear pair per input."""
        self._prepared = prepared
        self.calls: list[SightingEarPair] = []

    def prepare(self, pair: SightingEarPair) -> tuple[PreparedEar, PreparedEar]:
        """Record and return the prepared ears for `pair`."""
        self.calls.append(pair)
        return self._prepared[pair]


def _matcher(
    prepared: Mapping[SightingEarPair, tuple[PreparedEar, PreparedEar]],
) -> CurvRankMatcher:
    """Return a CurvRank matcher over a controlled recording preparer."""
    return CurvRankMatcher(
        prepare_ears=RecordingPreparer(prepared).prepare,
        config=_TEST_CONFIG,
    )


def _two_candidate_setup() -> tuple[
    CurvRankMatcher,
    SightingEarPair,
    dict[CandidateKey, tuple[SightingEarPair, ...]],
    CandidateKey,
    CandidateKey,
]:
    """Return a matcher with one same-contour and one different candidate."""
    query = _pair(10)
    same_evidence = _pair(20)
    different_evidence = _pair(30)
    query_left = _jagged_contour(1)
    query_right = _jagged_contour(2)
    prepared = {
        query: _prepared_pair(query, query_left, query_right),
        same_evidence: _prepared_pair(same_evidence, query_left, query_right),
        different_evidence: _prepared_pair(
            different_evidence,
            _jagged_contour(7, amplitude=30.0),
            _jagged_contour(8, amplitude=30.0),
        ),
    }
    same_key = CandidateKey(_uuid(500))
    different_key = CandidateKey(_uuid(501))
    catalog = {
        same_key: (same_evidence,),
        different_key: (different_evidence,),
    }
    return _matcher(prepared), query, catalog, same_key, different_key


def test_scores_cover_catalog_keys_with_finite_floats() -> None:
    """Every catalog key receives exactly one finite float score."""
    matcher, query, catalog, _, _ = _two_candidate_setup()

    scores = matcher.match(query, catalog)

    assert scores.keys() == catalog.keys()
    assert all(isinstance(score, float) for score in scores.values())
    assert all(math.isfinite(score) for score in scores.values())


def test_match_is_deterministic() -> None:
    """Two identical match calls produce identical scores."""
    matcher, query, catalog, _, _ = _two_candidate_setup()

    first = matcher.match(query, catalog)
    repeated = matcher.match(query, catalog)

    assert repeated == first


def test_same_contour_candidate_outscores_different_contour() -> None:
    """Evidence sharing the query contour beats a very different contour."""
    matcher, query, catalog, same_key, different_key = _two_candidate_setup()

    scores = matcher.match(query, catalog)

    assert scores[same_key] > scores[different_key]


def test_empty_catalog_returns_empty_mapping() -> None:
    """An empty catalog produces a complete empty score mapping."""
    query = _pair(10)
    prepared = {
        query: _prepared_pair(query, _jagged_contour(1), _jagged_contour(2)),
    }

    assert _matcher(prepared).match(query, {}) == {}


def test_empty_candidate_evidence_raises_runtime_error() -> None:
    """A listed candidate must contain at least one sighting ear pair."""
    query = _pair(10)
    prepared = {
        query: _prepared_pair(query, _jagged_contour(1), _jagged_contour(2)),
    }
    candidate_key = CandidateKey(_uuid(500))

    with pytest.raises(
        RuntimeError,
        match=f"{candidate_key} has no catalog evidence",
    ):
        _matcher(prepared).match(query, {candidate_key: ()})


def test_descriptor_extraction_yields_normalized_finite_descriptors() -> None:
    """Per-scale descriptors are finite unit-norm rows of `feat_dim` values."""
    ear = _prepared_ear(
        _jagged_contour(1),
        "left",
        Photo(photo_id=_uuid(1), sighting_id=_uuid(2)),
    )

    per_scale = extract_descriptors(ear, _TEST_CONFIG)

    assert len(per_scale) == len(_TEST_CONFIG.scales)
    assert any(len(descriptors) > 0 for descriptors in per_scale)
    for descriptors in per_scale:
        assert descriptors.ndim == 2
        assert descriptors.shape[1] == _TEST_CONFIG.feat_dim
        assert descriptors.dtype == np.float32
        assert np.isfinite(descriptors).all()
        if len(descriptors):
            norms = np.linalg.norm(descriptors, axis=1)
            assert norms == pytest.approx(np.ones(len(descriptors)), abs=1e-5)


@pytest.mark.parametrize(
    "overrides",
    [
        {"curv_length": 0},
        {"scales": ()},
        {"scales": (0.04, -0.06)},
        {"num_keypoints": 0},
        {"feat_dim": -1},
        {"lnbnn_k": 0},
    ],
)
def test_config_rejects_non_positive_parameters(overrides: dict[str, object]) -> None:
    """Every CurvRank parameter must be positive."""
    with pytest.raises(ValueError):
        CurvRankConfig(**overrides)  # type: ignore[arg-type]
