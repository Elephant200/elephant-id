"""Tests for the CurvRank v2 catalog matcher and its fidelity to the plugin."""

import importlib
import math
import sys
import types
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import numpy as np
import pytest
from numpy.typing import NDArray

from elephant_id.domain import Photo, SightingEarPair
from elephant_id.image.boxes import BoundingBox
from elephant_id.matching import CandidateKey, MatchingError, curvrank
from elephant_id.preparation import EarSide, PreparedEar

pytestmark = pytest.mark.filterwarnings("ignore:`trapz` is deprecated:DeprecationWarning")

_REFERENCE = Path(__file__).resolve().parents[1] / "curvrank_ref"
needs_reference = pytest.mark.skipif(
    not (_REFERENCE / "functional.py").exists(),
    reason="private copy of the wbia-tpl-curvrank-v2 plugin is absent",
)


class _ExactAnnoyIndex:
    """Exact stand-in for `annoy.AnnoyIndex` so reference LNBNN is deterministic."""

    _saved: ClassVar[dict[str, NDArray[np.float32]]] = {}

    def __init__(self, dimension: int, metric: str) -> None:
        """Start an empty Euclidean index."""
        assert metric == "euclidean"
        self._items: dict[int, NDArray[np.float32]] = {}

    def add_item(self, index: int, vector: NDArray[np.float32]) -> None:
        """Store one vector."""
        self._items[index] = np.asarray(vector)

    def build(self, num_trees: int) -> None:
        """Accept the reference build call."""

    def save(self, path: str) -> None:
        """Keep the vectors under a fake path."""
        self._saved[path] = np.vstack([self._items[i] for i in range(len(self._items))])

    def load(self, path: str) -> None:
        """Restore vectors saved under a fake path."""
        self._data = self._saved[path]

    def get_nns_by_vector(
        self, vector: NDArray[np.float32], count: int, search_k: int, include_distances: bool
    ) -> tuple[list[int], list[float]]:
        """Return exact nearest neighbors and Euclidean distances."""
        distances = np.linalg.norm(self._data - vector, axis=1)
        order = np.argsort(distances, kind="stable")[:count]
        return order.tolist(), distances[order].tolist()


@pytest.fixture(scope="module")
def reference() -> types.ModuleType:
    """Import the verbatim plugin `functional` module with training imports stubbed."""
    stubs = {
        name: types.ModuleType(name)
        for name in ("wbia_curvrank_v2", "wbia_curvrank_v2.costs", "annoy", "pyastar2d", "tqdm")
    }
    stubs["wbia_curvrank_v2.costs"].exp_cost_func = None
    stubs["wbia_curvrank_v2.costs"].hyp_cost_func = None
    stubs["annoy"].AnnoyIndex = _ExactAnnoyIndex
    stubs["tqdm"].tqdm = lambda items: items
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.get(name) for name in (*stubs, "curv", "utils")}
    sys.path.insert(0, str(_REFERENCE))
    try:
        sys.modules.update(stubs)
        package = stubs["wbia_curvrank_v2"]
        package.algo = types.ModuleType("algo")
        package.fcnn = types.ModuleType("fcnn")
        package.curv = importlib.import_module("curv")
        package.utils = importlib.import_module("utils")
        sys.modules["wbia_curvrank_v2.algo"] = package.algo
        sys.modules["wbia_curvrank_v2.fcnn"] = package.fcnn
        spec = importlib.util.spec_from_file_location(
            "curvrank_reference_functional", _REFERENCE / "functional.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.pop("wbia_curvrank_v2.algo", None)
        sys.modules.pop("wbia_curvrank_v2.fcnn", None)


def _uuid(value: int) -> UUID:
    """Return a deterministic UUIDv4-shaped value."""
    return UUID(f"00000000-0000-4000-8000-{value:012x}")


def _outline(seed: int, *, points: int = 400) -> NDArray[np.float64]:
    """Return a notched ear-like outline from the upper anchor to the lower anchor."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, np.pi, points)
    x = 120.0 + 260.0 * np.sin(t) + np.cumsum(rng.normal(0.0, 0.8, points))
    y = 60.0 + 300.0 * (1.0 - np.cos(t))
    for notch in rng.integers(40, points - 40, 5):
        x[notch - 4 : notch + 5] -= np.hanning(9) * rng.uniform(6.0, 14.0)
    return np.column_stack([x, y])


def _ear(
    contour: NDArray[np.float64],
    side: EarSide = "left",
    *,
    photo: Photo | None = None,
    source_box: BoundingBox | None = None,
    image_size: tuple[int, int] = (800, 900),
) -> PreparedEar:
    """Return one prepared ear around a contour."""
    box = source_box or BoundingBox(
        int(contour[:, 0].min()) - 40,
        int(contour[:, 1].min()) - 10,
        int(contour[:, 0].max()) + 5,
        int(contour[:, 1].max()) + 10,
    )
    anchors = (tuple(contour[0]), tuple(contour[-1]))
    return PreparedEar(
        source_photo=photo or Photo(photo_id=_uuid(1), sighting_id=_uuid(2)),
        source_box=box,
        image_size=image_size,
        contour=contour,
        original_landmarks=anchors,
        contour_anchors=anchors,
        inferred_side=side,
        cleaned_area=1000.0,
    )


def test_padded_crop_pads_by_integer_tenths_and_clips_to_image() -> None:
    """The crop reproduces the plugin's `crop_with_padding`."""
    ear = _ear(_outline(0), source_box=BoundingBox(55, 5, 350, 505), image_size=(370, 900))

    assert curvrank.padded_crop(ear) == (26, 0, 370, 555)


def test_fine_contour_is_an_upper_to_lower_minimal_eight_connected_row_column_path() -> None:
    """The fine contour has the shape of the plugin's diagonal A* path."""
    ear = _ear(_outline(1))
    path = curvrank.fine_contour(ear)
    steps = np.abs(np.diff(path, axis=0))
    skips = np.abs(path[2:] - path[:-2]).max(axis=1)

    assert path.dtype == np.int64
    assert steps.max() == 1 and (steps.sum(axis=1) > 0).all()
    assert (skips > 1).all()
    assert path[0, 0] < path[-1, 0]
    x0, _, x1, _ = curvrank.padded_crop(ear)
    assert path[:, 1].max() <= curvrank.FINE_WIDTH * (ear.contour[:, 0].max() - x0 + 1) / (x1 - x0)


def test_right_ear_is_mirrored_into_left_ear_geometry() -> None:
    """A mirrored right ear yields the same fine contour as its left ear."""
    left = _ear(_outline(2))
    x0, _, x1, _ = curvrank.padded_crop(left)
    mirrored = left.contour.copy()
    mirrored[:, 0] = (x0 + x1 - 1) - mirrored[:, 0]
    box = left.source_box
    right = _ear(
        mirrored,
        "right",
        source_box=BoundingBox(x0 + x1 - box.x2, box.y1, x0 + x1 - box.x1, box.y2),
    )

    np.testing.assert_array_equal(curvrank.fine_contour(right), curvrank.fine_contour(left))


@needs_reference
def test_curvature_matches_reference(reference: types.ModuleType) -> None:
    """Integral curvature equals the plugin's `oriented_curvature`."""
    contour = curvrank.fine_contour(_ear(_outline(3)))
    radii = curvrank.SCALES * 1024

    np.testing.assert_array_equal(
        curvrank.oriented_curvature(contour, radii),
        reference.curv.oriented_curvature(contour, radii),
    )


@needs_reference
@pytest.mark.parametrize("seed", range(6))
def test_descriptors_match_reference(reference: types.ModuleType, seed: int) -> None:
    """Per-scale descriptors equal the plugin's `curvature_descriptors`."""
    contour = curvrank.fine_contour(_ear(_outline(seed)))
    curvature = reference.curv.oriented_curvature(contour, curvrank.SCALES * 1024)
    success, expected = reference.curvature_descriptors(contour, curvature, curvrank.SCALES, 1024, 32, 32)
    actual = curvrank.curvature_descriptors(curvature)

    assert success
    for scale, descriptors in zip(curvrank.SCALES, actual, strict=True):
        assert descriptors.dtype == np.float32
        np.testing.assert_allclose(descriptors, expected[scale], rtol=0, atol=1e-6)


@needs_reference
def test_lnbnn_matches_reference(reference: types.ModuleType) -> None:
    """LNBNN margins equal the plugin's `lnbnn_identify` with exact search."""
    rng = np.random.default_rng(4)
    catalog = rng.normal(size=(60, 8)).astype(np.float32)
    catalog /= np.linalg.norm(catalog, axis=1, keepdims=True)
    query = catalog[:10] + rng.normal(0.0, 0.05, (10, 8)).astype(np.float32)
    labels = [CandidateKey(_uuid(i % 7)) for i in range(60)]
    reference.build_lnbnn_index(catalog, "fake.ann")
    expected = reference.lnbnn_identify("fake.ann", 2, query, labels)

    actual = curvrank.lnbnn_scores(query, catalog, labels)

    for label, score in expected.items():
        assert actual.get(label, 0.0) == pytest.approx(score, abs=1e-6)


def _pair(value: int) -> SightingEarPair:
    """Return one neutral sighting ear pair."""
    sighting_id = _uuid(value)
    return SightingEarPair(
        sighting_id=sighting_id,
        left_photo=Photo(photo_id=_uuid(value + 1), sighting_id=sighting_id),
        right_photo=Photo(photo_id=_uuid(value + 2), sighting_id=sighting_id),
    )


def _prepare(pair: SightingEarPair) -> tuple[PreparedEar, PreparedEar]:
    """Return distinct deterministic left and right ears for each sighting."""
    seed = int(pair.sighting_id) & 0xFFFF
    return (
        _ear(_outline(seed, points=200), "left", photo=pair.left_photo),
        _ear(_outline(seed + 1, points=200), "right", photo=pair.right_photo),
    )


def test_matcher_ranks_the_same_ears_first_and_scores_every_candidate() -> None:
    """A query whose ears reappear in the catalog wins with finite scores."""
    matcher = curvrank.CurvRank(prepare_ears=_prepare)
    catalog = {CandidateKey(_uuid(9000 + value)): (_pair(value),) for value in (100, 200, 300)}

    scores = matcher.match(_pair(200), catalog)

    assert set(scores) == set(catalog)
    assert all(math.isfinite(score) for score in scores.values())
    assert max(scores, key=scores.__getitem__) == CandidateKey(_uuid(9200))


def test_description_failure_raises_matching_error_with_photo_and_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A description failure names the photo and declared side of the ear."""

    def fail_on_right(ear: PreparedEar) -> curvrank.Descriptors:
        if ear.inferred_side == "right":
            raise ValueError("bad outline")
        return ()

    monkeypatch.setattr(curvrank, "extract_descriptors", fail_on_right)
    query = _pair(100)

    with pytest.raises(MatchingError, match="bad outline") as caught:
        curvrank.CurvRank(prepare_ears=_prepare).describe(query)

    assert caught.value.photo_id == query.right_photo.photo_id
    assert caught.value.side == "right"
    assert caught.value.stage == "curvrank description"
