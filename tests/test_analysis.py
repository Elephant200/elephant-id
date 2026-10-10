"""Behavior tests for sighting analysis and AlphaTear values."""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
import pytest
import shapely
from pycocotools import mask as coco_mask

from elephant_id.cache import CacheManager
from elephant_id.composition import (
    build_profile_tuning_alphaphant,
    build_standard_alphaphant,
)
from elephant_id.domain import Photo, SightingEarPair
from elephant_id.image.boxes import BoundingBox
from elephant_id.inference import Detection
from elephant_id.matching.alphaphant import AlphaPhant
from elephant_id.matching.alphaphant.cached import CachedTearProfileExtractor
from elephant_id.matching.alphaphant.extraction import (
    DEFAULT_VERSION,
    AlphaTearConfig,
    AlphaTearExtractor,
    AnalyzedEar,
    AnalyzedSightingEarPair,
    TearProfile,
    _alpha_shape_crossings,
)
from elephant_id.matching.alphaphant.similarity import EarComparator
from elephant_id.matching.protocol import MatchingError
from elephant_id.preparation import (
    EarSide,
    PreparationStage,
    PreparedEar,
    SightingPreparer,
    ear_side_path,
    prepare_ear,
)

PHOTO = Photo(
    photo_id=UUID("8c47c36d-a75d-4ee4-a58a-3a08fca2c833"),
    sighting_id=UUID("c44ac5bd-eb07-493d-b296-f69f3844e463"),
)
RIGHT_PHOTO = Photo(
    photo_id=UUID("b2fc7853-37f3-4474-bacf-090749af62cb"),
    sighting_id=PHOTO.sighting_id,
)


def test_tear_profile_owns_immutable_one_dimensional_depths() -> None:
    """A profile copies mutable caller data and exposes normalized depths only."""
    source = np.asarray([0.0, 0.1, -0.02])

    profile = TearProfile(source, start_degrees=5.0, end_degrees=175.0)
    source[1] = 0.9

    np.testing.assert_array_equal(profile.depths, [0.0, 0.1, -0.02])
    with pytest.raises(ValueError, match="read-only"):
        profile.depths[0] = 1.0


@pytest.mark.parametrize(
    ("start_degrees", "end_degrees"),
    [(-1.0, 175.0), (5.0, 181.0), (90.0, 90.0), (100.0, 80.0)],
)
def test_tear_profile_rejects_measured_interval_outside_0_to_180(
    start_degrees: float, end_degrees: float
) -> None:
    """The measured interval is a nonempty part of 0° to 180°."""
    with pytest.raises(ValueError, match="measured interval"):
        TearProfile(np.zeros(3), start_degrees=start_degrees, end_degrees=end_degrees)


def _prepared_ear(side: EarSide = "left") -> PreparedEar:
    """Build one small immutable prepared ear."""
    contour = np.asarray([[2.0, 3.0], [12.0, 8.0], [2.0, 15.0]])
    return PreparedEar(
        source_photo=PHOTO,
        source_box=BoundingBox(2, 3, 13, 16),
        contour=contour,
        original_landmarks=((2.2, 3.1), (2.1, 15.2)),
        contour_anchors=((2.0, 3.0), (2.0, 15.0)),
        inferred_side=side,
        cleaned_area=80.0,
    )


def _notched_ear() -> PreparedEar:
    """Build a left ear flap with one narrow and one wide notch."""
    theta = np.linspace(0, np.pi, 181)
    radius = 80 - 12 * np.exp(-(((theta - 0.9) / 0.09) ** 2))
    radius -= 15 * np.exp(-(((theta - 2.1) / 0.4) ** 2))
    contour = np.column_stack((100 + radius * np.sin(theta), 100 - radius * np.cos(theta)))
    contour[0], contour[-1] = (100.0, 20.0), (100.0, 180.0)
    return PreparedEar(
        source_photo=PHOTO,
        source_box=BoundingBox(10, 10, 190, 190),
        contour=contour,
        original_landmarks=((100.0, 20.0), (100.0, 180.0)),
        contour_anchors=((100.0, 20.0), (100.0, 180.0)),
        inferred_side="left",
        cleaned_area=10000.0,
    )


def test_alpha_tear_returns_one_profile_per_alpha_scale_in_order() -> None:
    """Each tear profile equals the extraction that uses only its alpha scale."""
    ear = _notched_ear()

    small, large = AlphaTearExtractor(AlphaTearConfig(alpha_scales=(0.35, 2.0))).extract(ear)

    (small_only,) = AlphaTearExtractor(AlphaTearConfig(alpha_scales=(0.35,))).extract(ear)
    (large_only,) = AlphaTearExtractor(AlphaTearConfig(alpha_scales=(2.0,))).extract(ear)
    np.testing.assert_array_equal(small.depths, small_only.depths)
    np.testing.assert_array_equal(large.depths, large_only.depths)
    assert not np.array_equal(small.depths, large.depths)


def test_alpha_tear_profiles_state_their_measured_interval() -> None:
    """Each tear profile is measured inside 5° to 175° and is zero outside."""
    profiles = AlphaTearExtractor(AlphaTearConfig(alpha_scales=(0.35, 2.0))).extract(_notched_ear())

    for profile in profiles:
        angles = np.linspace(0.0, 180.0, len(profile.depths))
        outside = (angles <= profile.start_degrees) | (angles >= profile.end_degrees)
        assert (profile.start_degrees, profile.end_degrees) == (5.0, 175.0)
        assert len(profile.depths) == 720
        assert np.all(profile.depths[outside] == 0.0)
        assert np.any(profile.depths[~outside] != 0.0)


@pytest.mark.parametrize("reverse", [False, True])
def test_alpha_shape_crossings_use_the_inward_normal_of_the_crossed_edge(reverse: bool) -> None:
    """Each normal is perpendicular to the crossed edge and points inward, in both windings."""
    boundary = np.asarray([(-2.0, -1.0), (2.0, -1.0), (2.0, 1.0), (-2.0, 1.0)])
    alpha_shape = shapely.Polygon(boundary[::-1] if reverse else boundary)
    directions = np.asarray([(1.0, 0.0), (2.0, 0.9), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    crossings, normals = _alpha_shape_crossings(np.zeros(2), directions, alpha_shape)

    np.testing.assert_allclose(
        crossings, [(2.0, 0.0), (2.0, 0.9), (0.0, 1.0), (-2.0, 0.0), (0.0, -1.0)], atol=1e-12
    )
    np.testing.assert_allclose(
        normals, [(-1.0, 0.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0)], atol=1e-12
    )


def test_alpha_shape_crossings_are_nan_for_a_ray_that_does_not_cross() -> None:
    """A ray that starts outside the alpha shape and points away has no crossing and no normal."""
    alpha_shape = shapely.Polygon([(1.0, -1.0), (3.0, -1.0), (3.0, 1.0), (1.0, 1.0)])

    crossings, normals = _alpha_shape_crossings(
        np.zeros(2), np.asarray([(-1.0, 0.0), (1.0, 0.0)]), alpha_shape
    )

    assert np.isnan(crossings[0]).all() and np.isnan(normals[0]).all()
    np.testing.assert_allclose(crossings[1], (3.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(normals[1], (-1.0, 0.0), atol=1e-12)


def test_settled_alpha_tear_uses_three_alpha_scales() -> None:
    """The settled extraction measures small, medium, and large tears."""
    assert DEFAULT_VERSION.config.alpha_scales == (0.35, 2.0, 10.0)


@pytest.mark.parametrize("alpha_scales", [(), (0.35, 0.0), (-1.0,)])
def test_alpha_tear_config_rejects_missing_or_nonpositive_alpha_scales(
    alpha_scales: tuple[float, ...],
) -> None:
    """Extraction needs at least one positive alpha scale."""
    with pytest.raises(ValueError, match="alpha_scales"):
        AlphaTearConfig(alpha_scales=alpha_scales)


@dataclass
class _ProfileExtractor:
    """Record prepared-ear extraction."""

    profiles: tuple[TearProfile, ...] = (
        TearProfile(np.asarray([0.0, 0.1, 0.0]), start_degrees=5.0, end_degrees=175.0),
        TearProfile(np.asarray([0.0, 0.3, -0.1]), start_degrees=10.0, end_degrees=170.0),
    )
    alpha_scales: tuple[float, ...] = (0.35, 2.0)
    producer_slug: str | None = "alpha-tear-v1"
    calls: list[PreparedEar] = field(default_factory=list)

    def extract(self, ear: PreparedEar) -> tuple[TearProfile, ...]:
        """Record and return the configured profiles."""
        self.calls.append(ear)
        return self.profiles


def _cached(inner: _ProfileExtractor, cache: CacheManager) -> CachedTearProfileExtractor:
    """Wrap an extractor with fixed inference producer slugs."""
    return CachedTearProfileExtractor(
        inner,
        cache,
        segmentation_producer_slug="sam3-features",
        landmark_producer_slug="yolo26n-keypoints-v1",
    )


_CACHE_KEY = (
    f"{PHOTO.photo_id}__seg_sam3-features__landmarks_"
    "yolo26n-keypoints-v1__side_left__crop_2_3_13_16"
)


def test_cached_profile_uses_prepared_ear_lineage(tmp_path: Path) -> None:
    """Profile persistence keys compact deterministic lineage and inferred side."""
    ear = _prepared_ear()
    inner = _ProfileExtractor()
    cache = CacheManager(cache_root=tmp_path)
    extractor = _cached(inner, cache)

    first = extractor.extract(ear)
    second = extractor.extract(ear)

    for profiles in (first, second):
        assert len(profiles) == 2
        np.testing.assert_array_equal(profiles[0].depths, [0.0, 0.1, 0.0])
        np.testing.assert_array_equal(profiles[1].depths, [0.0, 0.3, -0.1])
        assert (profiles[0].start_degrees, profiles[0].end_degrees) == (5.0, 175.0)
        assert (profiles[1].start_degrees, profiles[1].end_degrees) == (10.0, 170.0)
    assert inner.calls == [ear]
    assert extractor.producer_slug == inner.producer_slug
    assert extractor.alpha_scales == (0.35, 2.0)
    assert cache.exists("alpha-tear-v1", _CACHE_KEY)


@pytest.mark.parametrize("alpha_scales", [(2.0, 0.35), (0.35,), (0.35, 2.0, 10.0)])
def test_cached_profiles_reject_other_alpha_scales(
    tmp_path: Path, alpha_scales: tuple[float, ...]
) -> None:
    """A stored record must have the extractor's alpha scales in the same order."""
    cache = CacheManager(cache_root=tmp_path)
    _cached(_ProfileExtractor(), cache).extract(_prepared_ear())
    reader = _ProfileExtractor(alpha_scales=alpha_scales)

    with pytest.raises(ValueError, match="alpha scales"):
        _cached(reader, cache).extract(_prepared_ear())
    assert reader.calls == []


@pytest.mark.parametrize(
    "record",
    [
        {"depths": [0.0, 0.1, 0.0]},
        {
            "alpha_scales": [0.35, 2.0],
            "profiles": [{"start_degrees": 5.0, "end_degrees": 175.0, "depths": [0.0, 0.1, 0.0]}],
        },
        {
            "alpha_scales": [0.35, 2.0],
            "profiles": [
                {"start_degrees": 5.0, "end_degrees": 175.0, "depths": [0.0, 0.1]},
                "invalid",
            ],
        },
        {"alpha_scales": [0.35, 2.0], "profiles": [[0.0, 0.1, 0.0], [0.0, 0.3, -0.1]]},
        {
            "alpha_scales": [0.35, 2.0],
            "profiles": [
                {"depths": [0.0, 0.1, 0.0]},
                {"start_degrees": 5.0, "end_degrees": 175.0, "depths": [0.0, 0.3, -0.1]},
            ],
        },
    ],
)
def test_cached_profiles_reject_invalid_records(
    tmp_path: Path, record: dict[str, object]
) -> None:
    """A stored record needs one valid tear profile for each alpha scale."""
    cache = CacheManager(cache_root=tmp_path)
    cache.save("alpha-tear-v1", _CACHE_KEY, record)

    with pytest.raises(ValueError, match=r"(?i)tear.profile"):
        _cached(_ProfileExtractor(), cache).extract(_prepared_ear())


def test_cached_profile_rejects_unversioned_extractor(tmp_path: Path) -> None:
    """Experimental extractors cannot accidentally persist outputs."""
    with pytest.raises(ValueError, match="producer slug"):
        CachedTearProfileExtractor(
            _ProfileExtractor(producer_slug=None),
            CacheManager(cache_root=tmp_path),
            segmentation_producer_slug="segmentation-v1",
            landmark_producer_slug="landmarks-v1",
        )


def test_composition_caches_only_settled_profile_extraction(tmp_path: Path) -> None:
    """Standard and tuning builders differ only at profile persistence."""
    store = _PhotoStore(_encoded_image())

    standard = build_standard_alphaphant(store, cache_root=tmp_path / "standard")
    tuning = build_profile_tuning_alphaphant(
        store,
        AlphaTearConfig(alpha_scales=(0.4,)),
        cache_root=tmp_path / "tuning",
    )

    assert isinstance(standard._profile_extractor, CachedTearProfileExtractor)
    assert isinstance(tuning._profile_extractor, AlphaTearExtractor)
    assert tuning._profile_extractor.producer_slug is None


def _rle(mask: np.ndarray) -> dict[str, object]:
    """Encode one synthetic boolean mask as JSON-compatible COCO RLE."""
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {
        "size": list(mask.shape),
        "counts": encoded["counts"].decode("utf-8"),
    }


def _segmented_rectangle(x1: int, y1: int, x2: int, y2: int) -> Detection:
    """Build one rectangular segmented ear."""
    mask = np.zeros((30, 30), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return Detection(
        xyxy=(x1 + 0.2, y1 + 0.2, x2 - 0.2, y2 - 0.2),
        class_name="ear",
        class_id=2,
        confidence=0.9,
        rle_mask=_rle(mask),
    )


def _landmarks(
    upper: tuple[float, float],
    lower: tuple[float, float],
    confidence: float = 0.9,
) -> Detection:
    """Build one full-image two-landmark detection."""
    return Detection(
        xyxy=(0.0, 0.0, 1.0, 1.0),
        class_name="ear",
        class_id=0,
        confidence=confidence,
        keypoints=(upper, lower),
    )


def test_ear_preparation_preserves_detected_and_snapped_landmarks() -> None:
    """AlphaTear keeps detector geometry distinct from contour endpoints."""
    detection = _segmented_rectangle(2, 3, 12, 20)
    original = ((2.4, 3.3), (2.7, 19.2))

    prepared = prepare_ear(
        detection,
        _landmarks(*original),
        source_photo=PHOTO,
        source_box=BoundingBox(2, 3, 12, 20),
    )

    assert prepared.original_landmarks == original
    assert prepared.contour_anchors != original
    np.testing.assert_array_equal(prepared.contour[0], prepared.contour_anchors[0])
    np.testing.assert_array_equal(prepared.contour[-1], prepared.contour_anchors[1])


def _comb_and_flap_mask(mirrored: bool) -> np.ndarray:
    """Return an ear mask whose jagged head side has more contour points than its flap.

    Args:
        mirrored: Whether the flap is on the left of the landmark chord.

    Returns:
        A 50-by-50 boolean mask with the landmark chord at column 20, or 29 if mirrored.
    """
    rows, columns = np.mgrid[0:50, 0:50]
    flap = (columns >= 20) & ((columns - 20) ** 2 + (rows - 25) ** 2 <= 400)
    head = (columns >= 16) & (columns < 20) & (rows >= 5) & (rows <= 45)
    teeth = (columns >= 10) & (columns < 16) & (rows >= 5) & (rows <= 45) & (rows % 4 < 2)
    mask = flap | head | teeth
    return mask[:, ::-1] if mirrored else mask


@pytest.mark.parametrize(("mirrored", "side"), [(False, "left"), (True, "right")])
def test_ear_preparation_keeps_flap_over_jagged_head_side(mirrored: bool, side: EarSide) -> None:
    """Preparation keeps the larger flap when the head side has more contour points."""
    mask = _comb_and_flap_mask(mirrored)
    chord_x = 29.0 if mirrored else 20.0
    detection = Detection(
        xyxy=(0.0, 0.0, 50.0, 50.0),
        class_name="ear",
        class_id=2,
        confidence=0.9,
        rle_mask=_rle(mask),
    )
    source_contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    prepared = prepare_ear(
        detection,
        _landmarks((chord_x, 5.0), (chord_x, 45.0)),
        source_photo=PHOTO,
        source_box=BoundingBox(0, 0, 50, 50),
    )

    kept_points = len(prepared.contour)
    discarded_points = len(source_contours[0]) - kept_points + 2
    assert discarded_points > kept_points
    offsets = (prepared.contour[:, 0] - chord_x) * (-1.0 if mirrored else 1.0)
    assert offsets.min() >= 0.0
    assert offsets.max() >= 19.0
    assert prepared.inferred_side == side


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("start", [0, 2, 5])
def test_ear_side_path_keeps_larger_area_over_farther_spike(reverse: bool, start: int) -> None:
    """The flap path encloses more area with the chord, not the farthest point."""
    flap = np.asarray([[0.0, 0.0], [5.0, 0.0], [5.0, 10.0], [0.0, 10.0]])
    spike = np.asarray([[0.0, 6.0], [-8.0, 5.0], [0.0, 4.0]])
    boundary = np.roll(np.vstack([flap, spike]), -start, axis=0)
    if reverse:
        boundary = boundary[::-1]

    path = ear_side_path(boundary, np.asarray([0.0, 0.0]), np.asarray([0.0, 10.0]))

    np.testing.assert_array_equal(path, flap)


@dataclass
class _PhotoStore:
    """Return one encoded synthetic image and record source reads."""

    encoded: bytes
    reads: list[Photo] = field(default_factory=list)

    def read(self, photo: Photo) -> bytes:
        """Return encoded image bytes for a neutral Photo."""
        self.reads.append(photo)
        return self.encoded


@dataclass
class _Segmenter:
    """Return configured semantic ear detections by photo."""

    by_photo: dict[Photo, tuple[Detection, ...]]
    producer_slug: str = "synthetic-segmentation-v1"
    calls: list[Photo] = field(default_factory=list)

    def segment(self, photo: Photo, image: np.ndarray) -> tuple[Detection, ...]:
        """Return configured ears without class filtering downstream."""
        self.calls.append(photo)
        return self.by_photo[photo]


@dataclass
class _LandmarkDetector:
    """Return configured full-image landmarks by integer source box."""

    by_box: dict[BoundingBox, Detection | None]
    producer_slug: str = "synthetic-landmarks-v1"
    calls: list[tuple[Photo, BoundingBox]] = field(default_factory=list)

    def detect(
        self,
        photo: Photo,
        image: np.ndarray,
        ear_box: BoundingBox,
    ) -> Detection | None:
        """Return landmarks for the requested raster crop."""
        self.calls.append((photo, ear_box))
        return self.by_box[ear_box]


def _encoded_image() -> bytes:
    """Return one encoded 30-by-30 black image."""
    ok, encoded = cv2.imencode(".png", np.zeros((30, 30, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_alphaphant_analysis_returns_two_source_labelled_profiles() -> None:
    """A sighting ear pair is analyzed through all three semantic processors."""
    left = _segmented_rectangle(2, 3, 12, 20)
    right = _segmented_rectangle(16, 3, 27, 20)
    left_box = BoundingBox(2, 3, 12, 20)
    right_box = BoundingBox(16, 3, 27, 20)
    store = _PhotoStore(_encoded_image())
    segmenter = _Segmenter({PHOTO: (left,), RIGHT_PHOTO: (right,)})
    detector = _LandmarkDetector(
        {
            left_box: _landmarks((2.1, 3.2), (2.2, 19.1)),
            right_box: _landmarks((26.1, 3.2), (26.2, 19.1)),
        }
    )
    extractor = _ProfileExtractor()
    analyzer = AlphaPhant(
        prepare_ears=SightingPreparer(
            photo_store=store, ear_segmenter=segmenter, landmark_detector=detector
        ).prepare,
        profile_extractor=extractor,
        ear_comparator=EarComparator(),
    )
    pair = SightingEarPair(PHOTO.sighting_id, PHOTO, RIGHT_PHOTO)

    result = analyzer.analyze(pair)

    assert result == AnalyzedSightingEarPair(
        sighting_id=pair.sighting_id,
        left=AnalyzedEar(PHOTO, "left", left_box, result.left.profiles_by_scale),
        right=AnalyzedEar(
            RIGHT_PHOTO,
            "right",
            right_box,
            result.right.profiles_by_scale,
        ),
    )
    assert [ear.inferred_side for ear in extractor.calls] == ["left", "right"]


def test_alphaphant_analysis_prepares_one_photo_once_for_both_sides() -> None:
    """Same-photo pairs reuse segmentation, landmarks, and prepared geometry."""
    left = _segmented_rectangle(2, 3, 12, 20)
    right = _segmented_rectangle(16, 3, 27, 20)
    left_box = BoundingBox(2, 3, 12, 20)
    right_box = BoundingBox(16, 3, 27, 20)
    store = _PhotoStore(_encoded_image())
    segmenter = _Segmenter({PHOTO: (left, right)})
    detector = _LandmarkDetector(
        {
            left_box: _landmarks((2.0, 3.0), (2.0, 19.0)),
            right_box: _landmarks((26.0, 3.0), (26.0, 19.0)),
        }
    )
    analyzer = AlphaPhant(
        prepare_ears=SightingPreparer(
            photo_store=store, ear_segmenter=segmenter, landmark_detector=detector
        ).prepare,
        profile_extractor=_ProfileExtractor(),
        ear_comparator=EarComparator(),
    )

    result = analyzer.analyze(SightingEarPair(PHOTO.sighting_id, PHOTO, PHOTO))

    assert result.left.source_box == left_box
    assert result.right.source_box == right_box
    assert segmenter.calls == [PHOTO]
    assert detector.calls == [(PHOTO, left_box), (PHOTO, right_box)]


@pytest.mark.parametrize("shared_photo", [False, True])
@pytest.mark.parametrize("smaller_side", ["left", "right"])
def test_preparation_resolves_ears_with_unequal_mask_areas(
    shared_photo: bool,
    smaller_side: EarSide,
) -> None:
    """A declared ear remains available when the other ear is over twice its area."""
    left_x2 = 7 if smaller_side == "left" else 13
    right_x1 = 22 if smaller_side == "right" else 16
    left = _segmented_rectangle(2, 3, left_x2, 20)
    right = _segmented_rectangle(right_x1, 3, 27, 20)
    left_box = BoundingBox(2, 3, left_x2, 20)
    right_box = BoundingBox(right_x1, 3, 27, 20)
    detections = (right, left) if smaller_side == "left" else (left, right)
    right_photo = PHOTO if shared_photo else RIGHT_PHOTO
    segmenter = _Segmenter({PHOTO: detections, right_photo: detections})
    preparer = SightingPreparer(
        photo_store=_PhotoStore(_encoded_image()),
        ear_segmenter=segmenter,
        landmark_detector=_LandmarkDetector(
            {
                left_box: _landmarks((2.0, 3.0), (2.0, 19.0)),
                right_box: _landmarks((26.0, 3.0), (26.0, 19.0)),
            }
        ),
    )

    prepared_left, prepared_right = preparer.prepare(
        SightingEarPair(PHOTO.sighting_id, PHOTO, right_photo)
    )

    assert prepared_left.source_box == left_box
    assert prepared_right.source_box == right_box
    assert prepared_right.source_photo == right_photo
    assert segmenter.calls == ([PHOTO] if shared_photo else [PHOTO, RIGHT_PHOTO])


@pytest.mark.parametrize("shared_photo", [False, True])
def test_preparation_resolves_opposite_side_beyond_two_largest_masks(
    shared_photo: bool,
) -> None:
    """Two larger left-ear candidates do not exclude a smaller right ear."""
    largest_left = _segmented_rectangle(2, 3, 12, 20)
    second_left = _segmented_rectangle(13, 3, 21, 20)
    right = _segmented_rectangle(22, 3, 27, 20)
    left_box = BoundingBox(2, 3, 12, 20)
    second_left_box = BoundingBox(13, 3, 21, 20)
    right_box = BoundingBox(22, 3, 27, 20)
    detections = (second_left, right, largest_left)
    right_photo = PHOTO if shared_photo else RIGHT_PHOTO
    detector = _LandmarkDetector(
        {
            left_box: _landmarks((2.0, 3.0), (2.0, 19.0)),
            second_left_box: _landmarks((13.0, 3.0), (13.0, 19.0)),
            right_box: _landmarks((26.0, 3.0), (26.0, 19.0)),
        }
    )
    preparer = SightingPreparer(
        photo_store=_PhotoStore(_encoded_image()),
        ear_segmenter=_Segmenter({PHOTO: detections, right_photo: detections}),
        landmark_detector=detector,
    )

    prepared_left, prepared_right = preparer.prepare(
        SightingEarPair(PHOTO.sighting_id, PHOTO, right_photo)
    )

    assert prepared_left.source_box == left_box
    assert prepared_right.source_box == right_box
    every_ear = [second_left_box, right_box, left_box]
    expected_calls = [(PHOTO, box) for box in every_ear]
    if not shared_photo:
        expected_calls += [(RIGHT_PHOTO, box) for box in every_ear]
    assert detector.calls == expected_calls


def test_alphaphant_analysis_preserves_input_order_for_equal_matching_ears() -> None:
    """Exact cleaned-area ties resolve to the first matching candidate."""
    first = _segmented_rectangle(2, 3, 10, 20)
    second = _segmented_rectangle(12, 3, 20, 20)
    right = _segmented_rectangle(20, 3, 28, 20)
    first_box = BoundingBox(2, 3, 10, 20)
    second_box = BoundingBox(12, 3, 20, 20)
    right_box = BoundingBox(20, 3, 28, 20)
    analyzer = AlphaPhant(
        prepare_ears=SightingPreparer(
            photo_store=_PhotoStore(_encoded_image()),
            ear_segmenter=_Segmenter({PHOTO: (first, second), RIGHT_PHOTO: (right,)}),
            landmark_detector=_LandmarkDetector(
                {
                    first_box: _landmarks((2.0, 3.0), (2.0, 19.0)),
                    second_box: _landmarks((12.0, 3.0), (12.0, 19.0)),
                    right_box: _landmarks((27.0, 3.0), (27.0, 19.0)),
                }
            ),
        ).prepare,
        profile_extractor=_ProfileExtractor(),
        ear_comparator=EarComparator(),
    )

    result = analyzer.analyze(SightingEarPair(PHOTO.sighting_id, PHOTO, RIGHT_PHOTO))

    assert result.left.source_box == first_box


def test_alphaphant_analysis_reports_declared_side_and_domain_stage() -> None:
    """A missing declared ear raises an inspectable domain-level failure."""
    analyzer = AlphaPhant(
        prepare_ears=SightingPreparer(
            photo_store=_PhotoStore(_encoded_image()),
            ear_segmenter=_Segmenter({PHOTO: ()}),
            landmark_detector=_LandmarkDetector({}),
        ).prepare,
        profile_extractor=_ProfileExtractor(),
        ear_comparator=EarComparator(),
    )
    pair = SightingEarPair(PHOTO.sighting_id, PHOTO, RIGHT_PHOTO)

    with pytest.raises(MatchingError) as caught:
        analyzer.analyze(pair)

    assert caught.value.side == "left"
    assert caught.value.stage is PreparationStage.DECLARED_EAR_RESOLUTION
    assert caught.value.photo_id == PHOTO.photo_id


def test_alphaphant_analysis_wraps_extraction_failure_with_its_cause() -> None:
    """Processor failures retain both their domain stage and original cause."""
    left = _segmented_rectangle(2, 3, 12, 20)
    right = _segmented_rectangle(16, 3, 27, 20)

    class _FailingExtractor:
        """Raise one representative extraction failure."""

        producer_slug = None
        alpha_scales = (0.35,)

        def extract(self, ear: PreparedEar) -> tuple[TearProfile, ...]:
            """Fail during profile extraction."""
            raise ArithmeticError("synthetic extraction failure")

    analyzer = AlphaPhant(
        prepare_ears=SightingPreparer(
            photo_store=_PhotoStore(_encoded_image()),
            ear_segmenter=_Segmenter({PHOTO: (left,), RIGHT_PHOTO: (right,)}),
            landmark_detector=_LandmarkDetector(
                {
                    BoundingBox(2, 3, 12, 20): _landmarks((2.0, 3.0), (2.0, 19.0)),
                    BoundingBox(16, 3, 27, 20): _landmarks((26.0, 3.0), (26.0, 19.0)),
                }
            ),
        ).prepare,
        profile_extractor=_FailingExtractor(),
        ear_comparator=EarComparator(),
    )

    with pytest.raises(MatchingError) as caught:
        analyzer.analyze(SightingEarPair(PHOTO.sighting_id, PHOTO, RIGHT_PHOTO))

    assert caught.value.side == "left"
    assert caught.value.stage == "tear-profile extraction"
    assert isinstance(caught.value.__cause__, ArithmeticError)
