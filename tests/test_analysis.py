"""Behavior tests for sighting analysis and AlphaTear values."""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
import pytest
from pycocotools import mask as coco_mask

from elephant_id.analysis import (
    EarAnalysis,
    EarSide,
    PreparedEar,
    SightingAnalysis,
    SightingAnalysisError,
    SightingAnalysisStage,
    SightingAnalyzer,
    SightingPreparer,
    TearProfile,
    prepare_ear,
)
from elephant_id.analysis.profile_extraction import (
    AlphaTearConfig,
    AlphaTearExtractor,
    CachedTearProfileExtractor,
)
from elephant_id.cache import CacheManager
from elephant_id.composition import (
    build_profile_tuning_analyzer,
    build_standard_analyzer,
)
from elephant_id.domain import Photo, SightingEarPair
from elephant_id.image.boxes import BoundingBox
from elephant_id.inference import Detection

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

    profile = TearProfile(source)
    source[1] = 0.9

    np.testing.assert_array_equal(profile.depths, [0.0, 0.1, -0.02])
    with pytest.raises(ValueError, match="read-only"):
        profile.depths[0] = 1.0


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


@dataclass
class _ProfileExtractor:
    """Record prepared-ear extraction."""

    profile: TearProfile = field(
        default_factory=lambda: TearProfile(np.asarray([0.0, 0.1, 0.0]))
    )
    producer_slug: str | None = "alpha-tear-v3"
    calls: list[PreparedEar] = field(default_factory=list)

    def extract(self, ear: PreparedEar) -> TearProfile:
        """Record and return one configured profile."""
        self.calls.append(ear)
        return self.profile


def test_cached_profile_uses_prepared_ear_lineage(tmp_path: Path) -> None:
    """Profile persistence keys compact deterministic lineage and inferred side."""
    ear = _prepared_ear()
    inner = _ProfileExtractor()
    cache = CacheManager(cache_root=tmp_path)
    extractor = CachedTearProfileExtractor(
        inner,
        cache,
        segmentation_producer_slug="sam3-features",
        landmark_producer_slug="yolo26n-keypoints-v1",
    )

    first = extractor.extract(ear)
    second = extractor.extract(ear)

    np.testing.assert_array_equal(first.depths, inner.profile.depths)
    np.testing.assert_array_equal(second.depths, inner.profile.depths)
    assert inner.calls == [ear]
    assert extractor.producer_slug == inner.producer_slug
    key = (
        f"{PHOTO.photo_id}__seg_sam3-features__landmarks_"
        "yolo26n-keypoints-v1__side_left__crop_2_3_13_16"
    )
    assert cache.exists("alpha-tear-v3", key)


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

    standard = build_standard_analyzer(store, cache_root=tmp_path / "standard")
    tuning = build_profile_tuning_analyzer(
        store,
        AlphaTearConfig(alpha_fraction=0.4),
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


def test_sighting_analyzer_returns_two_source_labelled_profiles() -> None:
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
    analyzer = SightingAnalyzer(
        prepare_ears=SightingPreparer(
            photo_store=store, ear_segmenter=segmenter, landmark_detector=detector
        ).prepare,
        profile_extractor=extractor,
    )
    pair = SightingEarPair(PHOTO.sighting_id, PHOTO, RIGHT_PHOTO)

    result = analyzer.analyze(pair)

    assert result == SightingAnalysis(
        left=EarAnalysis(PHOTO, "left", left_box, result.left.tear_profile),
        right=EarAnalysis(
            RIGHT_PHOTO,
            "right",
            right_box,
            result.right.tear_profile,
        ),
    )
    assert [ear.inferred_side for ear in extractor.calls] == ["left", "right"]


@pytest.mark.parametrize("prepare_only", (False, True))
def test_sighting_analyzer_prepares_one_photo_once_for_both_sides(
    prepare_only: bool,
) -> None:
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
    extractor = _ProfileExtractor()
    analyzer = SightingAnalyzer(
        prepare_ears=SightingPreparer(
            photo_store=store, ear_segmenter=segmenter, landmark_detector=detector
        ).prepare,
        profile_extractor=extractor,
    )

    pair = SightingEarPair(PHOTO.sighting_id, PHOTO, PHOTO)
    if prepare_only:
        prepared_left, prepared_right = analyzer.prepare(pair)
        assert extractor.calls == []
    else:
        result = analyzer.analyze(pair)
        prepared_left, prepared_right = result.left, result.right
        assert len(extractor.calls) == 2

    assert prepared_left.source_box == left_box
    assert prepared_right.source_box == right_box
    assert segmenter.calls == [PHOTO]
    assert detector.calls == [(PHOTO, left_box), (PHOTO, right_box)]


def test_sighting_analyzer_preserves_input_order_for_equal_matching_ears() -> None:
    """Exact cleaned-area ties resolve to the first matching candidate."""
    first = _segmented_rectangle(2, 3, 10, 20)
    second = _segmented_rectangle(12, 3, 20, 20)
    right = _segmented_rectangle(20, 3, 28, 20)
    first_box = BoundingBox(2, 3, 10, 20)
    second_box = BoundingBox(12, 3, 20, 20)
    right_box = BoundingBox(20, 3, 28, 20)
    analyzer = SightingAnalyzer(
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
    )

    result = analyzer.analyze(SightingEarPair(PHOTO.sighting_id, PHOTO, RIGHT_PHOTO))

    assert result.left.source_box == first_box


def test_sighting_analyzer_reports_declared_side_and_domain_stage() -> None:
    """A missing declared ear raises an inspectable domain-level failure."""
    analyzer = SightingAnalyzer(
        prepare_ears=SightingPreparer(
            photo_store=_PhotoStore(_encoded_image()),
            ear_segmenter=_Segmenter({PHOTO: ()}),
            landmark_detector=_LandmarkDetector({}),
        ).prepare,
        profile_extractor=_ProfileExtractor(),
    )
    pair = SightingEarPair(PHOTO.sighting_id, PHOTO, RIGHT_PHOTO)

    with pytest.raises(SightingAnalysisError) as caught:
        analyzer.analyze(pair)

    assert caught.value.side == "left"
    assert caught.value.stage is SightingAnalysisStage.DECLARED_EAR_RESOLUTION
    assert caught.value.photo_id == PHOTO.photo_id


def test_sighting_analyzer_wraps_extraction_failure_with_its_cause() -> None:
    """Processor failures retain both their domain stage and original cause."""
    left = _segmented_rectangle(2, 3, 12, 20)
    right = _segmented_rectangle(16, 3, 27, 20)

    class _FailingExtractor:
        """Raise one representative extraction failure."""

        producer_slug = None

        def extract(self, ear: PreparedEar) -> TearProfile:
            """Fail during profile extraction."""
            raise ArithmeticError("synthetic extraction failure")

    analyzer = SightingAnalyzer(
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
    )

    with pytest.raises(SightingAnalysisError) as caught:
        analyzer.analyze(SightingEarPair(PHOTO.sighting_id, PHOTO, RIGHT_PHOTO))

    assert caught.value.side == "left"
    assert caught.value.stage is SightingAnalysisStage.TEAR_PROFILE_EXTRACTION
    assert isinstance(caught.value.__cause__, ArithmeticError)
