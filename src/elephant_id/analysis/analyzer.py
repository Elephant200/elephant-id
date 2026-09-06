"""Orchestration from a sighting ear pair to tear profiles."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from elephant_id.analysis.ear_preparation import (
    EarSide,
    PreparedEar,
    prepare_ear,
)
from elephant_id.analysis.profile_extraction.protocol import TearProfileExtractor
from elephant_id.analysis.tear_profile import TearProfile
from elephant_id.dataset import PhotoStore
from elephant_id.domain import Photo, SightingEarPair
from elephant_id.image import decode_image
from elephant_id.image.boxes import BoundingBox
from elephant_id.inference import Detection, EarLandmarkDetector, EarSegmenter

_MULTIPLE_EAR_AREA_RATIO = 2.0


class SightingAnalysisStage(StrEnum):
    """Domain stages at which sighting analysis can fail."""

    PHOTO_RETRIEVAL = "photo retrieval"
    IMAGE_DECODING = "image decoding"
    EAR_SEGMENTATION = "ear segmentation"
    EAR_LANDMARK_DETECTION = "ear landmark detection"
    EAR_CONTOUR_PREPARATION = "ear-contour preparation"
    DECLARED_EAR_RESOLUTION = "declared-ear resolution"
    TEAR_PROFILE_EXTRACTION = "tear-profile extraction"


class SightingAnalysisError(RuntimeError):
    """Report a failed declared side and domain stage.

    A shared source photo is prepared when its first declared side is analyzed,
    so a pre-resolution failure is attributed to that side.
    """

    def __init__(
        self,
        *,
        photo: Photo,
        side: EarSide,
        stage: SightingAnalysisStage,
        message: str,
    ) -> None:
        """Initialize one structured sighting-analysis failure."""
        self.photo_id = photo.photo_id
        self.side = side
        self.stage = stage
        super().__init__(
            f"{stage.value.capitalize()} failed for {side} ear in photo {photo.photo_id}: {message}"
        )


@dataclass(frozen=True, slots=True)
class EarAnalysis:
    """One ear tear profile with source photo and box required for reproduction."""

    source_photo: Photo
    side: EarSide
    source_box: BoundingBox
    tear_profile: TearProfile


@dataclass(frozen=True, slots=True)
class SightingAnalysis:
    """Left- and right-ear analysis for one sighting."""

    left: EarAnalysis
    right: EarAnalysis


class SightingPreparer:
    """Resolve neutral sighting ear pairs into shared prepared-ear geometry."""

    def __init__(
        self,
        *,
        photo_store: PhotoStore,
        ear_segmenter: EarSegmenter,
        landmark_detector: EarLandmarkDetector,
    ) -> None:
        """Initialize analysis with storage and processing dependencies."""
        self._photo_store = photo_store
        self._ear_segmenter = ear_segmenter
        self._landmark_detector = landmark_detector

    def prepare(self, pair: SightingEarPair) -> tuple[PreparedEar, PreparedEar]:
        """Prepare the declared ears for any catalog-matching representation.

        When one Photo supplies both sides, its inference and geometry are
        computed once. This method does not extract tear profiles.
        """
        left_candidates = self._prepare_photo_ears(pair.left_photo, "left")
        left_ear = self._resolve_declared_side(
            left_candidates,
            pair.left_photo,
            "left",
        )

        if pair.right_photo == pair.left_photo:
            right_candidates = left_candidates
        else:
            right_candidates = self._prepare_photo_ears(pair.right_photo, "right")
        right_ear = self._resolve_declared_side(
            right_candidates,
            pair.right_photo,
            "right",
        )

        return left_ear, right_ear

    def _prepare_photo_ears(
        self,
        photo: Photo,
        side: EarSide,
    ) -> tuple[PreparedEar, ...]:
        """Retrieve, segment, and prepare candidate ears from one Photo once."""
        try:
            encoded = self._photo_store.read(photo)
        except Exception as error:
            raise SightingAnalysisError(
                photo=photo,
                side=side,
                stage=SightingAnalysisStage.PHOTO_RETRIEVAL,
                message=str(error),
            ) from error
        try:
            image = decode_image(encoded)
        except Exception as error:
            raise SightingAnalysisError(
                photo=photo,
                side=side,
                stage=SightingAnalysisStage.IMAGE_DECODING,
                message=str(error),
            ) from error
        try:
            detections = self._ear_segmenter.segment(photo, image)
        except Exception as error:
            raise SightingAnalysisError(
                photo=photo,
                side=side,
                stage=SightingAnalysisStage.EAR_SEGMENTATION,
                message=str(error),
            ) from error

        prepared: list[PreparedEar] = []
        for detection in self._area_filtered_ears(detections):
            try:
                source_box = BoundingBox.from_float(
                    detection.xyxy,
                    image_width=image.shape[1],
                    image_height=image.shape[0],
                )
            except ValueError as error:
                raise SightingAnalysisError(
                    photo=photo,
                    side=side,
                    stage=SightingAnalysisStage.EAR_SEGMENTATION,
                    message=str(error),
                ) from error
            try:
                landmarks = self._landmark_detector.detect(
                    photo,
                    image,
                    source_box,
                )
            except Exception as error:
                raise SightingAnalysisError(
                    photo=photo,
                    side=side,
                    stage=SightingAnalysisStage.EAR_LANDMARK_DETECTION,
                    message=str(error),
                ) from error
            if landmarks is None:
                continue
            try:
                prepared.append(
                    prepare_ear(
                        detection,
                        landmarks,
                        source_photo=photo,
                        source_box=source_box,
                    )
                )
            except Exception as error:
                raise SightingAnalysisError(
                    photo=photo,
                    side=side,
                    stage=SightingAnalysisStage.EAR_CONTOUR_PREPARATION,
                    message=str(error),
                ) from error
        return tuple(prepared)

    @staticmethod
    def _area_filtered_ears(
        detections: tuple[Detection, ...],
    ) -> tuple[Detection, ...]:
        """An ear-area heuristic to limit the number of candidate ears."""
        if len(detections) > 2:
            detections = tuple(sorted(detections, key=Detection.area, reverse=True)[:2])
        if len(detections) != 2:
            return detections
        first_area = detections[0].area()
        second_area = detections[1].area()
        if first_area / second_area > _MULTIPLE_EAR_AREA_RATIO:
            return (detections[0],)
        if second_area / first_area > _MULTIPLE_EAR_AREA_RATIO:
            return (detections[1],)
        return detections

    @staticmethod
    def _resolve_declared_side(
        candidates: Sequence[PreparedEar],
        photo: Photo,
        side: EarSide,
    ) -> PreparedEar:
        """Resolve one declared side by inferred geometry and cleaned area."""
        matching = tuple(
            candidate for candidate in candidates if candidate.inferred_side == side
        )
        if not matching:
            raise SightingAnalysisError(
                photo=photo,
                side=side,
                stage=SightingAnalysisStage.DECLARED_EAR_RESOLUTION,
                message=f"no prepared ear matches inferred side {side}",
            )
        return max(matching, key=lambda candidate: candidate.cleaned_area)


class SightingAnalyzer:
    """Extract tear profiles from an injected ear-preparation computation."""

    def __init__(
        self,
        *,
        prepare_ears: Callable[[SightingEarPair], tuple[PreparedEar, PreparedEar]],
        profile_extractor: TearProfileExtractor,
    ) -> None:
        """Compose shared preparation with one tear-profile producer."""
        self._prepare_ears = prepare_ears
        self._profile_extractor = profile_extractor

    def prepare(self, pair: SightingEarPair) -> tuple[PreparedEar, PreparedEar]:
        """Expose the same prepared ears to alternative catalog matchers."""
        return self._prepare_ears(pair)

    def analyze(self, pair: SightingEarPair) -> SightingAnalysis:
        """Return left- and right-ear analysis for one sighting ear pair."""
        left_ear, right_ear = self.prepare(pair)
        return SightingAnalysis(
            left=self._extract(left_ear, "left"),
            right=self._extract(right_ear, "right"),
        )

    def _extract(self, ear: PreparedEar, side: EarSide) -> EarAnalysis:
        """Extract and label one resolved prepared ear."""
        try:
            profile = self._profile_extractor.extract(ear)
        except Exception as error:
            raise SightingAnalysisError(
                photo=ear.source_photo,
                side=side,
                stage=SightingAnalysisStage.TEAR_PROFILE_EXTRACTION,
                message=str(error),
            ) from error
        return EarAnalysis(
            source_photo=ear.source_photo,
            side=side,
            source_box=ear.source_box,
            tear_profile=profile,
        )
