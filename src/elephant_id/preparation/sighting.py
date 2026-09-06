"""Retrieve and prepare declared sighting ears for independent matchers."""

from collections.abc import Sequence
from enum import StrEnum

from elephant_id.dataset import PhotoStore
from elephant_id.domain import Photo, SightingEarPair
from elephant_id.image import decode_image
from elephant_id.image.boxes import BoundingBox
from elephant_id.inference import Detection, EarLandmarkDetector, EarSegmenter
from elephant_id.matching.protocol import MatchingError
from elephant_id.preparation.ear import EarSide, PreparedEar, prepare_ear

_MULTIPLE_EAR_AREA_RATIO = 2.0


class PreparationStage(StrEnum):
    """Domain stages at which sighting analysis can fail."""

    PHOTO_RETRIEVAL = "photo retrieval"
    IMAGE_DECODING = "image decoding"
    EAR_SEGMENTATION = "ear segmentation"
    EAR_LANDMARK_DETECTION = "ear landmark detection"
    EAR_CONTOUR_PREPARATION = "ear-contour preparation"
    DECLARED_EAR_RESOLUTION = "declared-ear resolution"


class SightingPreparer:
    """Resolve neutral sighting ear pairs into shared prepared-ear geometry."""

    def __init__(
        self,
        *,
        photo_store: PhotoStore,
        ear_segmenter: EarSegmenter,
        landmark_detector: EarLandmarkDetector,
    ) -> None:
        """Initialize preparation with storage and processing dependencies."""
        self._photo_store = photo_store
        self._ear_segmenter = ear_segmenter
        self._landmark_detector = landmark_detector

    @property
    def segmentation_producer_slug(self) -> str:
        """Identify the segmenter for dependent profile cache records."""
        return self._ear_segmenter.producer_slug

    @property
    def landmark_producer_slug(self) -> str:
        """Identify the landmark detector for dependent profile cache records."""
        return self._landmark_detector.producer_slug

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
            raise MatchingError(
                photo=photo,
                side=side,
                stage=PreparationStage.PHOTO_RETRIEVAL,
                message=str(error),
            ) from error
        try:
            image = decode_image(encoded)
        except Exception as error:
            raise MatchingError(
                photo=photo,
                side=side,
                stage=PreparationStage.IMAGE_DECODING,
                message=str(error),
            ) from error
        try:
            detections = self._ear_segmenter.segment(photo, image)
        except Exception as error:
            raise MatchingError(
                photo=photo,
                side=side,
                stage=PreparationStage.EAR_SEGMENTATION,
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
                raise MatchingError(
                    photo=photo,
                    side=side,
                    stage=PreparationStage.EAR_SEGMENTATION,
                    message=str(error),
                ) from error
            try:
                landmarks = self._landmark_detector.detect(
                    photo,
                    image,
                    source_box,
                )
            except Exception as error:
                raise MatchingError(
                    photo=photo,
                    side=side,
                    stage=PreparationStage.EAR_LANDMARK_DETECTION,
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
                raise MatchingError(
                    photo=photo,
                    side=side,
                    stage=PreparationStage.EAR_CONTOUR_PREPARATION,
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
            raise MatchingError(
                photo=photo,
                side=side,
                stage=PreparationStage.DECLARED_EAR_RESOLUTION,
                message=f"no prepared ear matches inferred side {side}",
            )
        return max(matching, key=lambda candidate: candidate.cleaned_area)
