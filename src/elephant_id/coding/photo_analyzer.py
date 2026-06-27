"""Compute per-photo elephant features from AI model outputs."""

from pathlib import Path
from typing import Literal

from loguru import logger

from elephant_id.ai.age import AgeService
from elephant_id.ai.anchor import AnchorService
from elephant_id.ai.detection import Detection
from elephant_id.ai.gender import GenderService
from elephant_id.ai.sam3 import Sam3Service
from elephant_id.coding.age import AgeFieldAnalyzer
from elephant_id.coding.ears import AnchoredEar, EarFieldAnalyzer
from elephant_id.coding.gender import GenderFieldAnalyzer
from elephant_id.coding.tusks import TuskFieldAnalyzer
from elephant_id.constants import (
    DEFAULT_CACHE_ROOT,
    MIN_FEATURE_BODY_OVERLAP,
    MIN_MULTIPLE_BODY_AREA_RATIO,
    MIN_MULTIPLE_EAR_AREA_RATIO,
)
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo


class PhotoAnalyzer:
    """Analyze a photo by combining model outputs."""

    def __init__(self, dataset: Dataset, cache_root: Path = Path(DEFAULT_CACHE_ROOT)) -> None:
        self.dataset = dataset

        # AI model services
        self.sam3: Sam3Service = Sam3Service(
            dataset=dataset,
            cache_root=cache_root,
        )
        self.anchor_model: AnchorService = AnchorService(
            dataset=dataset,
            cache_root=cache_root,
        )
        self.gender_model: GenderService = GenderService(
            dataset=dataset,
            cache_root=cache_root,
        )
        self.age_model: AgeService = AgeService(
            dataset=dataset,
            cache_root=cache_root,
        )

        # Field analyzers
        self.age_analyzer = AgeFieldAnalyzer(self.age_model)
        self.gender_analyzer = GenderFieldAnalyzer(self.gender_model)
        self.ear_analyzer = EarFieldAnalyzer()
        self.tusk_analyzer = TuskFieldAnalyzer()

    def analyze(self, photo: Photo) -> dict | None:
        """Analyze one photo into flexible per-field evidence dictionaries."""
        body_detections = self.sam3.run(photo, "body")
        feature_detections = self.sam3.run(photo, "features")

        if not body_detections or not feature_detections:
            return None

        body = self._choose_body(photo, body_detections)
        if body is None:
            return None

        features_on_body = self._features_on_body(body, feature_detections)
        trunks, ears, tusks = self._group_features(photo, features_on_body)
        ears = self._choose_usable_ears(photo, ears)
        anchored_ears = self._anchor_ears(photo, ears)

        if len(anchored_ears) == 0:
            logger.warning(f"No good ears found in photo {photo}")

        view = self._estimate_view(
            body=body,
            ears=anchored_ears,
            trunks=trunks,
            tusks=tusks,
        )

        feature_context = {
            "view": view,
            "body": body,
            "trunks": trunks,
            "ears": anchored_ears,
            "tusks": tusks,
        }

        age_evidence = self.age_analyzer.analyze(photo, feature_context)
        gender_evidence = self.gender_analyzer.analyze(photo, feature_context)
        ear_evidence = self.ear_analyzer.analyze(photo, feature_context)
        tusk_evidence = self.tusk_analyzer.analyze(photo, feature_context)

        return {
            "view": view,
            "shared_data": {
                "body": body,
                "trunks": trunks,
                "ears": anchored_ears,
                "tusks": tusks,
            },
            "age": age_evidence,
            "gender": gender_evidence,
            "ears": ear_evidence,
            "tusks": tusk_evidence,
        }

    def _choose_body(
        self,
        photo: Photo,
        body_detections: list[Detection],
    ) -> Detection | None:
        """Choose the body detection to analyze, or defer ambiguous photos."""
        if len(body_detections) == 1:
            return body_detections[0]

        bodies_by_area = sorted(body_detections, key=lambda d: d.area(), reverse=True)
        area_ratio = bodies_by_area[0].area() / bodies_by_area[1].area()
        if area_ratio > MIN_MULTIPLE_BODY_AREA_RATIO:
            return bodies_by_area[0]

        logger.warning(
            f"Multiple elephant bodies found in photo {photo}: {len(body_detections)}"
        )
        return None

    def _features_on_body(
        self,
        body: Detection,
        feature_detections: list[Detection],
    ) -> list[Detection]:
        """Keep feature detections mostly inside the selected body."""
        features_on_body: list[Detection] = []
        for feature in feature_detections:
            feature_area = feature.area()
            if feature_area == 0.0:
                continue
            # Body overlap is feature coverage, not IoU.
            overlap = feature.intersection_area(body) / feature_area
            if overlap > MIN_FEATURE_BODY_OVERLAP:
                features_on_body.append(feature)
        return features_on_body

    def _group_features(
        self,
        photo: Photo,
        features: list[Detection],
    ) -> tuple[list[Detection], list[Detection], list[Detection]]:
        """Split SAM3 feature detections into trunks, ears, and tusks."""
        trunks: list[Detection] = []
        ears: list[Detection] = []
        tusks: list[Detection] = []
        for feature in features:
            if feature.class_name == "elephant trunk":
                trunks.append(feature)
            elif feature.class_name == "ear":
                ears.append(feature)
            elif feature.class_name == "tusk":
                tusks.append(feature)
            elif feature.class_name != "tail":
                logger.warning(
                    f"Unknown feature found in photo {photo}: {feature.class_name}"
                )
        return trunks, ears, tusks

    def _choose_usable_ears(self, photo: Photo, ears: list[Detection]) -> list[Detection]:
        """Keep the usable ear detections for downstream anchoring."""
        if len(ears) > 2:
            logger.warning(f"Multiple ears found in photo {photo}: {len(ears)}")
            ears = sorted(ears, key=lambda d: d.area(), reverse=True)[:2]

        if len(ears) != 2:
            return ears

        if ears[0].area() / ears[1].area() > MIN_MULTIPLE_EAR_AREA_RATIO:
            return [ears[0]]
        if ears[1].area() / ears[0].area() > MIN_MULTIPLE_EAR_AREA_RATIO:
            return [ears[1]]
        return ears

    def _anchor_ears(self, photo: Photo, ears: list[Detection]) -> list[AnchoredEar]:
        """Attach the best anchor detection to each usable ear."""
        anchored_ears: list[AnchoredEar] = []
        for ear in ears:
            anchor_detections = self.anchor_model.run(photo, crop_xyxy=ear.xyxy)
            if len(anchor_detections) == 0:
                logger.warning(
                    f"No anchor detections found for ear on {photo} "
                    f"(ear coords: {ear.xyxy})"
                )
                continue
            if len(anchor_detections) > 1:
                logger.warning(
                    f"Multiple anchor detections found for ear on {photo} "
                    f"(ear coords: {ear.xyxy}): {len(anchor_detections)}"
                )
                anchor_detections = sorted(
                    anchor_detections,
                    key=lambda d: d.confidence,
                    reverse=True,
                )
            anchored_ears.append(AnchoredEar(ear, anchor_detections[0]))
        return anchored_ears

    def _estimate_view(
        self,
        body: Detection,
        ears: list[AnchoredEar],
        trunks: list[Detection],
        tusks: list[Detection],
    ) -> Literal["left", "right", "front", "unknown"]:
        """Estimate elephant view from ears, then trunk/tusk horizontal position."""
        if len(ears) > 0:
            if len(ears) == 1:
                return ears[0].side
            if ears[0].side == ears[1].side:
                logger.warning("Both ears are on the same side")
            return "front"

        elif len(trunks) > 0:
            trunk_center_x = (trunks[0].xyxy[0] + trunks[0].xyxy[2]) / 2
            return self._view_from_horizontal_position(body, trunk_center_x)

        elif len(tusks) > 0:
            tusk_center_x = sum(
                (tusk.xyxy[0] + tusk.xyxy[2]) / 2 for tusk in tusks
            ) / len(tusks)
            return self._view_from_horizontal_position(body, tusk_center_x)

        return "unknown"

    def _view_from_horizontal_position(
        self,
        body: Detection,
        feature_center_x: float,
    ) -> Literal["left", "right", "front"]:
        """Map a feature center to left/front/right body thirds."""
        ratio = (feature_center_x - body.xyxy[0]) / (body.xyxy[2] - body.xyxy[0])
        if ratio > 0.667:
            return "right"
        if ratio < 0.333:
            return "left"
        return "front"
