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
        body_detections = self.sam3.run(photo, "body")
        feature_detections = self.sam3.run(photo, "features")

        # If nothing visible in the photo, return None; it's useless to analyze.
        if not body_detections or not feature_detections:
            return None

        if len(body_detections) == 1:
            body = body_detections[0]
        else:
            body_detections.sort(key=lambda d: d.area(), reverse=True)
            # If largest elephant body is more than double the area of the second largest, use the largest; otherwise, flag.
            if body_detections[0].area() / body_detections[1].area() > MIN_MULTIPLE_BODY_AREA_RATIO: # Arbitrary cutoff
                body = body_detections[0]
            else:
                logger.warning(f"Multiple elephant bodies found in photo {photo}: {len(body_detections)}")
                # TODO: FLAG FOR REVIEW
                return None # for now; later, implement manual review process

        # Filter for features on the body itself
        features_on_body: list[Detection] = []
        for feature in feature_detections:
            feature_area = feature.area()
            if feature_area == 0.0:
                continue
            # Fraction of the feature's mask that lies on the body (not IoU).
            overlap = feature.intersection_area(body) / feature_area
            if overlap > MIN_FEATURE_BODY_OVERLAP:
                features_on_body.append(feature)

        # Categorize features
        trunks: list[Detection] = []
        ears: list[Detection] = []
        tusks: list[Detection] = []
        tails: list[Detection] = []
        for feature in features_on_body:
            if feature.class_name == "elephant trunk":
                trunks.append(feature)
            elif feature.class_name == "ear":
                ears.append(feature)
            elif feature.class_name == "tusk":
                tusks.append(feature)
            elif feature.class_name == "tail":
                tails.append(feature)
            else:
                logger.warning(f"Unknown feature found in photo {photo}: {feature.class_name}")

        if len(ears) > 2:
            # TODO: flag for manual review
            logger.warning(f"Multiple ears found in photo {photo}: {len(ears)}")
            ears.sort(key=lambda d: d.area(), reverse=True)
            ears = ears[:2] # placeholder for now

        if len(ears) == 2:
            # Compare sizes; if one is much larger than the other, ignore the smaller one
            if ears[0].area() / ears[1].area() > MIN_MULTIPLE_EAR_AREA_RATIO:
                ears = [ears[0]] # If one ear is much smaller, it's essentially not there.
            elif ears[1].area() / ears[0].area() > MIN_MULTIPLE_EAR_AREA_RATIO:
                ears = [ears[1]] # If one ear is much smaller, it's essentially not there.
            # Leave both ears if they are similar size.

        anchored_ears: list[AnchoredEar] = []
        for ear in ears:
            anchor_dets = self.anchor_model.run(photo, crop_xyxy=ear.xyxy)
            if len(anchor_dets) == 0:
                logger.warning(f"No anchor detections found for ear on {photo} (ear coords: {ear.xyxy})")
                continue
            elif len(anchor_dets) > 1:
                logger.warning(f"Multiple anchor detections found for ear on {photo} (ear coords: {ear.xyxy}): {len(anchor_dets)}")
                anchor_dets = sorted(anchor_dets, key=lambda d: d.confidence, reverse=True)
            anchored_ears.append(AnchoredEar(ear, anchor_dets[0]))

        if len(anchored_ears) == 0:
            logger.warning(f"No good ears found in photo {photo}")
            # Cancel ear analysis ONLY; continue with other analyses.

        # Now, compute the view
        view = self.compute_view(
            body=body,
            ears=anchored_ears,
            trunks=trunks,
            tusks=tusks,
        )

        # Prepare shared data for field analyzers
        shared_data = {
            "view": view,
            "body": body,
            "trunks": trunks,
            "ears": anchored_ears,
            "tusks": tusks,
        }

        # Run specific field analyzers
        age_evidence = self.age_analyzer.analyze(photo, shared_data)
        gender_evidence = self.gender_analyzer.analyze(photo, shared_data)
        ear_evidence = self.ear_analyzer.analyze(photo, shared_data)
        tusk_evidence = self.tusk_analyzer.analyze(photo, shared_data)

        return {
            "view": view,
            "shared_data": { # note: this is not json serializable
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

    def compute_view(self,
        body: Detection,
        ears: list[AnchoredEar],
        trunks: list[Detection],
        tusks: list[Detection],
    ) -> Literal["left", "right", "front", "unknown"]:
        view = "unknown"
        if len(ears) > 0:
            if len(ears) == 1:
                view = ears[0].side
            elif len(ears) == 2:
                if ears[0].side == ears[1].side:
                    logger.warning("Both ears are on the same side")
                    view = "front" # fallback; may be wrong, but better than "unknown"
                else:
                    view = "front"

        elif len(trunks) > 0: # Fallback to trunk positioning
            relative_trunk_x = (trunks[0].xyxy[0] + trunks[0].xyxy[2]) / 2 - body.xyxy[0] # Center of trunk relative to body
            body_width = body.xyxy[2] - body.xyxy[0]
            ratio = relative_trunk_x / body_width
            if ratio > 0.667:
                view = "right"
            elif ratio < 0.333:
                view = "left"
            else:
                view = "front"

        elif len(tusks) > 0: # Fallback to tusk positioning
            relative_tusk_x = sum((tusk.xyxy[0] + tusk.xyxy[2]) / 2 for tusk in tusks) / len(tusks) - body.xyxy[0]
            # Center of tusks relative to body
            body_width = body.xyxy[2] - body.xyxy[0]
            ratio = relative_tusk_x / body_width
            if ratio > 0.667:
                view = "right"
            elif ratio < 0.333:
                view = "left"
            else:
                view = "front"

        return view
