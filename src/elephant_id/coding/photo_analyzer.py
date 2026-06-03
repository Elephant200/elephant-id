"""Compute per-photo elephant features from AI model outputs."""


from pathlib import Path

from loguru import logger

from elephant_id.ai import (
    AgeService,
    AnchorService,
    Detection,
    GenderService,
    Sam3Service,
)
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

    def analyze(self, photo: Photo) -> dict | None:
        body_detections = self.sam3.run(photo, "body")
        feature_detections = self.sam3.run(photo, "features")

        # If nothing visible in the photo, return None; it's useless to analyze.
        if not body_detections or not feature_detections or len(body_detections) == 0 or len(feature_detections) == 0:
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
                raise ValueError(f"Unknown class: {feature.class_name}")

        if len(ears) > 2:
            # TODO: flag for manual review
            logger.warning(f"Multiple ears found in photo {photo}: {len(ears)}")
            ears = ears.sort(key=lambda d: d.area(), reverse=True)[:1] # placeholder for now

        if len(ears) == 2:
            # Compare sizes; if one is much larger than the other, ignore the smaller one
            if ears[0].area() / ears[1].area() > MIN_MULTIPLE_EAR_AREA_RATIO:
                ears = [ears[0]] # If one ear is much smaller, it's essentially not there.
            elif ears[1].area() / ears[0].area() > MIN_MULTIPLE_EAR_AREA_RATIO:
                ears = [ears[1]] # If one ear is much smaller, it's essentially not there.
            # Leave both ears if they are similar size.

        anchored_ears: list[Detection] = []
        anchor_predictions: dict[Detection, Detection] = {}
        for ear in ears:
            anchor_dets = self.anchor_model.run(photo, crop_xyxy=ear.xyxy)
            if len(anchor_dets) == 0:
                logger.warning(f"No anchor detections found for ear on {photo} (ear coords: {ear.xyxy})")
                continue
            elif len(anchor_dets) > 1:
                logger.warning(f"Multiple anchor detections found for ear on {photo} (ear coords: {ear.xyxy}): {len(anchor_dets)}")
                anchor_dets = sorted(anchor_dets, key=lambda d: d.confidence, reverse=True)[0]
            anchored_ears.append(ear)
            anchor_predictions[ear] = anchor_dets[0]
        ears = anchored_ears

        if len(ears) == 0:
            logger.warning(f"No good ears found in photo {photo}")
            # Don't run ear analysis, but continue with other analyses

        # TODO: Label each ear as left or right. If invalid, flag for manual review.

        # TODO: Convert mask to contour and cut using anchor points

        # Run gender model on body with background removed
        gender_results = self.gender_model.run(photo, body_rle_mask=body.rle_mask)
        bull_prob = gender_results["predictions"]["bull"]
        cow_prob = gender_results["predictions"]["cow"]

        # Run age model on body with background removed
        age_results = self.age_model.run(photo, body_rle_mask=body.rle_mask)
        age_confidence = age_results["predictions"]["confidence"]
        age_code = age_results["predictions"]["age"]

        # Return dict should include confidence scores and make clear when results are invalid or uncertain
        # It should also include all raw model outputs for traceability
        return {} # Placeholder
