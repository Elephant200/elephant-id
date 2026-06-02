"""
Module that computes features for an elephant in a given photo by running the SAM3, Anchor, Gender, and Age models.
"""


from pathlib import Path

from elephant_id.ai import (
    AgeService,
    AnchorService,
    Detection,
    GenderService,
    Sam3Service,
)
from elephant_id.constants import DEFAULT_CACHE_ROOT, MIN_FEATURE_BODY_OVERLAP
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo


class PhotoAnalyzer:
    """
    Service that analyzes a photo to compute features, such as the ear masks, gender, and age,
    for an elephant in the photo by running the SAM3, Anchor, Gender, and Age models and
    combining the results.
    """

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

        if not body_detections: # Nothing visible in the photo
            return None

        body = body_detections[0] # TODO: Choose by size; if two are similar size, flag for manual review

        features_on_body: list[Detection] = []
        for feature in feature_detections:
            feature_area = feature.area()
            if feature_area == 0.0:
                continue
            # Fraction of the feature's mask that lies on the body (not IoU).
            overlap = feature.intersection_area(body) / feature_area
            if overlap > MIN_FEATURE_BODY_OVERLAP:
                features_on_body.append(feature)

        trunks, ears, tusks, tails = [], [], [], []
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

        # TODO: Filter sam3 predictions to only include actually visible ears
        if len(ears) > 2:
            # TODO: flag for manual review
            pass

        if len(ears) == 2:
            # Compare sizes; if one is much larger than the other, ignore the smaller one
            pass

        anchor_predictions = []
        for ear in ears:
            anchor_predictions.append(self.anchor_model.run(photo, crop_xyxy=ear.xyxy))

        # TODO: Run anchor model on each ear; remove bad results entirely

        # TODO: Label each ear as left or right. If invalid, flag for manual review.

        # TODO: Convert mask to contour and cut using anchor points

        # Run gender model on body with background removed
        gender_results = self.gender_model.run(photo, body_rle_mask=body.rle_mask)
        bull_prob = gender_results["predictions"]["bull"]
        cow_prob = gender_results["predictions"]["cow"]
        if bull_prob > 0.6:
            gender_code = "B"
            gender_conf = bull_prob
        elif cow_prob > 0.6:
            gender_code = "C"
            gender_conf = cow_prob
        else:
            gender_code = "_"
            gender_conf = 0.5

        # Run age model on body with background removed
        age_results = self.age_model.run(photo, body_rle_mask=body.rle_mask)
        age_confidence = age_results["predictions"]["confidence"]
        age_code = age_results["predictions"]["age"]

        # Return dict should include confidence scores and make clear when results are invalid or uncertain
        # It should also include all raw model outputs for traceability
        return {
            "view": ...,
            "left_ear": {
                "x1": ...,
                "y1": ...,
                "x2": ...,
                "y2": ...,
                "rle_mask": ...,
                "contour": ...,
                "confidence": ...,
            },
            "right_ear": {
                "x1": ...,
                "y1": ...,
                "x2": ...,
                "y2": ...,
                "rle_mask": ...,
                "contour": ...,
                "confidence": ...,
            },
            "left_tusk": {
                "x1": ...,
                "y1": ...,
                "x2": ...,
                "y2": ...,
                "rle_mask": ...,
                "confidence": ...,
            },
            "right_tusk": {
                "x1": ...,
                "y1": ...,
                "x2": ...,
                "y2": ...,
                "rle_mask": ...,
                "confidence": ...,
            },
            "gender": {
                "confidence": gender_conf,
                "gender": gender_code,
            },
            "age": {
                "confidence": age_confidence,
                "age": age_code,
            },
        }
