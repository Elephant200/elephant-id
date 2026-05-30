"""
Module that computes features for an elephant in a given photo by running the SAM3, Anchor, Gender, and Age models.
"""


from pycocotools import mask as coco_mask

from elephant_id.ai import AgeService, AnchorService, GenderService, Sam3Service
from elephant_id.constants import MIN_FEATURE_BODY_OVERLAP
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo


class PhotoAnalyzer:
    """
    Service that analyzes a photo to compute features, such as the ear masks, gender, and age,
    for an elephant in the photo by running the SAM3, Anchor, Gender, and Age models and
    combining the results.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.sam3: Sam3Service = Sam3Service(
            dataset=dataset,
        )
        self.anchor_model: AnchorService = AnchorService(
            dataset=dataset,
        )
        self.gender_model: GenderService = GenderService(
            dataset=dataset,
        )
        self.age_model: AgeService = AgeService(
            dataset=dataset,
        )

    def analyze(self, photo: Photo) -> dict:
        sam3_body = self.sam3.run(photo, "body")
        body_rle_mask = sam3_body["predictions"][0]["rle_mask"]

        sam3_features = self.sam3.run(photo, "features")

        features_on_body = []
        for pred in sam3_features["predictions"]:
            feature_rle = pred["rle_mask"]
            feature_area = float(coco_mask.area([feature_rle])[0])
            if feature_area == 0.0:
                continue
            intersection = coco_mask.merge(
                [feature_rle, body_rle_mask], intersect=True
            )
            overlap = float(coco_mask.area([intersection])[0]) / feature_area
            if overlap > MIN_FEATURE_BODY_OVERLAP:
                features_on_body.append(pred)

        trunks, ears, tusks, tails = [], [], [], []
        for pred in features_on_body:
            if pred["class"] == "trunk":
                trunks.append(pred)
            elif pred["class"] == "ear":
                ears.append(pred)
            elif pred["class"] == "tusk":
                tusks.append(pred)
            elif pred["class"] == "tail":
                tails.append(pred)
            else:
                raise ValueError(f"Unknown class: {pred['class']}")

        # TODO: Filter sam3 predictions to only include actually visible ears

        anchor_predictions = []
        for ear in ears:
            crop_xyxy = (ear["x1"], ear["y1"], ear["x2"], ear["y2"])
            anchor_predictions.append(self.anchor_model.run(photo, crop_xyxy=crop_xyxy))

        # TODO: Run anchor model on each ear; remove bad results entirely

        # TODO: Label each ear as left or right. If invalid, flag for manual review.

        # TODO: Convert mask to contour and cut using anchor points

        # Run gender model on body with background removed
        gender_results = self.gender_model.run(photo, body_rle_mask=body_rle_mask)
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
        age_results = self.age_model.run(photo, body_rle_mask=body_rle_mask)
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
