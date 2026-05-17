import os

from elephant_id.ai import AgeService, AnchorService, GenderService, Sam3Service
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo


class FeatureComputeService:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.sam3: Sam3Service = Sam3Service(
            api_key=os.getenv("ROBOFLOW_API_KEY"),
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

    def compute(self, photo: Photo) -> dict:
        sam3_body = self.sam3.run(photo, "body")
        sam3_features = self.sam3.run(photo, "features")
        body_mask = sam3_body["predictions"][0]["rle_mask"] # TODO: add decode logic to anchor model

        anchor_predictions = []
        for pred in sam3_features["predictions"]:
            if pred["class"] == "ear":
                crop = (pred["x"], pred["y"], pred["width"], pred["height"])
                anchor_predictions.append(self.anchor_model.run(photo, crop=crop))
            else:
                continue


        # TODO: Filter sam3 predictions to only include predictions on the body

        # TODO: Filter sam3 predictions to only include actually visible ears

        # TODO: Run anchor model on each ear; remove bad results entirely

        # TODO: Label each ear as left or right. If invalid, flag for manual review.

        # TODO: Convert mask to contour and cut using anchor points

        # Run gender model on body with background removed
        gender = self.gender_model.run(photo, body_mask=body_mask)

        # Run age model on body with background removed
        age = self.age_model.run(photo, body_mask=body_mask)
        # Return dict should include confidence scores and make clear when results are invalid or uncertain

        return {
            "sam3_body": sam3_body,
            "sam3_features": sam3_features,
            "anchor_predictions": anchor_predictions,
            "gender": gender,
            "age": age,
        }
