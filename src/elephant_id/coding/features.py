from elephant_id.ai.sam3 import Sam3Service
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo


class FeatureComputeService:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.sam3: Sam3Service = ... # Should this be an argument or should it be initialized here?
        self.anchor_model = ... # Should this be an argument or should it be initialized here?
        self.gender_model = ... # Should this be an argument or should it be initialized here?
        self.age_model = ... # Should this be an argument or should it be initialized here?

    def compute(self, photo: Photo) -> dict:
        sam3_body = self.sam3.run(photo, "body")
        sam3_features = self.sam3.run(photo, "features")
        anchor_predictions = self.anchor_model.run(photo)

        # Filter sam3 predictions to only include predictions on the body

        # Filter sam3 predictions to only include actually visible ears

        # Run anchor model on each ear; remove bad results entirely

        # Label each ear as left or right. If invalid, flag for manual review.

        # Run gender model on body with background removed

        # Run age model on body with background removed


        # Return dict should include confidence scores and make clear when results are invalid or uncertain
        return {}
