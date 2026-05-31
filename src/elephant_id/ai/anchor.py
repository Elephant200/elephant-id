import json
from pathlib import Path

from ultralytics import YOLO

from elephant_id.cache import CacheManager
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.image import BgrImage
from elephant_id.image.transforms import apply_crop


class AnchorRunner:
    """
    **Local-only** runner for the anchor keypoint detection YOLO26 model. Uses ultralytics.
    """

    def __init__(self) -> None:
        # Initialize ultralytics model and configure for inference
        self.model = YOLO("model_weights/anchor_extraction_yolo26/weights.pt")

    def run(self, image: BgrImage) -> dict:
        """
        Runs the anchor keypoint detection YOLO26 model on the given image.

        Args:
            image: The image to run the model on.

        Returns:
            A dictionary containing the anchor keypoint detection results.
        """
        results = self.model.predict(source=image, device="mps", conf=0.25)

        # Only return first result
        predictions = json.loads(results[0].to_json(decimals=1))

        normalized_predictions = []
        for pred in predictions:
            normalized_pred = {}
            normalized_pred["confidence"] = pred["confidence"]
            normalized_pred["class_id"] = pred["class"]
            normalized_pred["class"] = pred["name"]
            normalized_pred["x1"] = pred["box"]["x1"]
            normalized_pred["y1"] = pred["box"]["y1"]
            normalized_pred["x2"] = pred["box"]["x2"]
            normalized_pred["y2"] = pred["box"]["y2"]
            normalized_pred["keypoints"] = [
                [pred["keypoints"]["x"][0], pred["keypoints"]["y"][0]],
                [pred["keypoints"]["x"][1], pred["keypoints"]["y"][1]],
            ]
            normalized_predictions.append(normalized_pred)

        # Can add metadata here if needed
        return {
            "predictions": normalized_predictions,
        }

class AnchorService:
    """
    Service for running the anchor keypoint detection YOLO26 model and caching the results.
    """

    def __init__(
        self,
        dataset: Dataset,
        cache_root: Path = Path(DEFAULT_CACHE_ROOT)
    ) -> None:
        self.runner = AnchorRunner()
        self.dataset = dataset
        self.cache_manager = CacheManager(
            namespace="anchor",
            cache_root=cache_root,
        )

    def run(self, photo: Photo, crop_xyxy: tuple[float, float, float, float]) -> dict:
        """
        Runs the anchor keypoint detection YOLO26 model on the given photo and ear crop coordinates.
        Should never be rerun for the same photo. Must be run on an image of a single ear.

        Args:
            photo: The photo to run the model on.
            crop_xyxy: The crop to apply to the image, in xyxy (top left, bottom right) coordinates.

        Returns:
            A dictionary containing the anchor keypoint detection results.
        """
        key = (
            f"{photo.identifier}__"
            f"crop_{int(crop_xyxy[0])}_{int(crop_xyxy[1])}_"
            f"{int(crop_xyxy[2])}_{int(crop_xyxy[3])}" # QUESTION: Each image will be run twice, once per ear. Is this sufficiently unique?
        )

        results = self.cache_manager.get_or_compute(
            key=key,
            compute_fn=lambda: self.runner.run(
                image=apply_crop(self.dataset.read_image(photo), crop_xyxy)
            ),
        )

        # Translate coordinates to absolute coordinates
        for prediction in results["predictions"]:
            prediction["x1"] += crop_xyxy[0] # left x
            prediction["y1"] += crop_xyxy[1] # top y
            prediction["x2"] += crop_xyxy[0] # right x
            prediction["y2"] += crop_xyxy[1] # bottom y
            prediction["keypoints"][0][0] += crop_xyxy[0] # first x
            prediction["keypoints"][0][1] += crop_xyxy[1] # first y
            prediction["keypoints"][1][0] += crop_xyxy[0] # second x
            prediction["keypoints"][1][1] += crop_xyxy[1] # second y

        return results
