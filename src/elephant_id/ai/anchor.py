import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from elephant_id.cache import CacheManager
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.image_utils import apply_crop


class AnchorRunner:
    """
    Runner for the anchor keypoint detection YOLO26 model. Uses ultralytics.
    """

    def __init__(self) -> None:
        # Initialize ultralytics model and configure for inference
        self.model = YOLO("model_weights/anchor_extraction_yolo26/weights.pt")

    def run(self, image: Image.Image) -> dict:
        """
        Runs the anchor keypoint detection YOLO26 model on the given image.

        Args:
            image: The image to run the model on.

        Returns:
            A dictionary containing the anchor keypoint detection results.
        """
        results = self.model.predict(source=image, device="mps", conf=0.25)
        for result in results:
            print(result.to_json(decimals=1))
        # Only return first result
        predictions = json.loads(results[0].to_json(decimals=1))[0]
        print(predictions)

        # Convert keypoints to list of points
        normalized = {}
        normalized["confidence"] = predictions["confidence"]
        normalized["class_id"] = predictions["class"]
        normalized["class"] = predictions["name"]
        normalized["x1"] = predictions["box"]["x1"]
        normalized["y1"] = predictions["box"]["y1"]
        normalized["x2"] = predictions["box"]["x2"]
        normalized["y2"] = predictions["box"]["y2"]
        normalized["keypoints"] = [
            [predictions["keypoints"]["x"][0], predictions["keypoints"]["y"][0]],
            [predictions["keypoints"]["x"][1], predictions["keypoints"]["y"][1]],
        ]

        # Can add metadata here if needed
        return {
            "predictions": normalized,
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
        results["predictions"]["x1"] += crop_xyxy[0] # left x
        results["predictions"]["y1"] += crop_xyxy[1] # top y
        results["predictions"]["x2"] += crop_xyxy[0] # right x
        results["predictions"]["y2"] += crop_xyxy[1] # bottom y
        results["predictions"]["keypoints"][0][0] += crop_xyxy[0] # first x
        results["predictions"]["keypoints"][0][1] += crop_xyxy[1] # first y
        results["predictions"]["keypoints"][1][0] += crop_xyxy[0] # second x
        results["predictions"]["keypoints"][1][1] += crop_xyxy[1] # second y

        return results
