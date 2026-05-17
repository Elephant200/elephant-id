import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from elephant_id.cache import CacheManager
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo


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
        # Only return first result
        predictions = json.loads(results[0].to_json(decimals=1))
        return {
            "predictions": predictions,
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
        key = f"{photo.identifier}" # Intentionally omit crop; no need to rerun for different crops.

        coords = self.cache_manager.get_or_compute(
            key=key,
            compute_fn=lambda: self.runner.run(
                image=self.dataset.read_image(photo, crop=crop_xyxy)
            ),
        )
        # TODO: translate coords to absolute coordinates
        return coords
