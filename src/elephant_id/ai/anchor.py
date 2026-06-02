import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from elephant_id.ai.detection import Detection
from elephant_id.cache import CacheManager
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.image import BgrImage
from elephant_id.image.transforms import apply_crop


def detection_from_prediction(prediction: dict[str, Any]) -> Detection:
    """
    Build a :class:`Detection` from a raw ultralytics keypoint
    prediction.
    """
    box = prediction["box"]
    keypoints = prediction["keypoints"]
    return Detection(
        xyxy=(box["x1"], box["y1"], box["x2"], box["y2"]),
        class_name=str(prediction["name"]).strip(),
        class_id=int(prediction["class"]),
        confidence=float(prediction["confidence"]),
        keypoints=tuple(zip(keypoints["x"], keypoints["y"], strict=True)),
    )


class AnchorRunner:
    """Local-only runner for the anchor keypoint YOLO26 model."""

    def __init__(self) -> None:
        # Initialize ultralytics model and configure for inference
        self.model = YOLO("model_weights/anchor_extraction_yolo26/weights.pt")

    def run(self, image: BgrImage) -> list[Detection]:
        """Run anchor keypoint detection on an image.

        Args:
            image: The image to run the model on.

        Returns:
            The anchor keypoint detections found in the image.
        """
        results = self.model.predict(source=image, device="mps", conf=0.25)

        predictions = json.loads(results[0].to_json(decimals=1))
        return [detection_from_prediction(prediction) for prediction in predictions]


class AnchorService:
    """Run the anchor keypoint detection YOLO26 model with caching."""

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

    def run(
        self,
        photo: Photo,
        crop_xyxy: tuple[float, float, float, float]
    ) -> list[Detection]:
        """Run anchor keypoint detection on a photo crop.

        Should never be rerun for the same photo. Must run on one ear.

        Args:
            photo: The photo to run the model on.
            crop_xyxy: Ear crop in xyxy coordinates.

        Returns:
            Anchor keypoint detections in absolute image coordinates.
        """
        # ``apply_crop`` crops at the floored, non-negative pixel origin (see
        # ``clip_xyxy``). Cached detections are relative to that integer origin,
        # so we derive it once and reuse it for both the cache key and the
        # translate-back, keeping the float crop request and the integer pixel
        # grid consistent.
        ox1, oy1, ox2, oy2 = (int(coord) for coord in crop_xyxy)
        key = f"{photo.identifier}__crop_{ox1}_{oy1}_{ox2}_{oy2}"

        envelope = self.cache_manager.get_or_compute(
            key=key,
            compute_fn=lambda: self._compute(photo, crop_xyxy),
        )

        # Cached detections are crop-relative; translate to absolute image coords.
        return [
            Detection.from_dict(d).translate(float(ox1), float(oy1))
            for d in envelope["detections"]
        ]

    def _compute(
        self, photo: Photo, crop_xyxy: tuple[float, float, float, float]
    ) -> dict:
        """Run the model on the ear crop and build the cache entry."""
        detections = self.runner.run(
            image=apply_crop(self.dataset.read_image(photo), crop_xyxy)
        )
        return {
            "detections": [detection.to_dict() for detection in detections],
        }
