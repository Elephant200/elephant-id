"""
Module wrapping the SAM3 Roboflow workflow.
"""

import os
from pathlib import Path

from inference_sdk import InferenceHTTPClient
from PIL import Image

from elephant_id.cache import CacheManager
from elephant_id.constants import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
    DEFAULT_SAM3_NMS,
    DEFAULT_SAM3_NMS_IOU_THRESHOLD,
    ROBOFLOW_API_URL,
    ROBOFLOW_SAM3_WORKFLOW_ID,
    ROBOFLOW_WORKSPACE,
    SAM3_QUERY_PRESETS,
)
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.image_utils import center_to_xyxy


def prediction_center_to_xyxy(prediction: dict) -> dict:
    """
    Return a copy of a SAM3 prediction with xyxy bbox keys instead of center keys.
    """
    x1, y1, x2, y2 = center_to_xyxy(
        float(prediction["x"]),
        float(prediction["y"]),
        float(prediction["width"]),
        float(prediction["height"]),
    )
    converted = {
        key: value
        for key, value in prediction.items()
        if key not in {"x", "y", "width", "height"}
    }
    converted["x1"] = x1
    converted["y1"] = y1
    converted["x2"] = x2
    converted["y2"] = y2
    return converted


def _resolve_preset(preset: str) -> tuple[str, ...]:
    """
    Resolve a preset name to its tuple of query strings.
    """
    if preset not in SAM3_QUERY_PRESETS:
        valid = ", ".join(sorted(SAM3_QUERY_PRESETS.keys()))
        raise ValueError(
            f"Unknown SAM3 query preset: {preset!r}. Valid presets: {valid}"
        )
    return SAM3_QUERY_PRESETS[preset]


class Sam3Runner:
    """
    **Local-only** runner for the Facebook SAM3 segmentation model. Uses the Roboflow Inference SDK.
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
        nms: bool = DEFAULT_SAM3_NMS,
        nms_iou_threshold: float = DEFAULT_SAM3_NMS_IOU_THRESHOLD,
    ) -> None:
        self.client = InferenceHTTPClient(
            api_url=ROBOFLOW_API_URL,
            api_key=os.getenv("ROBOFLOW_API_KEY"),
        )

        self.workspace_name: str = ROBOFLOW_WORKSPACE
        self.workflow_id: str = ROBOFLOW_SAM3_WORKFLOW_ID
        self.confidence_threshold: float = confidence_threshold
        self.nms: bool = nms
        self.nms_iou_threshold: float = nms_iou_threshold

    def run(self, image: Image.Image, query_preset: str) -> dict:
        """
        Run the SAM3 workflow for the given PIL image

        Args:
            image: PIL image to run SAM3 on
            query_preset: Name of a SAM3 query preset (see SAM3_QUERY_PRESETS)

        Returns:
            Dictionary containing the SAM3 output
        """
        queries = _resolve_preset(query_preset)
        response = self.client.run_workflow(
            workspace_name=self.workspace_name,
            workflow_id=self.workflow_id,
            images={"image": image},
            parameters={
                "queries": ",".join(queries),
                "confidence_threshold": self.confidence_threshold,
                "nms": self.nms,
                "nms_iou_threshold": self.nms_iou_threshold,
            },
        )

        if not response or not response[0] or not response[0].get("predictions"):
            raise ValueError(f"Unexpected response from SAM3: {response}")

        # Convert center-format bbox to corner coordinates
        predictions = [
            prediction_center_to_xyxy(prediction)
            for prediction in response[0]["predictions"]["predictions"]
        ]

        # Normalize output to match expected schema
        return {
            "queries": list(queries),
            "confidence_threshold": self.confidence_threshold,
            "nms": self.nms,
            "nms_iou_threshold": self.nms_iou_threshold,
            "predictions": predictions,
        }


class Sam3Service:
    """
    Service for running the Facebook SAM3 segmentation model and caching the results.
    """

    def __init__(
        self,
        api_key: str,
        dataset: Dataset,
        cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    ) -> None:
        self.runner = Sam3Runner(
            confidence_threshold=DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
            nms=DEFAULT_SAM3_NMS,
            nms_iou_threshold=DEFAULT_SAM3_NMS_IOU_THRESHOLD,
        )

        self.dataset: Dataset = dataset
        self.cache_managers: dict[str, CacheManager] = {
            preset: CacheManager(f"sam3/{preset}", cache_root=cache_root)
            for preset in SAM3_QUERY_PRESETS
        }

    def run(self, photo: Photo, query_preset: str) -> dict:
        """
        Run the SAM3 model for the given Photo object

        Args:
            photo: Photo object to run SAM3 for
            query_preset: Name of a SAM3 query preset (see SAM3_QUERY_PRESETS)

        Returns:
            Dictionary containing the SAM3 output
        """
        _resolve_preset(query_preset)

        key = (
            f"{photo.identifier}__"
            f"conf-{self.runner.confidence_threshold:.2f}__"
            f"nms-{self.runner.nms}__"
            f"iou-{self.runner.nms_iou_threshold:.2f}"
        )

        return self.cache_managers[query_preset].get_or_compute(
            key=key,
            compute_fn=lambda: self.runner.run(
                image=self.dataset.read_image(photo), query_preset=query_preset
            ),
        )
