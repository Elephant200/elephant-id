"""
Module wrapping the SAM3 Roboflow workflow.
"""

import os
from pathlib import Path

from inference_sdk import InferenceHTTPClient

from elephant_id.ai.detection import Detection
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
from elephant_id.image import BgrImage


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
        confidence_threshold: float,
        nms: bool,
        nms_iou_threshold: float,
    ) -> None:
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY is not set")
        self.client = InferenceHTTPClient(
            api_url=ROBOFLOW_API_URL,
            api_key=api_key,
        )

        self.workspace_name: str = ROBOFLOW_WORKSPACE
        self.workflow_id: str = ROBOFLOW_SAM3_WORKFLOW_ID
        self.confidence_threshold: float = confidence_threshold
        self.nms: bool = nms
        self.nms_iou_threshold: float = nms_iou_threshold

    def run(self, image: BgrImage, query_preset: str) -> list[Detection]:
        """
        Run the SAM3 workflow for the given image.

        Args:
            image: BGR image to run SAM3 on
            query_preset: Name of a SAM3 query preset (see SAM3_QUERY_PRESETS)

        Returns:
            The detections found in the image.
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

        if not response or not response[0] or not response[0].get("predictions") or not response[0]["predictions"].get("predictions"):
            raise ValueError(f"Unexpected response from SAM3: {response}")

        return [
            Detection.from_sam3(prediction)
            for prediction in response[0]["predictions"]["predictions"]
        ]


class Sam3Service:
    """
    Service for running the Facebook SAM3 segmentation model and caching the results.
    """

    def __init__(
        self,
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

    def run(self, photo: Photo, query_preset: str) -> list[Detection]:
        """
        Run the SAM3 model for the given Photo object.

        Args:
            photo: Photo object to run SAM3 for
            query_preset: Name of a SAM3 query preset (see SAM3_QUERY_PRESETS)

        Returns:
            The detections found in the photo.
        """
        _resolve_preset(query_preset)

        key = (
            f"{photo.identifier}__"
            f"conf-{self.runner.confidence_threshold:.2f}__"
            f"nms-{self.runner.nms}__"
            f"iou-{self.runner.nms_iou_threshold:.2f}"
        )

        envelope = self.cache_managers[query_preset].get_or_compute(
            key=key,
            compute_fn=lambda: self._compute(photo, query_preset),
        )
        return [Detection.from_dict(d) for d in envelope["detections"]]

    def _compute(self, photo: Photo, query_preset: str) -> dict:
        """Run the model and build the cache envelope (metadata + detections)."""
        detections = self.runner.run(
            image=self.dataset.read_image(photo), query_preset=query_preset
        )
        return {
            "queries": list(_resolve_preset(query_preset)),
            "confidence_threshold": self.runner.confidence_threshold,
            "nms": self.runner.nms,
            "nms_iou_threshold": self.runner.nms_iou_threshold,
            "detections": [detection.to_dict() for detection in detections],
        }
