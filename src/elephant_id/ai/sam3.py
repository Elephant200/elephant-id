"""
Module wrapping the SAM3 Roboflow workflow.
"""

import os
from pathlib import Path
from typing import Any

from inference_sdk import InferenceHTTPClient
from loguru import logger

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
from elephant_id.image.boxes import center_to_xyxy


def detection_from_prediction(prediction: dict[str, Any]) -> Detection:
    """Build a :class:`Detection` from a raw Roboflow SAM3 prediction."""
    xyxy = center_to_xyxy(
        float(prediction["x"]),
        float(prediction["y"]),
        float(prediction["width"]),
        float(prediction["height"]),
    )
    return Detection(
        xyxy=xyxy,
        class_name=str(prediction["class"]).strip(),  # SAM3 can emit leading whitespace
        class_id=int(prediction["class_id"]),
        confidence=float(prediction["confidence"]),
        rle_mask=prediction.get("rle_mask"),
    )


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
    """Roboflow-hosted runner for the Facebook SAM3 segmentation workflow."""

    def __init__(
        self,
        confidence_threshold: float,
        nms: bool,
        nms_iou_threshold: float,
        api_key: str | None = None,
        workspace_name: str = ROBOFLOW_WORKSPACE,
    ) -> None:
        resolved_api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not resolved_api_key:
            logger.error("ROBOFLOW_API_KEY is not set; SAM3 unavailable")
            raise ValueError("ROBOFLOW_API_KEY is not set")
        self.client = InferenceHTTPClient(
            api_url=ROBOFLOW_API_URL,
            api_key=resolved_api_key,
        )

        self.workspace_name: str = workspace_name
        self.workflow_id: str = ROBOFLOW_SAM3_WORKFLOW_ID
        self.confidence_threshold: float = confidence_threshold
        self.nms: bool = nms
        self.nms_iou_threshold: float = nms_iou_threshold

    def run(self, image: BgrImage, query_preset: str) -> list[Detection]:
        """
        Run the SAM3 workflow for the given image.

        Args:
            image: BGR image to run SAM3 on
            query_preset: Name of a SAM3 query preset.

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

        if not response or not response[0] or "predictions" not in response[0] or "predictions" not in response[0]["predictions"]:
            logger.error(f"Unexpected response from SAM3: {response} for preset {query_preset!r}")
            raise ValueError(f"Unexpected response from SAM3: {response} for preset {query_preset!r}")

        return [
            detection_from_prediction(prediction)
            for prediction in response[0]["predictions"]["predictions"]
        ]


class Sam3Service:
    """Run the Facebook SAM3 segmentation model with caching.

    The remote runner is constructed lazily on the first cache miss, so a
    fully warm cache serves detections without a ``ROBOFLOW_API_KEY``.
    """

    def __init__(
        self,
        dataset: Dataset,
        cache_root: Path = Path(DEFAULT_CACHE_ROOT),
        api_key: str | None = None,
        workspace_name: str = ROBOFLOW_WORKSPACE,
    ) -> None:
        self.dataset: Dataset = dataset
        self.confidence_threshold: float = DEFAULT_SAM3_CONFIDENCE_THRESHOLD
        self.nms: bool = DEFAULT_SAM3_NMS
        self.nms_iou_threshold: float = DEFAULT_SAM3_NMS_IOU_THRESHOLD
        self._api_key: str | None = api_key
        self._workspace_name: str = workspace_name
        self._runner: Sam3Runner | None = None
        self.cache_managers: dict[str, CacheManager] = {
            preset: CacheManager(f"sam3/{preset}", cache_root=cache_root)
            for preset in SAM3_QUERY_PRESETS
        }

    @property
    def runner(self) -> Sam3Runner:
        """The SAM3 runner, constructed on first access (needs an API key)."""
        if self._runner is None:
            self._runner = Sam3Runner(
                confidence_threshold=self.confidence_threshold,
                nms=self.nms,
                nms_iou_threshold=self.nms_iou_threshold,
                api_key=self._api_key,
                workspace_name=self._workspace_name,
            )
        return self._runner

    @runner.setter
    def runner(self, runner: Sam3Runner) -> None:
        self._runner = runner

    def cache_key(self, photo: Photo) -> str:
        """Return the cache key used for a photo under every SAM3 preset."""
        return (
            f"{photo.identifier}__"
            f"conf-{self.confidence_threshold:.2f}__"
            f"nms-{self.nms}__"
            f"iou-{self.nms_iou_threshold:.2f}"
        )

    def run(self, photo: Photo, query_preset: str) -> list[Detection]:
        """
        Run the SAM3 model for the given Photo object.

        Args:
            photo: Photo object to run SAM3 for
            query_preset: Name of a SAM3 query preset.

        Returns:
            The detections found in the photo.
        """
        _resolve_preset(query_preset)

        key = self.cache_key(photo)

        cached = self.cache_managers[query_preset].get_or_compute(
            key=key,
            compute_fn=lambda: self._compute(photo, query_preset),
        )
        detections = [Detection.from_dict(d) for d in cached["detections"]]
        logger.info(f"Ran SAM3 {query_preset} for {photo.identifier}: {len(detections)} detections")
        return detections

    def _compute(self, photo: Photo, query_preset: str) -> dict:
        """Run the model and assemble the result to cache."""
        detections = self.runner.run(
            image=self.dataset.read_image(photo), query_preset=query_preset
        )
        return {
            "queries": list(_resolve_preset(query_preset)),
            "confidence_threshold": self.confidence_threshold,
            "nms": self.nms,
            "nms_iou_threshold": self.nms_iou_threshold,
            "detections": [detection.to_dict() for detection in detections],
        }
