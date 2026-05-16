"""
Module wrapping the SAM3 Roboflow workflow.
"""

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
    Class for running the SAM3 workflow using Roboflow Inference SDK.
    """

    def __init__(
        self,
        api_key: str,
        workspace_name: str = ROBOFLOW_WORKSPACE,
        workflow_id: str = ROBOFLOW_SAM3_WORKFLOW_ID,
        confidence_threshold: float = DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
        nms: bool = DEFAULT_SAM3_NMS,
        nms_iou_threshold: float = DEFAULT_SAM3_NMS_IOU_THRESHOLD,
    ) -> None:
        self.api_key: str = api_key
        self.client = InferenceHTTPClient(
            api_url=ROBOFLOW_API_URL,
            api_key=self.api_key,
        )

        self.workspace_name: str = workspace_name
        self.workflow_id: str = workflow_id
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

        # Normalize output
        return {
            "queries": list(queries),
            "confidence_threshold": self.confidence_threshold,
            "nms": self.nms,
            "nms_iou_threshold": self.nms_iou_threshold,
            "predictions": response[0]["predictions"]["predictions"],
        }


class Sam3Service:
    """
    Service for running the SAM3 workflow and caching the results.
    """

    def __init__(
        self,
        api_key: str,
        dataset: Dataset,
        cache_root: Path = Path(DEFAULT_CACHE_ROOT),
        workspace_name: str = ROBOFLOW_WORKSPACE,
        workflow_id: str = ROBOFLOW_SAM3_WORKFLOW_ID,
        confidence_threshold: float = DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
        nms: bool = DEFAULT_SAM3_NMS,
        nms_iou_threshold: float = DEFAULT_SAM3_NMS_IOU_THRESHOLD,
    ) -> None:
        self.runner = Sam3Runner(
            api_key=api_key,
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            confidence_threshold=confidence_threshold,
            nms=nms,
            nms_iou_threshold=nms_iou_threshold,
        )

        self.dataset: Dataset = dataset
        self.cache_managers: dict[str, CacheManager] = {
            preset: CacheManager(f"sam3/{preset}", cache_root=cache_root)
            for preset in SAM3_QUERY_PRESETS
        }

    def run(self, photo: Photo, query_preset: str) -> dict:
        """
        Run the SAM3 workflow for the given Photo object

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
