"""
Module wrapping the SAM3 Roboflow workflow.
"""

from pathlib import Path

from inference_sdk import InferenceHTTPClient
from PIL import Image

from elephant_id.cache import CacheManager
from elephant_id.dataset import Dataset
from elephant_id.models import Photo


def normalize_queries(queries: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """
    Normalize the queries to a tuple of strings
    """
    if isinstance(queries, str):
        return tuple(q.strip() for q in queries.split(",") if q.strip())
    return tuple(q.strip() for q in queries if q.strip())


class Sam3Runner:
    """
    Class for running the SAM3 workflow using Roboflow Inference SDK.
    """

    def __init__(
        self,
        api_key: str,
        workspace_name: str = "seek-identification",
        workflow_id: str = "sam3",
        confidence_threshold: float = 0.6,
        nms: bool = True,
        nms_iou_threshold: float = 0.2,
    ) -> None:
        self.api_key: str = api_key
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=self.api_key,
        )

        self.workspace_name: str = workspace_name
        self.workflow_id: str = workflow_id
        self.confidence_threshold: float = confidence_threshold
        self.nms: bool = nms
        self.nms_iou_threshold: float = nms_iou_threshold

    def run(
        self, image: Image.Image, queries: str | list[str] | tuple[str, ...]
    ) -> dict:
        """
        Run the SAM3 workflow for the given PIL image

        Args:
            image: PIL image to run SAM3 on
            queries: Queries to send to SAM3

        Returns:
            Dictionary containing the SAM3 output
        """
        response = self.client.run_workflow(
            workspace_name=self.workspace_name,
            workflow_id=self.workflow_id,
            images={"image": image},
            parameters={
                "queries": ", ".join(normalize_queries(queries)),
                "confidence_threshold": self.confidence_threshold,
                "nms": self.nms,
                "nms_iou_threshold": self.nms_iou_threshold,
            },
        )

        if not response or not response[0] or not response[0].get("predictions"):
            raise ValueError(f"Unexpected response from SAM3: {response}")

        # Normalize output
        return {
            "queries": list(normalize_queries(queries)),
            "confidence_threshold": self.confidence_threshold,
            "nms": self.nms,
            "nms_iou_threshold": self.nms_iou_threshold,
            "predictions": response[0]["predictions"],
        }


class Sam3Service:
    """
    Service for running the SAM3 workflow and caching the results.
    """

    def __init__(
        self,
        api_key: str,
        dataset: Dataset,
        cache_root: Path | str = ".cache",
        workspace_name: str = "seek-identification",
        workflow_id: str = "sam3",
        confidence_threshold: float = 0.6,
        nms: bool = True,
        nms_iou_threshold: float = 0.2,
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
        self.cache_manager = CacheManager("sam3", cache_root=cache_root)

    def run(self, photo: Photo, queries: str | list[str] | tuple[str, ...]) -> dict:
        """
        Run the SAM3 workflow for the given Photo object

        Args:
            photo: Photo object to run SAM3 for
            queries: Queries to send to SAM3

        Returns:
            Dictionary containing the SAM3 output
        """
        key = (
            f"{photo.identifier}__"
            f"queries-{'-'.join(normalize_queries(queries))}__"
            f"conf-{self.runner.confidence_threshold:.2f}__"
            f"nms-{self.runner.nms}__"
            f"iou-{self.runner.nms_iou_threshold:.2f}"
        )

        return self.cache_manager.get_or_compute(
            key=key,
            compute_fn=lambda: self.runner.run(
                image=self.dataset.read_image(photo), queries=queries
            ),
        )
