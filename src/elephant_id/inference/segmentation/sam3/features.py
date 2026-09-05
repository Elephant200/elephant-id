"""Uncached SAM3 multi-feature segmentation."""

import os
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger

from elephant_id.constants import (
    DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
    DEFAULT_SAM3_NMS,
    DEFAULT_SAM3_NMS_IOU_THRESHOLD,
    ROBOFLOW_API_URL,
    ROBOFLOW_SAM3_WORKFLOW_ID,
    ROBOFLOW_WORKSPACE,
    SAM3_QUERY_PRESETS,
)
from elephant_id.domain import Photo
from elephant_id.image import BgrImage
from elephant_id.image.boxes import center_to_xyxy
from elephant_id.inference.detection import Detection


@dataclass(frozen=True, slots=True)
class Sam3FeatureConfig:
    """Intentional parameters of the settled SAM3 feature workflow."""

    queries: tuple[str, ...]
    confidence_threshold: float
    nms: bool
    nms_iou_threshold: float


DEFAULT_CONFIG = Sam3FeatureConfig(
    queries=tuple(SAM3_QUERY_PRESETS["features"]),
    confidence_threshold=DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
    nms=DEFAULT_SAM3_NMS,
    nms_iou_threshold=DEFAULT_SAM3_NMS_IOU_THRESHOLD,
)
PRODUCER_SLUG = "sam3-features"


class _WorkflowClient(Protocol):
    """Minimal hosted-workflow client used by SAM3."""

    def run_workflow(self, **kwargs: object) -> list[dict[str, object]]:
        """Run one configured hosted workflow."""
        ...


class _FeatureSegmenter(Protocol):
    """Internal contract shared by raw and cached SAM3 feature processors."""

    producer_slug: str

    def segment_features(
        self,
        photo: Photo,
        image: BgrImage,
    ) -> tuple[Detection, ...]:
        """Return every requested feature detection."""
        ...


def _detection_from_prediction(prediction: dict[str, Any]) -> Detection:
    """Build a full-image Detection from one SAM3 prediction."""
    return Detection(
        xyxy=center_to_xyxy(
            float(prediction["x"]),
            float(prediction["y"]),
            float(prediction["width"]),
            float(prediction["height"]),
        ),
        class_name=str(prediction["class"]).strip(),
        class_id=int(prediction["class_id"]),
        confidence=float(prediction["confidence"]),
        rle_mask=prediction.get("rle_mask"),
    )


class Sam3FeatureSegmenter:
    """Run the settled hosted SAM3 workflow for every requested feature."""

    producer_slug = PRODUCER_SLUG
    config = DEFAULT_CONFIG

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: _WorkflowClient | None = None,
    ) -> None:
        """Configure lazy access to the hosted workflow."""
        self._api_key = api_key
        self._client = client

    def _workflow_client(self) -> _WorkflowClient:
        """Return the hosted client, constructing it on first use."""
        if self._client is None:
            resolved_api_key = self._api_key or os.getenv("ROBOFLOW_API_KEY")
            if not resolved_api_key:
                raise ValueError("ROBOFLOW_API_KEY is not set")
            from inference_sdk import InferenceHTTPClient

            self._client = InferenceHTTPClient(
                api_url=ROBOFLOW_API_URL,
                api_key=resolved_api_key,
            )
        return self._client

    def segment_features(
        self,
        photo: Photo,
        image: BgrImage,
    ) -> tuple[Detection, ...]:
        """Return all requested feature detections in full-image coordinates."""
        response = self._workflow_client().run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_SAM3_WORKFLOW_ID,
            images={"image": image},
            parameters={
                "queries": ",".join(self.config.queries),
                "confidence_threshold": self.config.confidence_threshold,
                "nms": self.config.nms,
                "nms_iou_threshold": self.config.nms_iou_threshold,
            },
        )
        try:
            predictions = response[0]["predictions"]["predictions"]
        except (IndexError, KeyError, TypeError):
            raise ValueError("SAM3 returned an invalid feature response") from None
        if not isinstance(predictions, list):
            raise ValueError("SAM3 predictions must be a list")
        detections = tuple(
            _detection_from_prediction(prediction) for prediction in predictions
        )
        logger.debug(
            f"Segmented SAM3 features for photo {photo.photo_id}: "
            f"{len(detections)} detections"
        )
        return detections
