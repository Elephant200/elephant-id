"""
Module wrapping the SAM3 Roboflow workflow.
"""

import os

import numpy as np
from inference_sdk import InferenceHTTPClient
from tqdm import tqdm

BATCH_SIZE = 8

_client: InferenceHTTPClient | None = None


def _get_client() -> InferenceHTTPClient:
    global _client
    if _client is None:
        _client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=os.getenv("SAM3_API_KEY"), # API key for SEEK Identification workspace
        )
    return _client


def segment_image(
        image: np.ndarray | str,
        queries: list[str] | str,
        confidence_threshold: float = 0.5,
        nms: bool = True,
        nms_iou_threshold: float = 0.2,
    ):
    """
    Segment an image using the SAM3 Roboflow workflow.

    Args:
        image: The image to segment.
        queries: The queries to use for segmentation. Can be a single string, a comma-separated string, or a list of strings.
        confidence_threshold: The confidence threshold for the segmentation.
        nms: Whether to use non-maximum suppression.
        nms_iou_threshold: The IoU threshold for non-maximum suppression.
    
    Returns:
        The predictions from the segmentation.
    """
    response = _get_client().run_workflow(
        workspace_name="seek-identification",
        workflow_id="sam3",
        images={"image": image},
        parameters={
            "queries": queries if isinstance(queries, str) else ", ".join(queries),
            "confidence_threshold": confidence_threshold,
            "nms": nms,
            "nms_iou_threshold": nms_iou_threshold,
            },
    )
    return response[0]["predictions"]["predictions"]


def segment_image_batch(
        images: list[np.ndarray | str],
        queries: list[str],
        force: bool = False
    ):
    """Segment a batch of images using the SAM3 Roboflow workflow.

    Images are sent in batches of 8 to match the workflow's slot layout.

    Args:
        images: List of images to segment (paths or arrays).
        queries: List of queries to use for segmentation.
        force: Whether to bypass the workflow cache.

    Returns:
        List of prediction dicts, one per input image.
    """
    if len(images) != len(queries):
        raise ValueError("Images and queries must have the same length")

    results = []

    client = _get_client()
    progress = tqdm(total=len(images), desc="SAM3", unit="img")
    for start in range(0, len(images), BATCH_SIZE):
        batch_images = images[start:start + BATCH_SIZE]
        batch_queries = queries[start:start + BATCH_SIZE]
        
        # pad with empty images if necessary
        if len(batch_images) < BATCH_SIZE:
            batch_images.extend([np.zeros((5, 5, 3), dtype=np.uint8)] * (BATCH_SIZE - len(batch_images)))
            batch_queries.extend([""] * (BATCH_SIZE - len(batch_queries)))

        images_dict = {f"image{i}": img for i, img in enumerate(batch_images)}
        params_dict = {f"query{i}": q for i, q in enumerate(batch_queries)}

        response = client.run_workflow(
            workspace_name="elephantidentificationresearch",
            workflow_id="sam3-batch-8",
            images=images_dict,
            parameters=params_dict,
            use_cache=not force,
        )

        for key in sorted(response[0]):
            results.append(response[0][key])

        progress.update(len(batch_images))
    progress.close()

    # remove results corresponding to empty images
    results = results[:len(images)]
    

    assert len(results) == len(images)
    return results
