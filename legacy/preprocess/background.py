import os

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from pycocotools import mask as mask_util

BG_CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY"),
)

def remove_background(image: np.ndarray) -> np.ndarray:
    """Remove the background from an elephant image using a SAM3 Roboflow workflow.

    Segments the elephant body and tusks, merges overlapping tusk masks into the
    body mask, then applies the combined mask to isolate the foreground.

    Args:
        image: BGR image as a numpy array.

    Returns:
        The image with background pixels zeroed out.
    """
    response = BG_CLIENT.run_workflow(
        workspace_name="elephant-re-id",
        workflow_id="remove-bg",
        images={"image": image},
        use_cache=True,
    )

    if not response:
        return image

    body_preds = response[0].get("sam_body", {}).get("predictions", [])
    tusk_preds = response[0].get("sam_tusk", {}).get("predictions", [])

    body_rles = [p["rle_mask"] for p in body_preds if "rle_mask" in p]
    tusk_rles = [p["rle_mask"] for p in tusk_preds if "rle_mask" in p]

    if not body_rles:
        return image

    body_merged = mask_util.merge(body_rles, intersect=0)

    if tusk_rles:
        tusk_merged = mask_util.merge(tusk_rles, intersect=0)
        if mask_util.iou([body_merged], [tusk_merged], [0])[0, 0] > 0:
            body_merged = mask_util.merge([body_merged, tusk_merged], intersect=0)

    mask = mask_util.decode(body_merged)
    return cv2.bitwise_and(image, image, mask=mask)