"""
Module wrapping fine-tuned Roboflow-hosted models through InferenceHTTPClient.

TODO: Currently untested. Make sure to test the batch inference.
"""

import os

import numpy as np
from inference_sdk import InferenceHTTPClient, InferenceConfiguration


_client_cache: dict[tuple[float, float], InferenceHTTPClient] = {}


def _get_client(
    confidence: float = 0.5,
    iou: float = 0.5,
) -> InferenceHTTPClient:
    key = (confidence, iou)
    if key not in _client_cache:
        client = InferenceHTTPClient(
            api_url="https://outline.roboflow.com",
            api_key=os.getenv("ROBOFLOW_API_KEY"),
        )
        client.configure(InferenceConfiguration(
            confidence_threshold=confidence,
            iou_threshold=iou,
        ))
        _client_cache[key] = client
    return _client_cache[key]


def infer(
    image: np.ndarray | str,
    model_id: str,
    confidence: float = 0.5,
    iou: float = 0.5,
) -> list[dict]:
    """Run inference on a single image.

    Args:
        image: Path to an image file or a numpy array.
        model_id: Roboflow model identifier (e.g. "model_name/5").
        confidence: Minimum confidence threshold.
        iou: IoU threshold for NMS.

    Returns:
        List of prediction dicts from the model.
    """
    client = _get_client(confidence, iou)
    response = client.infer(image, model_id=model_id)
    return response["predictions"]


def infer_batch(
    images: list[np.ndarray | str],
    model_id: str,
    confidence: float = 0.5,
    iou: float = 0.5,
) -> list[list[dict]]:
    """Run inference on a list of images.

    Args:
        images: List of image paths or numpy arrays.
        model_id: Roboflow model identifier (e.g. "model_name/5").
        confidence: Minimum confidence threshold.
        iou: IoU threshold for NMS.

    Returns:
        List of prediction lists, one per input image.
    """
    client = _get_client(confidence, iou)
    response = client.infer(images, model_id=model_id)
    return [r["predictions"] for r in response]

if __name__ == "__main__":
    images = [
        "/Users/kayoko/Documents/GitHub/elephant-re-identification/dataset/sample/373_Ariel II left_Feb2011.jpg",
        "/Users/kayoko/Documents/GitHub/elephant-re-identification/dataset/sample/579_BHE II leftside_Nov2010.jpg",
        "/Users/kayoko/Documents/GitHub/elephant-re-identification/dataset/sample/579_BHE II_Feb2011.jpg",
        "/Users/kayoko/Documents/GitHub/elephant-re-identification/dataset/sample/669_Britt I leftside_Apr2011.jpg",
        "/Users/kayoko/Documents/GitHub/elephant-re-identification/dataset/sample/736_Carissa I right_Nov2010.jpg",
        "/Users/kayoko/Documents/GitHub/elephant-re-identification/dataset/sample/736_Carissa I rightside_Nov2010.jpg",
        "/Users/kayoko/Documents/GitHub/elephant-re-identification/dataset/sample/992_Cybele I front_Apr2011.jpg",
    ]
    predictions = infer_batch(images, "elephant-re-id/tusks")
    print(predictions)