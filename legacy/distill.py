"""
Auto-annotate images for a single class and output COCO-format JSON.

Supports two model backends:
  - "sam3": prompt-based segmentation via the SAM3 Roboflow workflow.
    The class name is used directly as the SAM3 query prompt.
  - Any other model string (e.g. "model_name/5"): treated as a Roboflow-
    hosted model ID, accessed via the InferenceHTTPClient.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import PIL
import PIL.Image
from pycocotools import mask as mask_util
from tqdm import tqdm

from vision.sam3 import segment_image_batch
from vision.model import infer

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ── Helpers ──────────────────────────────────────────────────────────

def _scan_images(image_dir: str) -> list[str]:
    """Return a sorted list of image file paths found in *image_dir*."""
    directory = Path(image_dir)
    paths = [
        str(p) for p in sorted(directory.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def _center_to_topleft(x: float, y: float, w: float, h: float) -> list[float]:
    """Convert a center-format bounding box to COCO top-left [x, y, w, h]."""
    return [x - w / 2, y - h / 2, w, h]


def _make_image_entry(image_path: str, image_id: int) -> dict | None:
    """Build a COCO ``images`` entry by reading the file for its dimensions."""
    shape = PIL.Image.open(image_path).size
    return {
        "id": image_id,
        "file_name": os.path.basename(image_path),
        "width": shape[0],
        "height": shape[1],
    }


def _rle_to_json_safe(rle: dict) -> dict:
    """Ensure an RLE dict is JSON-serializable (decode bytes counts)."""
    rle = dict(rle)
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def _polygon_to_rle(points: list[dict], height: int, width: int) -> dict:
    """Convert Roboflow polygon points to a COCO compressed-RLE mask.

    Args:
        points: List of {"x": float, "y": float} dicts from the prediction.
        height: Image height in pixels.
        width: Image width in pixels.
    """
    # Flatten polygon points into [x1, y1, x2, y2, ...] for pycocotools
    polygon = []
    for pt in points:
        polygon.extend([pt["x"], pt["y"]])

    # Encode via pycocotools: polygon -> binary mask -> compressed RLE
    rle_list = mask_util.frPyObjects([polygon], height, width)
    rle = mask_util.merge(rle_list)
    return _rle_to_json_safe(rle)


# ── SAM3 backend ─────────────────────────────────────────────────────

def _annotate_with_sam3(
    image_paths: list[str],
    class_name: str,
) -> tuple[list[dict], list[dict]]:
    """Run SAM3 on every image and collect COCO entries.

    The *class_name* is used directly as the SAM3 text query prompt.
    Response format: {"image": {"width": int, "height": int},
                      "predictions": [{"x", "y", "width", "height", "rle_mask", ...}]}
    """
    queries = [class_name] * len(image_paths)
    responses = segment_image_batch(image_paths, queries)

    images_list: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1

    for image_index, (image_path, response) in enumerate(zip(image_paths, responses)):
        image_id = image_index + 1

        images_list.append({
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": response["image"]["width"],
            "height": response["image"]["height"],
        })

        for prediction in response["predictions"]:
            bbox = _center_to_topleft(
                prediction["x"], prediction["y"],
                prediction["width"], prediction["height"],
            )
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": prediction["width"] * prediction["height"],
                "iscrowd": 0,
                "segmentation": _rle_to_json_safe(prediction["rle_mask"]),
            })
            annotation_id += 1

    return images_list, annotations


# ── Roboflow model backend ───────────────────────────────────────────

def _annotate_with_roboflow_model(
    image_paths: list[str],
    model_id: str,
    class_name: str,
    confidence: float,
    iou: float,
) -> tuple[list[dict], list[dict]]:
    """Run a Roboflow-hosted model on every image and collect COCO entries.

    Predictions are filtered to only include the requested *class_name*.
    If the model returns polygon segmentation data, it is converted to RLE.
    """
    images_list: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1
    errors = 0

    progress = tqdm(image_paths, desc=model_id, unit="img")
    for image_index, image_path in enumerate(progress):
        image_id = image_index + 1

        try:
            image_entry = _make_image_entry(image_path, image_id)
            if image_entry is None:
                logger.warning("Could not read %s, skipping", os.path.basename(image_path))
                continue
            images_list.append(image_entry)
            height = image_entry["height"]
            width = image_entry["width"]

            predictions = infer(
                image_path, model_id, confidence=confidence, iou=iou,
            )

            for prediction in predictions:
                pred_class = prediction["class"]
                if pred_class != class_name:
                    continue

                bbox = _center_to_topleft(
                    prediction["x"], prediction["y"],
                    prediction["width"], prediction["height"],
                )

                annotation: dict = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox,
                    "area": prediction["width"] * prediction["height"],
                    "iscrowd": 0,
                }

                # Convert polygon segmentation to RLE when available
                if "points" in prediction:
                    annotation["segmentation"] = _polygon_to_rle(
                        prediction["points"], height, width,
                    )

                annotations.append(annotation)
                annotation_id += 1

        except Exception:
            errors += 1
            logger.exception("Failed to annotate %s", os.path.basename(image_path))

        progress.set_postfix(annotations=len(annotations), errors=errors)

    if errors:
        logger.warning("Skipped %d images due to errors", errors)

    return images_list, annotations


# ── Public API ───────────────────────────────────────────────────────

def distill(
    image_dir: str,
    output_path: str,
    model: str,
    class_name: str,
    *,
    confidence: float = 0.5,
    iou: float = 0.5,
) -> None:
    """Auto-annotate a directory of images and write COCO-format JSON.

    Args:
        image_dir: Directory containing images to annotate.
        output_path: Where to write the COCO JSON file.
        model: Model to use. Pass ``"sam3"`` for the SAM3 segmentation
            workflow, or a Roboflow model ID (e.g. ``"model_name/5"``)
            for a hosted model.
        class_name: Category name for the output dataset. For SAM3 this
            also serves as the query prompt sent to the model.
        confidence: (Roboflow models only) Minimum confidence threshold.
        iou: (Roboflow models only) IoU threshold for NMS.
    """
    image_paths = _scan_images(image_dir)
    logger.info("Found %d images in %s", len(image_paths), image_dir)

    categories = [{"id": 1, "name": class_name, "supercategory": ""}]

    # Dispatch to the appropriate backend
    if model == "sam3" or True:
        images_list, annotations = _annotate_with_sam3(
            image_paths, class_name,
        )
    else:
        images_list, annotations = _annotate_with_roboflow_model(
            image_paths, model, class_name, confidence, iou,
        )

    # Assemble the full COCO dataset structure
    coco = {
        "info": {
            "description": "Auto-annotated dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": images_list,
        "annotations": annotations,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(
        "Wrote %d annotations across %d images to %s",
        len(annotations), len(images_list), output_path,
    )


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Auto-annotate images and output COCO-format JSON.",
    )
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument("image_dir", help="Directory of images to annotate")
    parser.add_argument("class_name", help="Category name (also the SAM3 query prompt)")
    parser.add_argument("--model", default="sam3", help='Model backend: "sam3" or a Roboflow model ID (e.g. "model/5")')
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    args = parser.parse_args()

    output = args.output or os.path.join(args.image_dir, "annotations.json")

    distill(
        image_dir=args.image_dir,
        output_path=output,
        model=args.model,
        class_name=args.class_name,
        confidence=args.confidence,
        iou=args.iou,
    )
