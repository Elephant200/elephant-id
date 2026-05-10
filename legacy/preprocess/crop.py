"""
Module to crop an image of any number of elephants into separate images of each elephant.

Supports two sources of detections:
  1. Pre-computed COCO annotations with SAM segmentations (batch directories).
  2. Live inference via the Roboflow elephant-body-segmentation model.
"""

import json
import os
import sys

from PIL import Image

BATCH_DIRECTORIES = [f"dataset/batch_{i}" for i in range(1, 6)]
OUTPUT_DIRECTORY = "dataset/elephants"

MODEL_ID = "elephant-body-segmentation/1"


def crop_elephants_from_coco(batch_directories: list[str], output_directory: str) -> None:
    """
    For each batch directory, load its COCO annotations and crop the
    largest detection per image by bounding box into the output directory.
    """
    os.makedirs(output_directory, exist_ok=True)

    for batch_dir in batch_directories:
        annotations_path = os.path.join(batch_dir, "_annotations.coco.json")
        with open(annotations_path) as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}

        anns_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        for image_id, anns in anns_by_image.items():
            img_info = images[image_id]
            img_path = os.path.join(batch_dir, img_info["file_name"])
            img = Image.open(img_path)
            stem = os.path.splitext(img_info["file_name"])[0]

            largest = max(anns, key=lambda a: a["area"])
            x, y, w, h = largest["bbox"]
            cropped = img.crop((x, y, x + w, y + h))
            cropped.save(os.path.join(output_directory, f"{stem}.jpg"))


def crop_elephants_from_roboflow(
    image_directory: str,
    output_directory: str,
    *,
    largest_only: bool = True,
) -> None:
    """
    Run the Roboflow elephant-body-segmentation model on every image in
    image_directory and save cropped elephants to output_directory.

    Args:
        image_directory: Directory containing source images.
        output_directory: Where to write cropped results.
        largest_only: If True, keep only the largest detection per image.
                      If False, keep all detections ranked by area.
    """
    from vision.model import infer

    os.makedirs(output_directory, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = sorted(
        p for p in os.listdir(image_directory)
        if os.path.splitext(p)[1].lower() in extensions
    )

    for filename in image_paths:
        img_path = os.path.join(image_directory, filename)
        preds = infer(img_path, MODEL_ID)
        if not preds:
            continue

        img = Image.open(img_path)
        stem = os.path.splitext(filename)[0]

        preds.sort(key=lambda p: p["width"] * p["height"], reverse=True)
        to_crop = preds[:1] if largest_only else preds

        for rank, pred in enumerate(to_crop):
            cx, cy, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            left = cx - w / 2
            top = cy - h / 2
            cropped = img.crop((left, top, left + w, top + h))
            suffix = "" if largest_only and len(to_crop) == 1 else f"_{rank}"
            cropped.save(os.path.join(output_directory, f"{stem}{suffix}.jpg"))


if __name__ == "__main__":
    crop_elephants_from_coco(BATCH_DIRECTORIES, OUTPUT_DIRECTORY)
