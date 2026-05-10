"""
Convert an "aggregated results" CSV from Roboflow Batch Processing API to COCO JSON.

Usage:
    python -m roboflow.batch_to_coco <image_dir>

Expects a single *_aggregated_results*.csv file alongside the images.
Writes _annotations.coco.json into the same directory.
"""

import argparse
import json
import os
from datetime import datetime
from glob import glob

import pandas as pd


def _center_to_topleft(x: float, y: float, w: float, h: float) -> list[float]:
    """Convert a center-format bounding box to COCO top-left [x, y, w, h]."""
    return [x - w / 2, y - h / 2, w, h]


def batch_to_coco(image_dir: str, clean: bool = False, include_blank: bool = True) -> dict:
    """
    Convert an "aggregated results" CSV from Roboflow Batch Processing API to COCO JSON.

    Finds the CSV automatically in *image_dir*, builds the COCO dict, and
    writes ``_annotations.coco.json`` beside the images.
    """
    csv_files = glob(os.path.join(image_dir, "*_aggregated_results*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No *_aggregated_results*.csv found in {image_dir}"
        )
    csv_path = csv_files[0]

    df = pd.read_csv(csv_path)

    category_map: dict[int, str] = {}
    images_list: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1

    for image_index, row in df.iterrows():
        image_id = int(image_index) + 1
        file_name = row["image"]
        response = json.loads(row["predictions"])

        preds = response["predictions"]
        if not include_blank and len(preds) == 0:
            continue

        images_list.append({
            "id": image_id,
            "file_name": file_name,
            "width": response["image"]["width"],
            "height": response["image"]["height"],
        })

        for pred in preds:
            cat_id = pred["class_id"] + 1
            if cat_id not in category_map:
                category_map[cat_id] = pred["class"]

            bbox = _center_to_topleft(
                pred["x"], pred["y"], pred["width"], pred["height"],
            )

            ann = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": pred["width"] * pred["height"],
                "iscrowd": 0,
            }
            if "rle_mask" in pred:
                ann["segmentation"] = pred["rle_mask"]

            annotations.append(ann)
            annotation_id += 1

    categories = [
        {"id": cid, "name": name, "supercategory": ""}
        for cid, name in sorted(category_map.items())
    ]

    coco = {
        "info": {
            "description": "Converted from Roboflow batch aggregated results",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": images_list,
        "annotations": annotations,
        "categories": categories,
    }

    output_path = os.path.join(image_dir, "_annotations.coco.json")
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(
        f"Wrote {len(annotations)} annotations across "
        f"{len(images_list)} images to {output_path}"
    )
    if clean:
        os.remove(csv_path)
    return coco


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Roboflow batch aggregated results CSV to COCO JSON.",
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing images and *_aggregated_results*.csv",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the input CSV after conversion",
    )
    parser.add_argument(
        "--exclude-blank",
        action="store_true",
        help="Exclude images with no predictions (by default they are included)",
    )
    args = parser.parse_args()
    batch_to_coco(args.image_dir, args.clean, include_blank=not args.exclude_blank)
