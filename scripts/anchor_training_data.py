"""Module to generate improved training data for the anchor keypoint detection model."""

import json
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from elephant_id.ai import AnchorService, Detection, Sam3Service
from elephant_id.coding.ears import AnchoredEar
from elephant_id.constants import (
    MIN_FEATURE_BODY_OVERLAP,
    MIN_MULTIPLE_BODY_AREA_RATIO,
    MIN_MULTIPLE_EAR_AREA_RATIO,
)
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.image.transforms import apply_crop
from elephant_id.log import configure_logging


def get_ears(photo: Photo, sam3: Sam3Service, anchor_model: AnchorService) -> list[AnchoredEar]:
    """
    Get the ears from a photo. Copy-pasted from PhotoAnalyzer.analyze().
    """
    body_detections = sam3.run(photo, "body")
    feature_detections = sam3.run(photo, "features")

    # If nothing visible in the photo, return None; it's useless to analyze.
    if not body_detections or not feature_detections:
        return None

    if len(body_detections) == 1:
        body = body_detections[0]
    else:
        body_detections.sort(key=lambda d: d.area(), reverse=True)
        # If largest elephant body is more than double the area of the second largest, use the largest; otherwise, flag.
        if body_detections[0].area() / body_detections[1].area() > MIN_MULTIPLE_BODY_AREA_RATIO: # Arbitrary cutoff
            body = body_detections[0]
        else:
            logger.warning(f"Multiple elephant bodies found in photo {photo}: {len(body_detections)}")
            # TODO: FLAG FOR REVIEW
            return None # for now; later, implement manual review process

    # Filter for features on the body itself
    features_on_body: list[Detection] = []
    for feature in feature_detections:
        feature_area = feature.area()
        if feature_area == 0.0:
            continue
        # Fraction of the feature's mask that lies on the body (not IoU).
        overlap = feature.intersection_area(body) / feature_area
        if overlap > MIN_FEATURE_BODY_OVERLAP:
            features_on_body.append(feature)

    # Categorize features; deviates from PhotoAnalyzer.analyze() by not categorizing other features.
    ears: list[Detection] = [feature for feature in features_on_body if feature.class_name == "ear"]

    if len(ears) > 2:
        # TODO: flag for manual review
        logger.warning(f"Multiple ears found in photo {photo}: {len(ears)}")
        ears.sort(key=lambda d: d.area(), reverse=True)
        ears = ears[:2] # placeholder for now

    if len(ears) == 2:
        # Compare sizes; if one is much larger than the other, ignore the smaller one
        if ears[0].area() / ears[1].area() > MIN_MULTIPLE_EAR_AREA_RATIO:
            ears = [ears[0]] # If one ear is much smaller, it's essentially not there.
        elif ears[1].area() / ears[0].area() > MIN_MULTIPLE_EAR_AREA_RATIO:
            ears = [ears[1]] # If one ear is much smaller, it's essentially not there.
        # Leave both ears if they are similar size.

    anchored_ears: list[AnchoredEar] = []
    for ear in ears:
        anchor_dets = anchor_model.run(photo, crop_xyxy=ear.xyxy)
        if len(anchor_dets) == 0:
            logger.warning(f"No anchor detections found for ear on {photo} (ear coords: {ear.xyxy})")
            continue
        elif len(anchor_dets) > 1:
            logger.warning(f"Multiple anchor detections found for ear on {photo} (ear coords: {ear.xyxy}): {len(anchor_dets)}")
            anchor_dets = sorted(anchor_dets, key=lambda d: d.confidence, reverse=True)
        anchored_ears.append(AnchoredEar(ear, anchor_dets[0]))

    if len(anchored_ears) == 0:
        logger.warning(f"No good ears found in photo {photo}")
        # Cancel ear analysis ONLY; continue with other analyses.

    return anchored_ears

def main() -> None:
    load_dotenv()
    configure_logging(level="ERROR")

    dataset = Dataset(
        dataset_root=Path("dataset/elephants-alive/coded"),
        metadata_path=Path("dataset/elephants-alive/images.csv"),
    )
    out = Path("outputs/anchor_training")
    out.mkdir(parents=True, exist_ok=True)

    sam3 = Sam3Service(dataset=dataset)
    anchor_model = AnchorService(dataset=dataset) # the worse version of the model; use it for training data generation.

    # Randomly sample photos from the dataset. One per 5 images in a sighting
    if Path("sampled_photos.json").exists():
        with open("sampled_photos.json") as f:
            sampled_photos = [photo for photo in tqdm(json.load(f))]
    else:
        sampled_photos: list[str] = []
        failures = []
        for sighting in tqdm(dataset.iter_sightings()):
            counter = 0
            for photo in sighting.photos:
                try:
                    ears = get_ears(photo, sam3, anchor_model)
                    if ears is None:
                        continue
                    if counter % 5 == 0:
                        sampled_photos.append(photo.identifier)
                    counter += 1
                except Exception as exc:
                    failures.append({"photo": photo, "error": exc})
                    logger.error(f"Error getting ears for photo {photo}: {exc}")
                    continue
        for failure in failures:
            logger.error(failure)

        with open("sampled_photos.json", "w") as f:
            json.dump([photo for photo in sampled_photos], f, indent=4)

    print(len(sampled_photos))

    # For every image in the photo, get the ears and save the image with the ears and anchor points.

    coco_dataset = {
        "info": {
            "description": "Anchor training data",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "left ear",
                "supercategory": "",
                "keypoints": [
                    "lower", "upper",
                ]
            },
            {
                "id": 2,
                "name": "right ear",
                "supercategory": "",
                "keypoints": [
                    "lower", "upper",
                ],
                "skeleton": [
                    3, 4,
                ]
            },
        ],
    }

    try:
        random.shuffle(sampled_photos)
        for identifier in tqdm(sampled_photos):
            if len(coco_dataset["images"]) >= 500:
                break
            photo = dataset.get_photo(identifier)
            ears = get_ears(photo, sam3, anchor_model)
            if ears is None:
                raise AssertionError(f"No ears found for photo {identifier}; this should never happen.")

            image = dataset.read_image(photo)
            for ear in ears:
                width = ear.xyxy[2] - ear.xyxy[0]
                height = ear.xyxy[3] - ear.xyxy[1]
                crop_xyxy = (
                    round(max(0, ear.xyxy[0] - width * 0.15)),
                    round(max(0, ear.xyxy[1] - height * 0.15)),
                    round(min(image.shape[1], ear.xyxy[2] + width * 0.15)),
                    round(min(image.shape[0], ear.xyxy[3] + height * 0.15)),
                )
                image_copy = image.copy()

                # Find the point on the contour that maximizes the quantity y + x
                contour = ear.contour
                if ear.side == "right":
                    lower_anchor = np.argmax(contour[:, 1] + contour[:, 0])
                    upper_anchor = np.argmin(contour[:, 1] - contour[:, 0])
                else:
                    lower_anchor = np.argmax(contour[:, 1] - contour[:, 0])
                    upper_anchor = np.argmin(contour[:, 1] + contour[:, 0])

                # Compute image quality score
                area = ear.area
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    logger.debug("Bad due to zero perimeter;",end="")
                    continue
                circularity = 4.0 * np.pi * area / (perimeter * perimeter)
                logger.debug(f"Circularity: {circularity};\t", end="")
                aspect_ratio = (ear.xyxy[2] - ear.xyxy[0]) / (ear.xyxy[3] - ear.xyxy[1])
                logger.debug(f"Area: {area}, Aspect ratio: {aspect_ratio};\t", end="")
                logger.debug(f"Relative area: {area / (image.shape[0] * image.shape[1])};\t", end="")
                bad = "OK"
                if circularity < 0.4 or circularity > 0.9:
                    logger.debug("Bad due to circularity;",end="")
                    bad = "CIRCULARITY"
                if aspect_ratio < 0.5 or aspect_ratio > 1.2:
                    logger.debug("Bad due to aspect ratio;\t", end="")
                    bad = "ASPECT_RATIO"
                if area / (image.shape[0] * image.shape[1]) < 0.025 or area < 50_000:
                    logger.debug("Bad due to area;",end="")
                    bad = "AREA"
                if abs(ear.xyxy[2] - image.shape[1]) < 5 or abs(ear.xyxy[3] - image.shape[0]) < 5 or ear.xyxy[0] < 5 or ear.xyxy[1] < 5:
                    logger.debug("Bad due to proximity to image edge;",end="")
                    bad = "EDGE"

                if bad != "OK":
                    continue

                # Add to coco dataset
                try:
                    cv2.imwrite(f"outputs/anchor_training/{photo.identifier}_{ear.side}.jpg", apply_crop(image_copy, crop_xyxy))
                except Exception as exc:
                    logger.error(f"Error saving image {photo.identifier}_{ear.side}: {exc}")
                    continue

                coco_dataset["images"].append({
                    "id": len(coco_dataset["images"]) + 1,
                    "file_name": f"{photo.identifier}_{ear.side}.jpg",
                    "width": crop_xyxy[2] - crop_xyxy[0],
                    "height": crop_xyxy[3] - crop_xyxy[1],
                })
                coco_dataset["annotations"].append({
                    "id": len(coco_dataset["annotations"]) + 1,
                    "image_id": coco_dataset["images"][-1]["id"],
                    "category_id": 1 if ear.side == "left" else 2,
                    "bbox": [1, 1, crop_xyxy[2] - crop_xyxy[0] - 2, crop_xyxy[3] - crop_xyxy[1] - 2],
                    "area": (ear.xyxy[2] - ear.xyxy[0]) * (ear.xyxy[3] - ear.xyxy[1]),
                    "iscrowd": 0,
                    "keypoints": [
                        int(contour[lower_anchor, 0]) - crop_xyxy[0], int(contour[lower_anchor, 1]) - crop_xyxy[1], 2,
                        int(contour[upper_anchor, 0]) - crop_xyxy[0], int(contour[upper_anchor, 1]) - crop_xyxy[1], 2,
                    ]
                })

                # color = (0, 255, 0) if bad == "OK" else (0, 0, 255)
                # cv2.drawContours(image_copy, [contour], -1, color, 2)
                # cv2.circle(image_copy, (int(contour[lower_anchor, 0]), int(contour[lower_anchor, 1])), 25, color, -1)
                # cv2.circle(image_copy, (int(contour[upper_anchor, 0]), int(contour[upper_anchor, 1])), 25, color, -1)

                # cv2.imshow(f"Ear {ear.side} ({bad})", apply_crop(image_copy, crop_xyxy))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
    finally:
        with open(out / "anchor_training_data.json", "w") as f:
            json.dump(coco_dataset, f, indent=4)


if __name__ == "__main__":
    main()
