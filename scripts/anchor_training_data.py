"""Module to generate improved training data for the anchor keypoint detection model."""

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
from elephant_id.image.transforms import apply_crop, apply_mask
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
    configure_logging()

    dataset = Dataset(
        dataset_root=Path("dataset/elephants-alive/coded"),
        metadata_path=Path("dataset/elephants-alive/images.csv"),
    )

    sam3 = Sam3Service(dataset=dataset)
    anchor_model = AnchorService(dataset=dataset) # the worse version of the model; use it for training data generation.

    all_sightings = list(dataset.iter_sightings())
    print(len(all_sightings))

    # Randomly sample photos from the dataset. One per 5 images in a sighting
    sampled_photos = []
    failures = []
    for sighting in tqdm(all_sightings):
        counter = 0
        for photo in sighting.photos:
            try:
                ears = get_ears(photo, sam3, anchor_model)
                if ears is None:
                    continue
                if counter % 5 == 0:
                    sampled_photos.append(photo)
                counter += 1
            except Exception as exc:
                failures.append({"photo": photo, "error": exc})
                logger.error(f"Error getting ears for photo {photo}: {exc}")
                continue
    for failure in failures:
        logger.error(failure)

    print(len(sampled_photos))


    image = dataset.read_image(photo)

    ears = get_ears(photo, sam3, anchor_model)
    print(ears)
    for ear in ears:
        ear_image = apply_mask(image, ear.mask)
        # Find the point on the contour that maximizes the quantity y + x
        contour: np.ndarray = ear.contour
        if ear.side == "right":
            lower_anchor = np.argmax(contour[:, 1] + contour[:, 0])
            upper_anchor = np.argmin(contour[:, 1] - contour[:, 0])
        else:
            lower_anchor = np.argmax(contour[:, 1] - contour[:, 0])
            upper_anchor = np.argmin(contour[:, 1] + contour[:, 0])
        cv2.circle(ear_image, (int(contour[lower_anchor, 0]), int(contour[lower_anchor, 1])), 25, (0, 0, 255), -1)
        cv2.circle(ear_image, (int(contour[upper_anchor, 0]), int(contour[upper_anchor, 1])), 25, (0, 0, 255), -1)

        cv2.imshow(f"Ear {ear.side}", apply_crop(ear_image, ear.xyxy))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
