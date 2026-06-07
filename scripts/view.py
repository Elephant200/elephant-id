import argparse
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv
from loguru import logger

from elephant_id.ai import AnchorService, Detection, Sam3Service
from elephant_id.coding.analyzers.ears import Ear
from elephant_id.constants import (
    MIN_FEATURE_BODY_OVERLAP,
    MIN_MULTIPLE_BODY_AREA_RATIO,
    MIN_MULTIPLE_EAR_AREA_RATIO,
)
from elephant_id.dataset import Dataset
from elephant_id.log import configure_logging

if __name__ == "__main__":
    load_dotenv()
    configure_logging(level="DEBUG")

    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", type=str, required=False, default="Nguyen_left")
    args = parser.parse_args()

    photo_preset = args.photo

    dataset = Dataset(
        dataset_root=Path("dataset/elephants-alive/coded"),
        metadata_path=Path("dataset/elephants-alive/images.csv"),
    )

    sam3 = Sam3Service(dataset=dataset)
    anchor_model = AnchorService(dataset=dataset)

    photos = {
        "Nguyen_left": "Nguyen_2012-08-02_07",
        "Nguyen_front": "Nguyen_2012-08-02_05",
        "Nguyen_marginal": "Nguyen_2012-08-02_02",
        "Nguyen_right": "Nguyen_2012-08-02_01",
        "Matambu_right": "Matambu_2019-06-11_09",
        "Wallow_right": "Wallow_2005-12-17_03",
        "Wallow_front": "Wallow_2005-12-17_08",
        "Wallow_left": "Wallow_2005-12-17_07",
    }
    identifier = photos.get(photo_preset, photo_preset)

    photo = dataset.get_photo(identifier)
    image = dataset.read_image(photo)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    body_detections = sam3.run(photo, "body")
    feature_detections = sam3.run(photo, "features")

    # If nothing visible in the photo, return None; it's useless to analyze.
    if not body_detections or not feature_detections:
        logger.warning(f"No body or features found in photo {photo}")
        sys.exit(1)

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
            sys.exit(1)

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

    # Categorize features
    trunks: list[Detection] = []
    ears: list[Detection] = []
    tusks: list[Detection] = []
    tails: list[Detection] = []
    for feature in features_on_body:
        if feature.class_name == "elephant trunk":
            trunks.append(feature)
        elif feature.class_name == "ear":
            ears.append(feature)
        elif feature.class_name == "tusk":
            tusks.append(feature)
        elif feature.class_name == "tail":
            tails.append(feature)
        else:
            logger.warning(f"Unknown feature found in photo {photo}: {feature.class_name}")

    if len(trunks) > 1:
        logger.warning(f"Multiple trunks found in photo {photo}: {len(trunks)}")
        sys.exit(1)
    if len(tails) > 1:
        logger.warning(f"Multiple tails found in photo {photo}: {len(tails)}")
        sys.exit(1)

    if len(tusks) > 2:
        logger.warning(f"Too many tusks found in photo {photo}: {len(tusks)}")
        tusks.sort(key=lambda d: d.area() * d.confidence, reverse=True)
        tusks = tusks[:2] # placeholder for now

    if len(ears) > 2:
        # TODO: flag for manual review
        logger.warning(f"Too many ears found in photo {photo}: {len(ears)}")
        ears.sort(key=lambda d: d.area() * d.confidence, reverse=True)
        ears = ears[:2] # placeholder for now

    if len(ears) == 2:
        # Compare sizes; if one is much larger than the other, ignore the smaller one
        if ears[0].area() / ears[1].area() > MIN_MULTIPLE_EAR_AREA_RATIO:
            ears = [ears[0]] # If one ear is much smaller, it's essentially not there.
        elif ears[1].area() / ears[0].area() > MIN_MULTIPLE_EAR_AREA_RATIO:
            ears = [ears[1]] # If one ear is much smaller, it's essentially not there.
        # Leave both ears if they are similar size.

    anchored_ears: list[Ear] = []
    for ear in ears:
        anchor_dets = anchor_model.run(photo, crop_xyxy=ear.xyxy)
        if len(anchor_dets) == 0:
            logger.warning(f"No anchor detections found for ear on {photo} (ear coords: {ear.xyxy})")
            continue
        elif len(anchor_dets) > 1:
            logger.warning(f"Multiple anchor detections found for ear on {photo} (ear coords: {ear.xyxy}): {len(anchor_dets)}")
            anchor_dets = sorted(anchor_dets, key=lambda d: d.confidence, reverse=True)[0]
        anchored_ears.append(Ear(ear, anchor_dets[0]))

    if len(anchored_ears) == 0:
        logger.warning(f"No good ears found in photo {photo}")
        # Cancel ear analysis ONLY; continue with other analyses.

    print(f"Body Coordinates: \t({body.xyxy[0]:>7.2f}, {body.xyxy[1]:>7.2f}), ({body.xyxy[2]:>7.2f}, {body.xyxy[3]:>7.2f})")
    for trunk in trunks:
        print(f"Trunk Coordinates: \t({trunk.xyxy[0]:>7.2f}, {trunk.xyxy[1]:>7.2f}), ({trunk.xyxy[2]:>7.2f}, {trunk.xyxy[3]:>7.2f})")
    for tusk in tusks:
        print(f"Tusk Coordinates: \t({tusk.xyxy[0]:>7.2f}, {tusk.xyxy[1]:>7.2f}), ({tusk.xyxy[2]:>7.2f}, {tusk.xyxy[3]:>7.2f})")
    for ear in ears:
        print(f"Ear Coordinates: \t({ear.xyxy[0]:>7.2f}, {ear.xyxy[1]:>7.2f}), ({ear.xyxy[2]:>7.2f}, {ear.xyxy[3]:>7.2f})")
        print(f"Aspect ratio: \t{(ear.xyxy[2] - ear.xyxy[0]) / (ear.xyxy[3] - ear.xyxy[1])}")
    for tail in tails:
        print(f"Tail Coordinates: \t({tail.xyxy[0]:>7.2f}, {tail.xyxy[1]:>7.2f}), ({tail.xyxy[2]:>7.2f}, {tail.xyxy[3]:>7.2f})")

    view = "Unknown"
    reason = "Unknown"
    if len(anchored_ears) > 0:
        for ear in anchored_ears:
            print(ear)

        if len(anchored_ears) == 1:
            if anchored_ears[0].side == "left":
                view = "left"
                reason = "One left ear found"
            else:
                view = "right"
                reason = "One right ear found"
        elif len(anchored_ears) == 2:
            if anchored_ears[0].side == anchored_ears[1].side:
                logger.warning(f"Both ears are on the same side in photo {photo}")
                sys.exit(1)
            else:
                view = "front"
                reason = "One left ear and one right ear found"

    print(f"View: {view:<10} {reason:<40}")

    if len(trunks) > 0: # Fallback to trunk positioning
        relative_trunk_x = (trunks[0].xyxy[0] + trunks[0].xyxy[2]) / 2 - body.xyxy[0] # Center of trunk relative to body
        body_width = body.xyxy[2] - body.xyxy[0]
        ratio = relative_trunk_x / body_width
        print(ratio)
        if ratio > 0.667:
            view = "right"
            reason = "Trunk is on the right side of the body"
        elif ratio < 0.333:
            view = "left"
            reason = "Trunk is on the left side of the body"
        else:
            view = "front"
            reason = "Trunk is in the middle of the body"

    print(f"View: {view:<10} {reason:<40}")

    if len(tusks) > 0: # Fallback to tusk positioning
        relative_tusk_x = sum((tusk.xyxy[0] + tusk.xyxy[2]) / 2 for tusk in tusks) / len(tusks) - body.xyxy[0]
        # Center of tusks relative to body
        body_width = body.xyxy[2] - body.xyxy[0]
        ratio = relative_tusk_x / body_width
        print(ratio)
        if ratio > 0.667:
            view = "right"
            reason = "Tusks are on the right side of the body"
        elif ratio < 0.333:
            view = "left"
            reason = "Tusks are on the left side of the body"
        else:
            view = "front"
            reason = "Tusks are in the middle of the body"

    print(f"View: {view:<10} {reason:<40}")

    # logger.warning(f"This should not happen, as it should have exited earlier; but it could happen if the only objects found are the ears and they are removed by the anchor model.")
    # sys.exit(1)

    print(f"View: {view}")
