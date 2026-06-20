"""Module to generate improved training data for the anchor keypoint detection model."""

import json
import os
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

def generate_preliminary_data() -> None:
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
            if len(coco_dataset["images"]) >= 250:
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

def update_hand_annotated_data(input_dir: Path, output_dir: Path) -> None:
    """
    Update the hand-annotated data, which does not have proper boxes
    nor side data, sam3 and the old anchor model predictions.
    """
    import shutil

    coco_datasets = {
        "train": input_dir / "train" / "_annotations.coco.json",
        "valid": input_dir / "valid" / "_annotations.coco.json",
        "test": input_dir / "test" / "_annotations.coco.json",
    }

    dataset = Dataset(
        dataset_root=Path("dataset/elephants-alive/coded"),
        metadata_path=Path("dataset/elephants-alive/images.csv"),
    )
    sam3 = Sam3Service(dataset=dataset)
    anchor_model = AnchorService(dataset=dataset)

    for dataset_name, coco_dir in coco_datasets.items():
        with open(coco_dir) as f:
            coco_data = json.load(f)
        os.makedirs(output_dir / dataset_name, exist_ok=True)
        for image in coco_data["images"]:
            roboflow_filename = image["file_name"]
            filename = image["extra"]["name"]
            del image["extra"]
            del image["date_captured"]
            filename = "_".join([filename.split("_")[0].replace("-", " ")] + filename.split("_")[1:])

            # Change the coco data to match the canonical filename, and rename the file to match the canonical filename.
            shutil.copy2(input_dir / dataset_name / roboflow_filename, output_dir / dataset_name / filename)
            image["file_name"] = filename

            side = None
            if "_right" in filename:
                side = "right"
            elif "_left" in filename:
                side = "left"
            else:
                raise AssertionError(f"Invalid filename: {filename}")

            identifier = filename.replace(f"_{side}", "").replace(".jpg", "")
            try:
                photo = dataset.get_photo(identifier)
            except Exception as exc:
                logger.error(f"Error getting photo {filename}: {exc}")
                continue

            ears = get_ears(photo, sam3, anchor_model)

            ear = None

            for e in ears:
                if e.side == side:
                    ear = e
                    break
            else:
                raise AssertionError(f"No matching {side} ear found for {filename}")

            # Update the coco annotation bbox to align with the ear xyxy AND the keypoints (in case of mismatch)
            try:
                annotation = next(a for a in coco_data["annotations"] if a["image_id"] == image["id"])
            except StopIteration:
                #print(f"Image {filename} has no ears.")
                continue

            if "keypoints" not in annotation:
                print(f"Image {filename} has no keypoints.")
                continue


            # Compute x and y buffer based on the ear width AND image coordinates
            crop_x = max(0, ear.xyxy[0] - (ear.xyxy[2] - ear.xyxy[0]) * 0.15)
            crop_y = max(0, ear.xyxy[1] - (ear.xyxy[3] - ear.xyxy[1]) * 0.15)
            x_buffer = ear.xyxy[0] - crop_x
            y_buffer = ear.xyxy[1] - crop_y


            x1 = min(x_buffer, annotation["keypoints"][0], annotation["keypoints"][3])
            y1 = min(y_buffer, annotation["keypoints"][1], annotation["keypoints"][4])
            x2 = max(ear.xyxy[2] - ear.xyxy[0] + x_buffer, annotation["keypoints"][0], annotation["keypoints"][3])
            y2 = max(ear.xyxy[3] - ear.xyxy[1] + y_buffer, annotation["keypoints"][1], annotation["keypoints"][4])
            width = x2 - x1
            height = y2 - y1

            annotation["bbox"] = [round(x1, 3), round(y1, 3), round(width, 3), round(height, 3)]
            annotation["area"] = round(width * height, 3)

            upper_keypoint = annotation["keypoints"][0:3] if annotation["keypoints"][1] > annotation["keypoints"][4] else annotation["keypoints"][3:6]
            lower_keypoint = annotation["keypoints"][0:3] if annotation["keypoints"][1] < annotation["keypoints"][4] else annotation["keypoints"][3:6]
            annotation["keypoints"] = [
                *upper_keypoint,
                *lower_keypoint,
            ]

            # ear_image = dataset.read_image(photo)
            # cv2.drawContours(ear_image, [ear.contour], -1, (0, 0, 255), 2)
            # ear_image = apply_crop(ear_image, (ear.xyxy[0] - x_buffer, ear.xyxy[1] - y_buffer, ear.xyxy[2] + x_buffer, ear.xyxy[3] + y_buffer))

            # cv2.rectangle(ear_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            # cv2.circle(ear_image, (int(annotation["keypoints"][0]), int(annotation["keypoints"][1])), 5, (0, 0, 255), -1)
            # cv2.circle(ear_image, (int(annotation["keypoints"][3]), int(annotation["keypoints"][4])), 5, (0, 0, 255), -1)
            # cv2.imshow(f"Image {image['id']}", ear_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


        with open(output_dir / dataset_name / "_annotations.coco.json", "w") as f:
            json.dump(coco_data, f, indent=4)

# Ultralytics default keypoint-detection augmentation hyperparameters.
DEFAULT_AUG_CONFIG = {
    "hsv_h": 0.01,
    "hsv_s": 0.3,
    "hsv_v": 0.3,
    "degrees": 5,
    "translate": 0,
    "scale": 0,
    "shear": 2,
    "perspective": 0,
    "flipud": 0,
    "fliplr": 0.5,
    "mosaic": 0.0,
    "mixup": 0,
    "copy_paste": 0,
}


SPLITS = ("train", "valid", "test")


def flip_augment(
    input_dir: str = "outputs/anchor_training_data",
    output_dir: str = "outputs/anchor_training_data_flipped",
    flip_splits: tuple[str, ...] = ("train",),
) -> None:
    """Write a COCO dataset where flip_splits contain every original image plus a horizontally-flipped
    copy; other splits are copied through unchanged. Expects the images/{split} + annotations/
    instances_{split}.json layout. Flips use the pixel-exact (W-1)-x convention to match cv2.flip;
    lower/upper keypoints keep their order (a vertical pair is flip-invariant), category is unchanged."""
    import shutil

    root, out = Path(input_dir), Path(output_dir)
    for split in SPLITS:
        with open(root / "annotations" / f"instances_{split}.json") as f:
            coco = json.load(f)
        img_src, img_dst = root / "images" / split, out / "images" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        for im in coco["images"]:
            shutil.copy2(img_src / im["file_name"], img_dst / im["file_name"])

        if split in flip_splits:
            anns_by_image: dict[int, list[dict]] = {}
            for ann in coco["annotations"]:
                anns_by_image.setdefault(ann["image_id"], []).append(ann)
            next_image_id = max(im["id"] for im in coco["images"]) + 1
            next_ann_id = max((a["id"] for a in coco["annotations"]), default=0) + 1

            flipped_images, flipped_anns = [], []
            for im in coco["images"]:
                w = im["width"]
                flip_name = f"{Path(im['file_name']).stem}_flip{Path(im['file_name']).suffix}"
                cv2.imwrite(str(img_dst / flip_name), cv2.flip(cv2.imread(str(img_src / im["file_name"])), 1))
                flipped_images.append({**im, "id": next_image_id, "file_name": flip_name})

                for ann in anns_by_image.get(im["id"], []):
                    x, y, bw, bh = ann["bbox"]
                    kpts = ann.get("keypoints", [])
                    flipped_kpts = []
                    for i in range(0, len(kpts), 3):
                        kx, ky, kv = kpts[i], kpts[i + 1], kpts[i + 2]
                        flipped_kpts += [(w - 1 - kx) if kv > 0 else kx, ky, kv]
                    flipped_anns.append({
                        **ann,
                        "id": next_ann_id,
                        "image_id": next_image_id,
                        "bbox": [max(0.0, w - 1 - x - bw), y, bw, bh],
                        "keypoints": flipped_kpts,
                    })
                    next_ann_id += 1
                next_image_id += 1

            coco["images"] += flipped_images
            coco["annotations"] += flipped_anns

        (out / "annotations").mkdir(parents=True, exist_ok=True)
        with open(out / "annotations" / f"instances_{split}.json", "w") as f:
            json.dump(coco, f, indent=2)
        logger.info(f"{split}: wrote {len(coco['images'])} images, {len(coco['annotations'])} annotations")


def restructure_to_coco(input_dir: str) -> None:
    """Modify a per-split COCO dataset ({split}/*.jpg + {split}/_annotations.coco.json) in place into
    the layout convert_coco expects: images/{split}/ + annotations/instances_{split}.json. Idempotent."""
    import shutil

    root = Path(input_dir)
    if (root / "annotations").exists():
        return
    (root / "annotations").mkdir(parents=True)
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        for jpg in (root / split).glob("*.jpg"):
            shutil.move(str(jpg), str(root / "images" / split / jpg.name))
        shutil.move(str(root / split / "_annotations.coco.json"), str(root / "annotations" / f"instances_{split}.json"))
        (root / split).rmdir()


def convert(input_dir: str = "outputs/anchor_training_data", output_dir: str = "outputs/anchor_training_data_yolo") -> None:
    """Restructure the per-split COCO dataset, convert it to YOLO pose format, place images alongside
    labels, write the pose dataset.yaml, and verify per the Ultralytics coco-to-yolo guide (step 6)."""
    import shutil

    from ultralytics.data.converter import convert_coco

    root, out = Path(input_dir), Path(output_dir)
    restructure_to_coco(input_dir)

    # convert_coco strips "instances_", so labels land in <save_dir>/labels/{train,valid,test}/.
    convert_coco(labels_dir=str(root / "annotations"), save_dir=str(out), use_keypoints=True, cls91to80=False)

    # YOLO expects labels/ to mirror images/; copy the images in and write the pose dataset.yaml.
    for split in SPLITS:
        shutil.copytree(root / "images" / split, out / "images" / split, dirs_exist_ok=True)
    (out / "dataset.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n"
        "kpt_shape: [2, 3]\n"
        "flip_idx: [0, 1]\n"
        "names:\n  0: ear\n"
    )

    # Verify Your Conversion: class IDs non-negative, normalized bbox coords within [0, 1].
    for label_file in (out / "labels").rglob("*.txt"):
        for line in label_file.read_text().strip().splitlines():
            parts = line.split()
            cls_id = int(parts[0])
            coords = [float(v) for v in parts[1:5]]
            assert cls_id >= 0, f"Negative class ID {cls_id} — category_id in your JSON may start from 0"
            assert all(0 <= v <= 1 for v in coords), f"Coordinates out of [0, 1] range: {coords}"


def train(cfg: dict, data: str = "dataset/anchors/data.yaml", model: str = "yolo26n-pose.pt", epochs: int = 100, imgsz: int = 640) -> None:
    """Fine-tune a pretrained YOLO pose model. The aug config keys are Ultralytics train
    args, so cfg is splatted straight in. plots=True (default) writes train_batch*.jpg
    (augmented batches with keypoints drawn) to the run directory."""
    from ultralytics import YOLO

    YOLO(model).train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        **cfg)

def main() -> None:
    load_dotenv()
    configure_logging(level="ERROR")

    train(DEFAULT_AUG_CONFIG)


if __name__ == "__main__":
    main()
