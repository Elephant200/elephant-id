from pathlib import Path

import cv2
import numpy as np

from elephant_id.ai import AnchorService, Sam3Service
from elephant_id.coding.analyzers.ears import Ear
from elephant_id.dataset import Dataset
from elephant_id.image.transforms import apply_crop, apply_mask


def find_log_contours(
    gray_image: np.ndarray,
    *,
    blur_kernel_size: int = 5,
    threshold: int = 20,
) -> tuple[np.ndarray, ...]:
    """Find contours from a thresholded Laplacian-of-Gaussian response."""
    gaussian = cv2.GaussianBlur(gray_image, (blur_kernel_size, blur_kernel_size), 0)
    laplacian = cv2.Laplacian(gaussian, cv2.CV_16S)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    _, binary = cv2.threshold(laplacian_abs, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return tuple(contours)

if __name__ == "__main__":
    from dotenv import load_dotenv

    from elephant_id.ai import Sam3Service
    from elephant_id.dataset import Dataset
    from elephant_id.log import configure_logging

    load_dotenv()
    configure_logging()

    dataset = Dataset(
        dataset_root=Path("dataset/elephants-alive/coded"),
        metadata_path=Path("dataset/elephants-alive/images.csv"),
    )

    sam3 = Sam3Service(dataset=dataset)
    anchor_model = AnchorService(dataset=dataset)

    photo = dataset.get_photo("Centaures_2018-11-24_10")

    detections = sam3.run(photo, "features")
    ear_detections = [detection for detection in detections if detection.class_name == "ear"]
    ears: list[Ear] = []
    for ear_detection in ear_detections:
        anchor_dets = anchor_model.run(photo, crop_xyxy=ear_detection.xyxy)
        if len(anchor_dets) == 0:
            print(f"No anchor detections found for ear {ear_detection.xyxy}")
            continue
        elif len(anchor_dets) > 1:
            print(f"Multiple anchor detections found for ear {ear_detection.xyxy}: {len(anchor_dets)}")
            anchor_dets = sorted(anchor_dets, key=lambda d: d.confidence, reverse=True)[0]
        ears.append(Ear(ear_detection, anchor_dets[0]))

    ear = min(ears, key=lambda e: e.area)

    # crop to ear
    image = apply_crop(dataset.read_image(photo), ear.xyxy)
    ear_mask = ear.get_mask()[int(ear.xyxy[1]):int(ear.xyxy[3]), int(ear.xyxy[0]):int(ear.xyxy[2])].copy()
    ear_only = apply_mask(image, ear_mask)
    cv2.imshow("Ear only", ear_only)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    lab = cv2.cvtColor(ear_only, cv2.COLOR_BGR2LAB)
    ear_grayscale, _, _ = cv2.split(lab)
    cv2.imshow("Ear grayscale", ear_grayscale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Preprocess by subtracting the background

    local_mean = cv2.GaussianBlur(ear_grayscale, (121, 121), 0)
    # Delete area outside mask
    local_mean[~ear_mask] = 0
    cv2.imshow("Local mean", local_mean)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    normalized_grayscale = cv2.absdiff(ear_grayscale, local_mean)
    cv2.imshow("Normalized grayscale", normalized_grayscale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    edges = cv2.Canny(normalized_grayscale, 125, 300)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    binary_copy = normalized_grayscale.copy()
    original_copy = image.copy()
    for contour in contours:
        cv2.drawContours(binary_copy, [contour], -1, (255, 255, 255), 1)
        cv2.drawContours(original_copy, [contour], -1, (0, 255, 0), 1)
    cv2.imshow("Contours", binary_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("edges.png", binary_copy)
    cv2.imwrite("original.png", original_copy)

    # Filter for connected contours


    # 3) Run Laplacian of Gaussian
    laplacian = cv2.Laplacian(normalized_grayscale, cv2.CV_16S)
    cv2.imshow("Laplacian", cv2.convertScaleAbs(laplacian))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours = find_log_contours(normalized_grayscale)
    for contour in contours:
        cv2.drawContours(ear_only, [contour], -1, (0, 255, 0), 2)
    cv2.imshow("Contours", ear_only)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
