from pathlib import Path

import cv2
import numpy as np

from elephant_id.ai import AnchorService, Sam3Service
from elephant_id.coding.analyzers.ears import Ear
from elephant_id.dataset import Dataset
from elephant_id.image.transforms import apply_mask


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

    photo = dataset.get_photo("Bloom_2016-06-06_08")

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

    ear = max(ears, key=lambda e: e.area)

    # crop to ear
    ear_only = apply_mask(dataset.read_image(photo), ear.get_mask(), crop=True)
    cv2.imshow("Ear only", ear_only)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ear_only_gray = 255 - cv2.cvtColor(ear_only, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Ear only gray (inverted)", ear_only_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Test different hole finding algorithms

    # 1) Canny edge detection

    # Run Canny edge detection, then find contours
    blurred = cv2.GaussianBlur(ear_only, (3, 3), 0)
    edges = cv2.Canny(blurred, 75, 200)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    ear_only_copy = ear_only.copy()
    for contour in contours:
        cv2.drawContours(ear_only_copy, [contour], -1, (0, 255, 0), 2)
    cv2.imshow("Contours", ear_only_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 2)  Run blob finding (pretty bad)
    # params = cv2.SimpleBlobDetector_Params()

    # params.filterByArea = True
    # params.minArea = 10

    # detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(ear_only_gray)
    # print(len(keypoints))
    # for keypoint in keypoints:
    #     cv2.circle(ear_only, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size), (0, 255, 0), 5)
    # cv2.imshow("Blobs", ear_only)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 3) Run Laplacian of Gaussian
    # gaussian = cv2.GaussianBlur(ear_only_gray, (5, 5), 0)
    # cv2.imshow("Gaussian", gaussian)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # laplacian = cv2.Laplacian(gaussian, cv2.CV_16S)
    # cv2.imshow("Laplacian", cv2.convertScaleAbs(laplacian))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # contours = find_log_contours(ear_only_gray)
    # for contour in contours:
    #     cv2.drawContours(ear_only, [contour], -1, (0, 255, 0), 2)
    # cv2.imshow("Contours", ear_only)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
