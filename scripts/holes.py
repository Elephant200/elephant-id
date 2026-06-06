from pathlib import Path

import cv2
import numpy as np

from elephant_id.ai import AnchorService, Sam3Service
from elephant_id.coding.analyzers.ears import Ear
from elephant_id.dataset import Dataset
from elephant_id.image.transforms import apply_crop, apply_mask

EDGE_MARGIN_FRACTION = 0.015
ANCHOR_LINE_MARGIN_FRACTION = 0.2
MIN_CONNECTED_CONTOUR_PIXELS = 10


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


def trim_binary_mask(
    mask: np.ndarray,
    *,
    margin_fraction: float = EDGE_MARGIN_FRACTION,
) -> np.ndarray:
    """Return a binary mask with its nearest boundary band removed."""
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    if not 0.0 <= margin_fraction < 1.0:
        raise ValueError(f"margin_fraction must be in [0, 1), got {margin_fraction}")

    bool_mask = mask.astype(bool)
    if not bool_mask.any():
        raise ValueError("Cannot trim an empty mask")

    distances = cv2.distanceTransform(
        bool_mask.astype(np.uint8),
        cv2.DIST_L2,
        maskSize=5,
    )
    edge_margin = margin_fraction * max(mask.shape)
    return distances > edge_margin


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

    # Interesting photos:
    photos = {
        "Gap": "Gap_2019-11-20_02",
        "Centaures": "Centaures_2018-11-24_10",
        "Bloom": "Bloom_2016-06-06_08",
        "Intwandamela": "Intwandamela_2021-05-27_03",
    }
    photo = dataset.get_photo(photos["Gap"])

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
    image = apply_crop(dataset.read_image(photo), ear.xyxy)
    ear_mask = ear.mask[int(ear.xyxy[1]):int(ear.xyxy[3]), int(ear.xyxy[0]):int(ear.xyxy[2])].copy()
    ear_only = apply_mask(image, ear_mask)
    cv2.imshow("Ear only", ear_only)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    lab = cv2.cvtColor(ear_only, cv2.COLOR_BGR2LAB)
    ear_grayscale, _, _ = cv2.split(lab)
    cv2.imshow("Ear grayscale", ear_grayscale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    interior_mask = trim_binary_mask(ear_mask, margin_fraction=EDGE_MARGIN_FRACTION)
    edge_exclusion = ear_mask & ~interior_mask
    (x1, y1), (x2, y2) = ear.anchor_points
    crop_x1, crop_y1 = ear.xyxy[:2]
    x1 -= crop_x1
    x2 -= crop_x1
    y1 -= crop_y1
    y2 -= crop_y1

    yy, xx = np.indices(ear_mask.shape)
    anchor_line_margin = ANCHOR_LINE_MARGIN_FRACTION * max(ear_mask.shape)

    anchor_line_mask = np.zeros(ear_mask.shape, dtype=np.uint8)
    cv2.line(
        anchor_line_mask,
        (round(x1), round(y1)),
        (round(x2), round(y2)),
        color=1,
        thickness=1,
    )
    anchor_distances = cv2.distanceTransform(
        1 - anchor_line_mask,
        cv2.DIST_L2,
        maskSize=5,
    )
    anchor_line_exclusion = anchor_distances <= anchor_line_margin
    interior_mask &= ~anchor_line_exclusion

    upper_x, upper_y = min(((x1, y1), (x2, y2)), key=lambda point: point[1])
    ear_ys, ear_xs = np.where(ear_mask)
    middle_x = round((ear_xs.min() + ear_xs.max()) / 2)
    top_x = ear_xs[np.argmin(np.abs(ear_xs - middle_x))]
    top_y = ear_ys[ear_xs == top_x].min()
    if abs(top_x - upper_x) < 1e-6:
        top_line_exclusion = yy <= min(top_y, upper_y) + anchor_line_margin
    else:
        line_y = upper_y + ((top_y - upper_y) / (top_x - upper_x)) * (xx - upper_x)
        line_distance = np.abs(
            (top_y - upper_y) * xx
            - (top_x - upper_x) * yy
            + top_x * upper_y
            - top_y * upper_x
        )
        line_distance /= np.hypot(top_y - upper_y, top_x - upper_x)
        top_line_exclusion = (yy < line_y) | (line_distance <= anchor_line_margin)
    interior_mask &= ~top_line_exclusion

    edge_debug = ear_only.copy()
    edge_debug[edge_exclusion] = (0, 0, 255)
    edge_debug[anchor_line_exclusion & ear_mask] = (255, 0, 0)
    edge_debug[top_line_exclusion & ear_mask] = (0, 255, 0)
    cv2.line(
        edge_debug,
        (round(x1), round(y1)),
        (round(x2), round(y2)),
        color=(255, 255, 255),
        thickness=2,
    )
    cv2.circle(edge_debug, (round(x1), round(y1)), 4, (255, 255, 255), -1)
    cv2.circle(edge_debug, (round(x2), round(y2)), 4, (255, 255, 255), -1)
    cv2.imshow("Edge exclusion", edge_debug)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Preprocess using CLAHE
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # ear_grayscale_clahe = clahe.apply(ear_grayscale)
    # cv2.imshow("CLAHE processed", ear_grayscale_clahe)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Preprocess by subtracting the background
    # sigma = 15
    # local_mean = cv2.GaussianBlur(ear_grayscale, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # local_mean[~ear_mask] = 0 # Delete area outside mask
    # cv2.imshow(f"Local mean ({sigma*6+1}, {sigma*6+1})", local_mean)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # normalized_grayscale = cv2.absdiff(ear_grayscale, local_mean)
    # cv2.imshow("Normalized grayscale", normalized_grayscale)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    normalized_grayscale = ear_grayscale

    edges = cv2.Canny(normalized_grayscale, 50, 100)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    interior_edges = edges.copy()
    interior_edges[~interior_mask] = 0
    cv2.imshow("Interior edges", interior_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply morpholgical closing to canny edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_edges = cv2.morphologyEx(interior_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow("Closed edges", closed_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, _ = cv2.findContours(
        closed_edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    original_contours = normalized_grayscale.copy()
    for contour in contours:
        cv2.drawContours(original_contours, [contour], -1, (255, 255, 255), 1)
    cv2.imshow("Original contours", original_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    min_area = 50.0
    min_circularity = 0.3

    filtered_contours = []
    rejected_by_area = []
    rejected_by_circularity = []
    rejected_by_weird_score = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter == 0: # This should never happen, but just in case of divide by zero errors.
            continue

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

        if area < min_area:
            verdict = "Rejected by area"
            rejected_by_area.append(contour)
        elif circularity < min_circularity:
            verdict = "Rejected by circularity"
            rejected_by_circularity.append(contour)
        else:
            verdict = "Accepted"
            filtered_contours.append(contour)
        weird_score = circularity ** 2 * area
        if weird_score < 15:
            rejected_by_weird_score.append(contour)
        print(f"{verdict:<25s}| Circularity: {circularity:<10.4f}| Area: {area:<10.3f}| Perimeter: {perimeter:<10.3f}| Weird Score: {weird_score:<10.5}")

    print(f"Filtered contours: {len(filtered_contours)}")
    print(f"Rejected by area: {len(rejected_by_area)}")
    print(f"Rejected by circularity: {len(rejected_by_circularity)}")
    print(f"Rejected by weird score: {len(rejected_by_weird_score)}")

    binary_copy = cv2.cvtColor(normalized_grayscale, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(binary_copy, filtered_contours, -1, (255, 255, 255), 1)
    cv2.drawContours(binary_copy, rejected_by_weird_score, -1, (255, 255, 0), 1)
    cv2.drawContours(binary_copy, rejected_by_area, -1, (0, 0, 255), 1)
    cv2.drawContours(binary_copy, rejected_by_circularity, -1, (0, 255, 0), 1)

    cv2.imshow("Contours", binary_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    original_copy = image.copy()
    cv2.drawContours(original_copy, filtered_contours, -1, (0, 0, 255), 1)
    cv2.imshow("Contours on Original", original_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 3) Run Laplacian of Gaussian
    # laplacian = cv2.Laplacian(normalized_grayscale, cv2.CV_16S)
    # cv2.imshow("Laplacian", cv2.convertScaleAbs(laplacian))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # contours = find_log_contours(normalized_grayscale)
    # for contour in contours:
    #     cv2.drawContours(ear_only, [contour], -1, (0, 255, 0), 2)
    # cv2.imshow("Contours", ear_only)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
