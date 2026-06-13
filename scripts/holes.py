from pathlib import Path

import cv2
import numpy as np

from elephant_id.ai import AnchorService, Sam3Service
from elephant_id.coding.ears import AnchoredEar
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
        "Nguyen": "Nguyen_2012-08-02_07",
        "Scar": "Scar_2010-11-30_08",
        "Delani": "Delani_2017-10-09_06",
    }
    photo = dataset.get_photo(photos["Gap"])

    detections = sam3.run(photo, "features")
    ear_detections = [detection for detection in detections if detection.class_name == "ear"]
    ears: list[AnchoredEar] = []
    for ear_detection in ear_detections:
        anchor_dets = anchor_model.run(photo, crop_xyxy=ear_detection.xyxy)
        if len(anchor_dets) == 0:
            print(f"No anchor detections found for ear {ear_detection.xyxy}")
            continue
        elif len(anchor_dets) > 1:
            print(f"Multiple anchor detections found for ear {ear_detection.xyxy}: {len(anchor_dets)}")
            anchor_dets = sorted(anchor_dets, key=lambda d: d.confidence, reverse=True)[0]
        ears.append(AnchoredEar(ear_detection, anchor_dets[0]))

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
    sigma = 15
    local_mean = cv2.GaussianBlur(ear_grayscale, (0, 0), sigmaX=sigma, sigmaY=sigma)
    local_mean[~ear_mask] = 0 # Delete area outside mask
    cv2.imshow(f"Local mean ({sigma*6+1}, {sigma*6+1})", local_mean)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    normalized_grayscale = cv2.absdiff(ear_grayscale, local_mean)
    cv2.imshow("Normalized grayscale", normalized_grayscale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # normalized_grayscale = ear_grayscale

    edges = cv2.Canny(normalized_grayscale, 50, 125)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    interior_edges = edges.copy()
    interior_edges[~interior_mask] = 0
    cv2.imshow("Interior edges", interior_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Apply morpholgical closing to canny edges
    # Determine kernel size based on ear area
    close_kernel_size = round(np.sqrt(ear.area) * 0.0025) * 2 + 1
    print(f"Close kernel size: {close_kernel_size}")
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
    closed_edges = cv2.morphologyEx(interior_edges, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    cv2.imshow("Closed edges", closed_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Re-open with smaller kernel to denoise
    open_kernel_size = round(np.sqrt(ear.area) * 0.00125) * 2 + 1
    print(f"Open kernel size: {open_kernel_size}")
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
    closed_edges = cv2.morphologyEx(closed_edges, cv2.MORPH_OPEN, open_kernel, iterations=1)
    cv2.imshow("Denoised edges", closed_edges)
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

    # Filter by area / circularity composite and color difference
    lab_float = cv2.cvtColor(ear_only.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)

    min_area = 0.00003 * ear.area
    min_circularity = 0.4
    min_composite = min_area * 0.75
    print(f"Min area: {min_area:.5f}, Min circularity: {min_circularity:.4f}, Min composite: {min_composite:.5f}")

    filtered_contours = []
    rejected_by_area = []
    rejected_by_circularity = []
    rejected_by_composite = []
    rejected_by_color_difference = []
    for contour in contours:
        done = False
        area = cv2.contourArea(contour)

        if area < min_area:
            verdict = "Rejected by area"
            rejected_by_area.append(contour)
            done = True

        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter == 0: # This should never happen, but just in case of divide by zero errors.
            continue

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        if not done and circularity < min_circularity:
            verdict = "Rejected by circularity"
            rejected_by_circularity.append(contour)
            done = True

        composite = circularity ** 2 * area
        if not done and composite < min_composite:
            verdict = "Rejected by composite"
            rejected_by_composite.append(contour)
            done = True

        # Compute color difference from immediate surrounding area
        hole_mask = np.zeros(ear_mask.shape, dtype=np.uint8)
        cv2.drawContours(hole_mask, [contour], -1, 1, cv2.FILLED)

        hole_radius = np.sqrt(area / np.pi)

        surrounding_radius = max(3, round(hole_radius * 3.0))
        surrounding_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (surrounding_radius * 2 + 1, surrounding_radius * 2 + 1))

        outer_buffer_radius = max(1, round(hole_radius * 1.0))
        outer_buffer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_buffer_radius * 2 + 1, outer_buffer_radius * 2 + 1))
        inner_buffer_radius = max(1, round(hole_radius * 0.5))
        inner_buffer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_buffer_radius * 2 + 1, inner_buffer_radius * 2 + 1))

        area_mask = cv2.dilate(hole_mask, surrounding_kernel, iterations=1)
        exclusion_mask = cv2.dilate(hole_mask, outer_buffer_kernel, iterations=1)
        surrounding_mask = area_mask.astype(bool) & ~exclusion_mask.astype(bool) & ear_mask.astype(bool)

        hole_core = cv2.erode(hole_mask, inner_buffer_kernel, iterations=1).astype(bool)

        hole_color = np.mean(lab_float[hole_core], axis=0)
        surrounding_color = np.mean(lab_float[surrounding_mask], axis=0)
        color_difference = np.linalg.norm(hole_color - surrounding_color)

        if not done and color_difference < 10:
            verdict = "Rejected by color"
            rejected_by_color_difference.append(contour)
            done = True

        if not done:
            verdict = "Accepted"
            filtered_contours.append(contour)

        print(f"{verdict:<25s}| Circularity: {circularity:<13.4f}| Area: {area:<8.1f}({area/ear.area:.4%})| Perimeter: {perimeter:<10.3f}| Composite: {composite:<10.5}| Color difference: {color_difference:<10.5}")

    print(f"Filtered contours: {len(filtered_contours)}")
    print(f"Rejected by area: {len(rejected_by_area)}")
    print(f"Rejected by circularity: {len(rejected_by_circularity)}")
    print(f"Rejected by composite: {len(rejected_by_composite)}")
    print(f"Rejected by color difference: {len(rejected_by_color_difference)}")

    for contour in filtered_contours:
        hole_mask = np.zeros(ear_mask.shape, dtype=np.uint8)
        cv2.drawContours(hole_mask, [contour], -1, 1, cv2.FILLED)

        hole_radius = np.sqrt(area / np.pi)

        surrounding_radius = max(3, round(hole_radius * 3.0))
        surrounding_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (surrounding_radius * 2 + 1, surrounding_radius * 2 + 1))

        outer_buffer_radius = max(1, round(hole_radius * 1.0))
        outer_buffer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_buffer_radius * 2 + 1, outer_buffer_radius * 2 + 1))
        inner_buffer_radius = max(1, round(hole_radius * 0.5))
        inner_buffer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_buffer_radius * 2 + 1, inner_buffer_radius * 2 + 1))

        area_mask = cv2.dilate(hole_mask, surrounding_kernel, iterations=1)
        exclusion_mask = cv2.dilate(hole_mask, outer_buffer_kernel, iterations=1)
        surrounding_mask = area_mask.astype(bool) & ~exclusion_mask.astype(bool) & ear_mask.astype(bool)

        hole_core = cv2.erode(hole_mask, inner_buffer_kernel, iterations=1).astype(bool)

        hole_color = np.mean(lab_float[hole_core], axis=0)
        surrounding_color = np.mean(lab_float[surrounding_mask], axis=0)
        color_difference = np.linalg.norm(hole_color - surrounding_color)
        print(f"Color difference: {color_difference} | Inner color: {hole_color} | Surrounding color: {surrounding_color}")

        # Show the surrounding area in red and the hole core in green
        color_difference_copy = ear_only.copy()
        color_difference_copy[surrounding_mask] = (0, 0, 255)
        color_difference_copy[hole_core] = (0, 255, 0)
        cv2.imshow("Color difference", color_difference_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    binary_copy = cv2.cvtColor(normalized_grayscale, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(binary_copy, filtered_contours, -1, (255, 255, 255), 1)
    cv2.drawContours(binary_copy, rejected_by_composite, -1, (255, 255, 0), 1)
    cv2.drawContours(binary_copy, rejected_by_area, -1, (0, 0, 255), 1)
    cv2.drawContours(binary_copy, rejected_by_circularity, -1, (0, 255, 0), 1)
    cv2.drawContours(binary_copy, rejected_by_color_difference, -1, (255, 0, 255), 1)

    cv2.imshow("Contours", binary_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    original_copy = image.copy()
    cv2.drawContours(original_copy, filtered_contours, -1, (0, 0, 255), 2)
    # circle the area around the contour in green
    for contour in filtered_contours:
        area = cv2.contourArea(contour)
        radius = np.sqrt(area / np.pi) * 3.0
        center = np.mean(contour, axis=0)[0]
        cv2.circle(original_copy, (int(center[0]), int(center[1])), int(radius), (0, 255, 0), 1)
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
