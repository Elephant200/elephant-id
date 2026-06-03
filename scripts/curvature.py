from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from elephant_id.ai import Detection
from elephant_id.ai.sam3 import Sam3Service
from elephant_id.coding.curvature import (
    contour_max_dimension,
    oriented_curvature,
    resample2d,
)
from elephant_id.dataset import Dataset
from elephant_id.image.bgr import BgrImage
from elephant_id.image.masks import decode_rle_mask
from elephant_id.log import configure_logging
from elephant_id.visualize import visualize_predictions

PHOTO_ID = "Adam_2011-03-31_03"
ANCHOR_PRESETS = {
    "Adam_2011-03-31_03": {
        "start": (917, 258),
        "end": (1078, 953),
    },
    "Adam_2021-11-18_05": {
        "start": (2414, 1011),
        "end": (2516, 2485),
    },
    "Ripley_2008-06-25_06": {
        "start": (1448, 372),
        "end": (1481, 751),
    }
}

CURVATURE_POINTS = 1024
MARKER_STEP = 32
SCALES = np.array([0.02, 0.04, 0.06, 0.08, 0.10], dtype=np.float32)
WEIGHTS = np.array([0.6, 0.9, 1.0, 0.9, 0.6], dtype=np.float32)
MATPLOTLIB_COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (128, 128, 128),
    (211, 211, 211),
)


def get_largest_external_contour(prediction: Detection) -> np.ndarray:
    mask = decode_rle_mask(prediction.rle_mask)
    mask_u8 = np.ascontiguousarray(mask.astype(np.uint8) * 255)

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError("No contour found for prediction mask")

    return max(contours, key=cv2.contourArea)


def squared_distance_to_contour(
    contour: np.ndarray,
    point: tuple[int, int],
) -> float:
    points = contour[:, 0, :]
    target = np.array(point)
    return float(np.min(np.sum((points - target) ** 2, axis=1)))


def score_contour_for_anchors(
    contour: np.ndarray,
    start_point: tuple[int, int],
    end_point: tuple[int, int],
) -> float:
    return squared_distance_to_contour(
        contour,
        start_point,
    ) + squared_distance_to_contour(
        contour,
        end_point,
    )


def select_ear_by_anchors(
    predictions: list[Detection],
    start_point: tuple[int, int],
    end_point: tuple[int, int],
) -> tuple[Detection, np.ndarray]:
    ears = [prediction for prediction in predictions if prediction.class_name == "ear"]
    if not ears:
        raise ValueError("No ear predictions found")

    candidates: list[tuple[float, Detection, np.ndarray]] = []
    for ear in ears:
        contour = get_largest_external_contour(ear)
        score = score_contour_for_anchors(contour, start_point, end_point)
        candidates.append((score, ear, contour))

    candidates.sort(key=lambda item: item[0])
    print(f"Ear candidates: {len(candidates)}")
    for i, (score, ear, _) in enumerate(candidates):
        print(
            f"  {i}: score={score:.1f}, confidence={ear.confidence:.3f}, "
        )

    _, selected_ear, selected_contour = candidates[0]
    return selected_ear, selected_contour


def anchors_for_photo(photo_id: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if photo_id not in ANCHOR_PRESETS:
        available = ", ".join(sorted(ANCHOR_PRESETS))
        raise KeyError(
            f"No anchor preset for {photo_id!r}. "
            f"Add one to ANCHOR_PRESETS. Available presets: {available}"
        )

    preset = ANCHOR_PRESETS[photo_id]
    return preset["start"], preset["end"]


def draw_contour(
    image: BgrImage,
    contour: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
) -> BgrImage:
    """
    Draw a contour on an image. (color is RGB)
    """
    output = image.copy()
    cv2.drawContours(output, [contour], contourIdx=-1, color=color[::-1], thickness=thickness)
    return output


def draw_polyline(
    image: BgrImage,
    points: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
    marker_step: int | None = None,
) -> BgrImage:
    output = image.copy()
    points_i32 = points.astype(np.int32)
    cv2.polylines(
        output,
        [points_i32.reshape((-1, 1, 2))],
        isClosed=False,
        color=color[::-1], # RGB -> BGR
        thickness=thickness,
    )
    if marker_step is not None:
        for idx in range(0, len(points_i32), marker_step):
            x, y = points_i32[idx]
            cv2.circle(output, (int(x), int(y)), 4, (255, 255, 0)[::-1], -1) # RGB -> BGR
            cv2.putText(
                output,
                str(idx),
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                str(idx),
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0)[::-1], # RGB -> BGR
                1,
                cv2.LINE_AA,
            )
    return output


def crop_to_contour(
    image: BgrImage,
    contour: np.ndarray,
    padding: int = 80,
) -> tuple[BgrImage, np.ndarray]:
    points = contour.astype(np.float32)
    min_x, min_y = np.floor(points.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(points.max(axis=0)).astype(int)

    left = max(0, min_x - padding)
    top = max(0, min_y - padding)
    right = min(image.shape[1], max_x + padding)
    bottom = min(image.shape[0], max_y + padding)

    cropped = image[top:bottom, left:right].copy()
    shifted = points - np.array([left, top], dtype=np.float32)
    return cropped, shifted


def draw_radius_scale(
    image: BgrImage,
    radii: np.ndarray,
    scales: np.ndarray,
    colors: tuple[tuple[int, int, int], ...],
    ) -> BgrImage:
    output = image.copy()
    max_radius = int(np.ceil(float(np.max(radii))))
    x = image.shape[1] - max_radius - 64
    y = image.shape[0] - max_radius - 24

    for radius, color in zip(radii, colors, strict=False):
        radius_int = round(radius)
        cv2.circle(
            output,
            (x, y),
            radius_int,
            color[::-1], # RGB -> BGR
            2,
            cv2.LINE_AA,
        )

    cv2.circle(output, (x, y), 2, (0, 0, 0), -1, cv2.LINE_AA)

    return output


def _closed_path(points: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    if start_idx <= end_idx:
        return points[start_idx : end_idx + 1]
    return np.concatenate([points[start_idx:], points[: end_idx + 1]])


def remove_head_connection(
    contour: np.ndarray,
    start_point: tuple[int, int],
    end_point: tuple[int, int],
) -> np.ndarray:
    points = contour[:, 0, :]
    start = np.array(start_point)
    end = np.array(end_point)

    start_idx = int(np.argmin(np.sum((points - start) ** 2, axis=1)))
    end_idx = int(np.argmin(np.sum((points - end) ** 2, axis=1)))

    forward_path = _closed_path(points, start_idx, end_idx)
    backward_path = _closed_path(points, end_idx, start_idx)[::-1]

    # The manually selected head/ear attachment is guaranteed to be the shorter
    # anchor-to-anchor path.
    if len(forward_path) >= len(backward_path):
        kept_points = forward_path
        removed_points = backward_path
    else:
        kept_points = backward_path
        removed_points = forward_path

    # CurvRank treats left/right ear contours as consistently ordered
    # top-to-bottom before curvature is computed.
    if kept_points[0, 1] > kept_points[-1, 1]:
        kept_points = kept_points[::-1]

    print(f"Start snapped to contour[{start_idx}]: {tuple(points[start_idx])}")
    print(f"End snapped to contour[{end_idx}]: {tuple(points[end_idx])}")
    print(f"Forward path points: {len(forward_path)}")
    print(f"Backward path points: {len(backward_path)}")
    print(f"Removed head connection points: {len(removed_points)}")
    print(f"Kept contour points: {len(kept_points)}")

    return kept_points


def plot_integral_curvature(
    curvature: np.ndarray,
    marker_step: int,
) -> None:
    """Plot curvature.

    Args:
        curvature: Curvature values, shaped ``(num_points,)``.
        marker_step: Subsample markers along the contour index.

    Returns:
        None.
    """
    plt.figure(figsize=(19.2, 4.8))

    plt.plot(
        curvature,
        color="black",
        label="Curvature",
    )

    plt.axhline(0.5, color="0.65", linewidth=1, alpha=0.75, zorder=1)
    plt.title("Elephant ear")
    plt.xlabel("Contour point")
    plt.ylabel("Curvature")
    ymin = 0.1
    ymax = 0.7
    plt.ylim(ymin, ymax)
    plt.xticks(np.arange(0, curvature.shape[0], marker_step), rotation=90)
    plt.yticks(np.arange(ymin, ymax, 0.025))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    load_dotenv()
    configure_logging()
    dataset = Dataset(
        dataset_root=Path("dataset/elephants-alive/coded"),
        metadata_path=Path("dataset/elephants-alive/images.csv"),
    )
    photo = dataset.get_photo(PHOTO_ID)
    start_point, end_point = anchors_for_photo(PHOTO_ID)
    image = dataset.read_image(photo)

    sam3 = Sam3Service(
        dataset=dataset,
    )

    segmentation = sam3.run(photo, "features")
    ear, contour = select_ear_by_anchors(segmentation, start_point, end_point)
    ears = [ear]
    print(ear)

    ear_contour = remove_head_connection(
        contour,
        start_point,
        end_point,
    )
    resampled_contour = resample2d(ear_contour, num_points=CURVATURE_POINTS)

    radii = SCALES * contour_max_dimension(resampled_contour)
    curvature = oriented_curvature(resampled_contour, radii, weights=WEIGHTS)
    print(f"Curvature min/max: {curvature.min():.4f}, {curvature.max():.4f}")

    visualized_image = visualize_predictions(image, ears)
    cropped_image, cropped_contour = crop_to_contour(visualized_image, resampled_contour)
    contour_image = draw_polyline(
        cropped_image,
        cropped_contour,
        marker_step=MARKER_STEP,
    )
    contour_image = draw_radius_scale(contour_image, radii, SCALES, MATPLOTLIB_COLORS)
    cv2.imshow("Contour Image", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plot_integral_curvature(
        curvature,
        marker_step=MARKER_STEP,
    )
    print(curvature.shape)
