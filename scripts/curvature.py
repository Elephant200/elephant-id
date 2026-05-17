import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from elephant_id.ai.sam3 import Sam3Service
from elephant_id.dataset import Dataset
from elephant_id.visualize import decode_rle_mask, visualize_predictions

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
MATPLOTLIB_COLORS = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (128, 128, 128),
    (211, 211, 211),
)


def get_largest_external_contour(prediction: dict) -> np.ndarray:
    mask = decode_rle_mask(prediction["rle_mask"])
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
    predictions: list[dict],
    start_point: tuple[int, int],
    end_point: tuple[int, int],
) -> tuple[dict, np.ndarray]:
    ears = [obj for obj in predictions if obj["class"].strip() == "ear"]
    if not ears:
        raise ValueError("No ear predictions found")

    candidates = []
    for ear in ears:
        contour = get_largest_external_contour(ear)
        score = score_contour_for_anchors(contour, start_point, end_point)
        candidates.append((score, ear, contour))

    candidates.sort(key=lambda item: item[0])
    print(f"Ear candidates: {len(candidates)}")
    for i, (score, ear, _) in enumerate(candidates):
        print(
            f"  {i}: score={score:.1f}, confidence={ear.get('confidence', 0):.3f}, "
            f"id={ear.get('detection_id', 'unknown')}"
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
    image: Image.Image,
    contour: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
) -> Image.Image:
    output = np.array(image.convert("RGB")).copy()
    cv2.drawContours(output, [contour], contourIdx=-1, color=color, thickness=thickness)
    return Image.fromarray(output)


def draw_polyline(
    image: Image.Image,
    points: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
    marker_step: int | None = None,
) -> Image.Image:
    output = np.array(image.convert("RGB")).copy()
    points_i32 = points.astype(np.int32)
    cv2.polylines(
        output,
        [points_i32.reshape((-1, 1, 2))],
        isClosed=False,
        color=color,
        thickness=thickness,
    )
    if marker_step is not None:
        for idx in range(0, len(points_i32), marker_step):
            x, y = points_i32[idx]
            cv2.circle(output, (int(x), int(y)), 4, (255, 255, 0), -1)
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
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
    return Image.fromarray(output)


def crop_to_contour(
    image: Image.Image,
    contour: np.ndarray,
    padding: int = 80,
) -> tuple[Image.Image, np.ndarray]:
    points = contour.astype(np.float32)
    min_x, min_y = np.floor(points.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(points.max(axis=0)).astype(int)

    left = max(0, min_x - padding)
    top = max(0, min_y - padding)
    right = min(image.width, max_x + padding)
    bottom = min(image.height, max_y + padding)

    cropped = image.crop((left, top, right, bottom))
    shifted = points - np.array([left, top], dtype=np.float32)
    return cropped, shifted


def draw_radius_scale(
    image: Image.Image,
    radii: np.ndarray,
    scales: np.ndarray,
    colors: tuple[tuple[int, int, int], ...],
) -> Image.Image:
    output = np.array(image.convert("RGB")).copy()
    max_radius = int(np.ceil(float(np.max(radii))))
    x = image.width - max_radius - 64
    y = image.height - max_radius - 24

    for radius, color in zip(radii, colors, strict=False):
        radius_int = round(radius)
        cv2.circle(
            output,
            (x, y),
            radius_int,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.circle(output, (x, y), 2, (0, 0, 0), -1, cv2.LINE_AA)

    return Image.fromarray(output)


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


def resample2d(points: np.ndarray, num_points: int) -> np.ndarray:
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    arc_lengths = np.concatenate([[0], np.cumsum(distances)])
    if arc_lengths[-1] == 0:
        raise ValueError("Cannot resample a zero-length contour")

    sample_lengths = np.linspace(0, arc_lengths[-1], num_points)
    x = np.interp(sample_lengths, arc_lengths, points[:, 0])
    y = np.interp(sample_lengths, arc_lengths, points[:, 1])
    return np.column_stack([x, y])


def rotate(radians: float) -> np.ndarray:
    rotation = np.eye(3)
    rotation[0, 0] = np.cos(radians)
    rotation[1, 1] = np.cos(radians)
    rotation[0, 1] = np.sin(radians)
    rotation[1, 0] = -np.sin(radians)
    return rotation


def reorient(points: np.ndarray, theta: float, center: np.ndarray) -> np.ndarray:
    matrix = rotate(theta)
    points_translated = points - center
    points_augmented = np.hstack(
        (points_translated, np.ones((points.shape[0], 1)))
    )
    points_transformed = np.dot(matrix, points_augmented.T).T[:, :2]
    return points_transformed + center


def oriented_curvature(contour: np.ndarray, radii: np.ndarray) -> np.ndarray:
    curvature = np.zeros((len(radii), contour.shape[0]), dtype=np.float32)

    for i, (x, y) in enumerate(contour):
        center = np.array([x, y])
        distances = ((contour - center) ** 2).sum(axis=1)
        inside = distances[:, np.newaxis] <= radii * radii

        for j, radius in enumerate(radii):
            curve = contour[inside[:, j]]
            if curve.shape[0] == 1:
                curv = 0.5
            else:
                normal = curve[-1] - curve[0]
                theta = np.arctan2(normal[1], normal[0])

                curve_p = reorient(curve, theta, center)
                center_p = np.squeeze(reorient(center[None], theta, center))

                lower = center_p - radius
                upper = center_p + radius
                curve_p = np.clip(curve_p, lower, upper)

                area = np.trapezoid(curve_p[:, 1] - lower[1], curve_p[:, 0], axis=0)
                curv = area / ((2 * radius) ** 2)

            curvature[j, i] = curv

    return curvature


def plot_integral_curvature(
    curvature: np.ndarray,
    mean_curvature: np.ndarray,
    radii: np.ndarray,
    marker_step: int,
) -> None:
    """Plot multi-scale curvature and several Gaussian-weighted means on one axes.

    Args:
        curvature: Per-radius curvature values, shape ``(len(radii), num_points)``.
        mean_curvature: Mean curvature values, shape ``(num_points)``.
        radii: Physical radius per scale (for legend only).
        marker_step: Subsample markers along the contour index.

    Returns:
        None.
    """
    plt.figure(figsize=(19.2, 4.8))
    marker_indices = np.arange(0, curvature.shape[1], marker_step)
    for i, radius in enumerate(radii):
        plt.plot(
            curvature[i],
            color=np.array(MATPLOTLIB_COLORS[i]) / 255,
            marker="o",
            markevery=marker_indices,
            markersize=3,
            label=f"r={radius:g}",
            zorder=2,
        )

    plt.plot(
        mean_curvature,
        color="black",
        marker="o",
        markevery=marker_indices,
        markersize=3,
        label="Mean",
    )

    plt.axhline(0.5, color="0.65", linewidth=1, alpha=0.75, zorder=1)
    plt.title("Elephant ear")
    plt.xlabel("Contour point")
    plt.ylabel("Curvature")
    ymin = 0.1
    ymax = 0.7
    plt.ylim(ymin, ymax)
    plt.xticks(marker_indices, rotation=90)
    plt.grid(axis="x", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_weighted_mean_curvature_configs(
    curvature: np.ndarray,
    scales: np.ndarray,
    weight_configs: list[tuple[float, float]],
    marker_step: int,
) -> None:
    """Plot Gaussian-weighted mean curvature for several ``(mean, std)`` pairs.

    Args:
        curvature: Per-radius curvature values, shape ``(len(scales), num_points)``.
        scales: Normalized scale values matching ``curvature`` rows (same as ``SCALES``).
        weight_configs: ``(weight_mean_scale, weight_std_dev)`` pairs for the Gaussian weights.
        marker_step: Subsample markers along the contour index.

    Returns:
        None.
    """
    plt.figure(figsize=(19.2, 4.8))
    marker_indices = np.arange(0, curvature.shape[1], marker_step)
    ax = plt.gca()

    for idx, (mean_scale, std_dev) in enumerate(weight_configs):
        if std_dev <= 0:
            raise ValueError("weight std dev must be positive")
        weights = np.exp(-0.5 * ((scales - mean_scale) / std_dev) ** 2)
        mean_curve = np.average(curvature, axis=0, weights=weights)
        color = np.array(MATPLOTLIB_COLORS[idx % len(MATPLOTLIB_COLORS)]) / 255
        plt.plot(
            mean_curve,
            color=color,
            marker="o",
            markevery=marker_indices,
            markersize=3,
            label=f"mean={mean_scale:g}, std={std_dev:g}",
            zorder=2,
        )

    plt.axhline(0.5, color="0.65", linewidth=1, alpha=0.75, zorder=1)
    plt.title("Weighted mean curvature (Gaussian scale weights)")
    plt.xlabel("Contour point")
    plt.ylabel("Curvature")
    ymin = 0.1
    ymax = 0.7
    plt.ylim(ymin, ymax)
    plt.yticks(np.arange(ymin, ymax + 0.025, 0.025))
    plt.xticks(marker_indices, rotation=90)
    plt.grid(axis="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def contour_max_dimension(contour: np.ndarray) -> float:
    minimum = contour.min(axis=0)
    maximum = contour.max(axis=0)
    return float(np.max(maximum - minimum))


if __name__ == "__main__":
    load_dotenv()
    dataset = Dataset(
        dataset_root=Path("dataset/elephants-alive/coded"),
        metadata_path=Path("dataset/elephants-alive/images.csv"),
    )
    photo = dataset.get_photo(PHOTO_ID)
    start_point, end_point = anchors_for_photo(PHOTO_ID)
    image = dataset.read_image(photo)

    sam3 = Sam3Service(
        api_key=os.getenv("ROBOFLOW_API_KEY"),
        dataset=dataset,
    )

    segmentation = sam3.run(photo, "features")["predictions"]
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
    curvature = oriented_curvature(resampled_contour, radii)
    print(f"Curvature min/max: {curvature.min():.4f}, {curvature.max():.4f}")

    visualized_image = visualize_predictions(image, ears)
    cropped_image, cropped_contour = crop_to_contour(visualized_image, resampled_contour)
    contour_image = draw_polyline(
        cropped_image,
        cropped_contour,
        marker_step=MARKER_STEP,
    )
    contour_image = draw_radius_scale(contour_image, radii, SCALES, MATPLOTLIB_COLORS)
    contour_image.show()

    weights = [
        (0.06, 0.02),
        (0.06, 0.04),
        (0.07, 0.04),
        (0.08, 0.04),
    ]
    #weights = np.exp(-0.5 * ((SCALES - we) / WEIGHT_STD_DEV) ** 2)
    #mean_curvature = np.average(curvature, axis=0, weights=weights)
    plot_weighted_mean_curvature_configs(
        curvature,
        SCALES,
        weights,
        marker_step=MARKER_STEP,
    )
    # plot_integral_curvature(
    #     curvature,
    #     mean_curvature,
    #     radii,
    #     marker_step=MARKER_STEP,
    # )
    print(curvature.shape)
