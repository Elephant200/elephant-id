from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from elephant_id.ai import AnchorService, Sam3Service
from elephant_id.coding.analyzers.ears import Ear
from elephant_id.coding.curvature import (
    contour_max_dimension,
    oriented_curvature,
)
from elephant_id.dataset import Dataset
from elephant_id.image.bgr import BgrImage
from elephant_id.image.transforms import apply_crop, apply_mask
from elephant_id.log import configure_logging

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
    photo = dataset.get_photo("Bloom_2016-06-06_08")
    image = dataset.read_image(photo)

    sam3 = Sam3Service(
        dataset=dataset,
    )
    anchor_model = AnchorService(
        dataset=dataset,
    )

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
    # Display the ear image and anchor points
    ear_image = apply_mask(image, ear.mask, crop=False)

    # Display the anchor points
    for anchor_point in ear.anchor_points:
        cv2.circle(ear_image, (int(anchor_point[0]), int(anchor_point[1])), 25, (0, 0, 255), -1)

    cv2.imshow("Ear Image with Anchor Points", apply_crop(ear_image, ear.xyxy))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ear_contour = ear.resampled_contour(CURVATURE_POINTS)

    ear_image = draw_polyline(ear_image, ear_contour, color=(0, 255, 0), thickness=2, marker_step=MARKER_STEP)
    cv2.imshow("Ear Image with Contour", apply_crop(ear_image, ear.xyxy))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    radii = SCALES * contour_max_dimension(ear_contour)
    curvature = oriented_curvature(ear_contour, radii, weights=WEIGHTS)
    print(f"Curvature min/max: {curvature.min():.4f}, {curvature.max():.4f}")

    ear_image = draw_radius_scale(apply_crop(ear_image, ear.xyxy), radii, SCALES, MATPLOTLIB_COLORS)
    cv2.imshow("Contour Image", ear_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plot_integral_curvature(
        curvature,
        marker_step=MARKER_STEP,
    )
    print(curvature.shape)
