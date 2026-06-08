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
CONTOUR_ENDPOINT_MARGIN = 96
CURVATURE_THRESHOLD = 0.44
SCALES = np.array([0.02, 0.04, 0.06, 0.08, 0.10], dtype=np.float32)
WEIGHTS = np.array([0.6, 0.9, 1.0, 0.9, 0.6], dtype=np.float32)


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


def low_curvature_mask(
    curvature: np.ndarray,
    *,
    threshold: float,
    endpoint_margin: int,
) -> np.ndarray:
    """Return indices where curvature is at or below the threshold, excluding contour ends."""
    mask = curvature <= threshold
    mask[:endpoint_margin] = False
    mask[-endpoint_margin:] = False
    return mask


def shade_low_curvature_contour(
    image: BgrImage,
    contour: np.ndarray,
    low_curvature: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 0, 255),
    thickness: int = 10,
    alpha: float = 0.45,
) -> BgrImage:
    """Highlight contour segments whose curvature is at or below the threshold."""
    if len(contour) != len(low_curvature):
        raise ValueError(
            f"Contour and mask length mismatch: {len(contour)} vs {len(low_curvature)}"
        )

    output = image.copy()
    overlay = image.copy()
    points_i32 = contour.astype(np.int32)

    start: int | None = None
    for idx, is_low in enumerate(low_curvature):
        if is_low and start is None:
            start = idx
        elif not is_low and start is not None:
            segment = points_i32[start:idx]
            if len(segment) >= 2:
                cv2.polylines(
                    overlay,
                    [segment.reshape((-1, 1, 2))],
                    isClosed=False,
                    color=color[::-1],
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )
            start = None

    if start is not None:
        segment = points_i32[start:]
        if len(segment) >= 2:
            cv2.polylines(
                overlay,
                [segment.reshape((-1, 1, 2))],
                isClosed=False,
                color=color[::-1],
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    return cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0.0)


def plot_integral_curvature(
    ax: plt.Axes,
    curvature: np.ndarray,
    *,
    low_curvature: np.ndarray,
    threshold: float,
    ymin: float,
    ymax: float,
) -> None:
    """Draw curvature and low-curvature shading on an axes."""
    x = np.arange(curvature.shape[0])

    ax.fill_between(
        x,
        ymin,
        ymax,
        where=low_curvature,
        color=(1.0, 0.75, 0.85),
        alpha=0.55,
        zorder=0,
        label=f"Curvature <= {threshold:g}",
    )
    ax.plot(curvature, color="black", label="Curvature")
    ax.axhline(threshold, color="0.65", linewidth=1, alpha=0.75, zorder=1)


if __name__ == "__main__":
    load_dotenv()
    configure_logging()
    dataset = Dataset(
        dataset_root=Path("dataset/elephants-alive/coded"),
        metadata_path=Path("dataset/elephants-alive/images.csv"),
    )
    photo = dataset.get_photo("Classic_2017-01-24_10")
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
    curvature = oriented_curvature(ear_contour, radii, weights=WEIGHTS, side=ear.side)
    low_curvature = low_curvature_mask(
        curvature,
        threshold=CURVATURE_THRESHOLD,
        endpoint_margin=CONTOUR_ENDPOINT_MARGIN,
    )
    print(f"Curvature min/max: {curvature.min():.4f}, {curvature.max():.4f}")
    print(
        "Low-curvature points: "
        f"{int(low_curvature.sum())} / {len(curvature) - 2 * CONTOUR_ENDPOINT_MARGIN} "
        f"(threshold {CURVATURE_THRESHOLD}, margin {CONTOUR_ENDPOINT_MARGIN})"
    )

    ear_image = shade_low_curvature_contour(
        ear_image,
        ear_contour,
        low_curvature,
    )
    cv2.imshow("Ear Image with Low-Curvature Segments", apply_crop(ear_image, ear.xyxy))
    cv2.imwrite("ear_image_with_low_curvature_segments.png", apply_crop(ear_image, ear.xyxy))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    curvature_ymin = 0.1
    curvature_ymax = 0.7
    _, ax = plt.subplots(figsize=(19.2, 4.8))
    plot_integral_curvature(
        ax,
        curvature,
        low_curvature=low_curvature,
        threshold=CURVATURE_THRESHOLD,
        ymin=curvature_ymin,
        ymax=curvature_ymax,
    )
    ax.set_title("Elephant ear")
    ax.set_xlabel("Contour point")
    ax.set_ylabel("Curvature")
    ax.set_ylim(curvature_ymin, curvature_ymax)
    ax.set_xticks(np.arange(0, curvature.shape[0], MARKER_STEP))
    ax.tick_params(axis="x", rotation=90)
    ax.set_yticks(np.arange(curvature_ymin, curvature_ymax, 0.025))
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()
    print(curvature.shape)
