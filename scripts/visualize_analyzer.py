"""Show one photo analysis as a fixed Matplotlib dashboard.

Run with ``uv run python scripts/visualize_analyzer.py PHOTO_IDENTIFIER``. The figure is
displayed and saved under ``outputs/analyzer/``.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from elephant_id.coding.photo_analyzer import PhotoAnalyzer
from elephant_id.constants import TEAR_PROFILE_BINS, TEAR_TRIM_DEGREES
from elephant_id.dataset import Dataset
from elephant_id.image.boxes import clip_xyxy
from elephant_id.image.masks import decode_rle_mask
from elephant_id.image.transforms import apply_crop
from elephant_id.log import configure_logging


def draw_box(
    image: np.ndarray,
    xyxy: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a labeled half-open xyxy box on a BGR image in place."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clip_xyxy(*xyxy, width, height)
    bgr = color[::-1]
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), bgr, 2, cv2.LINE_AA)
    cv2.putText(
        image,
        label,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        bgr,
        2,
        cv2.LINE_AA,
    )


def plot_ear_diagnostic(
    crop_ax: Axes,
    profile_ax: Axes,
    image: np.ndarray,
    ear_data: dict,
) -> None:
    """Plot one ear crop alongside its tear-depth embedding."""
    ear = ear_data["ear"]
    profile = ear_data["tear_profile"]
    x1, y1, _, _ = clip_xyxy(*ear.xyxy, image.shape[1], image.shape[0])
    offset = np.array((x1, y1))
    angles_degrees = np.linspace(0.0, 180.0, TEAR_PROFILE_BINS)
    coded_angle_mask = (
        (angles_degrees > TEAR_TRIM_DEGREES)
        & (angles_degrees < 180.0 - TEAR_TRIM_DEGREES)
    )
    deepest = int(np.argmax(np.where(coded_angle_mask, profile.profile, -np.inf)))
    ray_end = (
        profile.origins[deepest]
        + profile.profile[deepest] * profile.scale * profile.normals[deepest]
    )

    crop_ax.imshow(cv2.cvtColor(apply_crop(image, ear.xyxy), cv2.COLOR_BGR2RGB))
    crop_ax.plot(
        *(ear.resampled_contour(1024) - offset).T,
        color="white",
        linewidth=1,
        label="cut contour",
    )
    crop_ax.plot(
        *(profile.reference - offset).T,
        color="tab:cyan",
        linewidth=1.4,
        label="reference",
    )
    crop_ax.plot(
        [profile.origins[deepest, 0] - offset[0], ray_end[0] - offset[0]],
        [profile.origins[deepest, 1] - offset[1], ray_end[1] - offset[1]],
        color="tab:red",
        linewidth=2,
        label="deepest profile ray",
    )
    crop_ax.text(
        0.5,
        0.98,
        f"{ear.side.title()} ear",
        color="white",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="semibold",
        transform=crop_ax.transAxes,
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 2, "edgecolor": "none"},
    )
    # crop_ax.legend(loc="lower right", fontsize=7, framealpha=0.8)
    crop_ax.set_xticks([])
    crop_ax.set_yticks([])
    for spine in crop_ax.spines.values():
        spine.set_visible(False)

    profile_ax.plot(
        angles_degrees,
        profile.profile,
        color="tab:red",
        linewidth=1.4,
    )
    profile_ax.axvspan(0, TEAR_TRIM_DEGREES, color="0.85")
    profile_ax.axvspan(
        180 - TEAR_TRIM_DEGREES,
        180,
        color="0.85",
    )
    profile_ax.set(xlim=(0, 180), ylim=(-0.03, 0.4))
    profile_ax.set_xticks((0, 45, 90, 135, 180))
    profile_ax.set_xlabel("angle (degrees)", fontsize=8, labelpad=2)
    profile_ax.set_ylabel("depth / R", fontsize=8, labelpad=2)
    profile_ax.tick_params(axis="both", labelsize=8, pad=2)
    profile_ax.grid(alpha=0.3)
    profile_ax.text(
        0.5,
        0.96,
        "Tear embedding",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="semibold",
        transform=profile_ax.transAxes,
    )


def main() -> None:
    """Analyze one photo and show its feature evidence on one screen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo")
    args = parser.parse_args()

    load_dotenv()
    configure_logging()
    repo_root = Path(__file__).resolve().parent.parent
    dataset = Dataset(
        dataset_root=repo_root / "dataset/elephants-alive/coded",
        metadata_path=repo_root / "dataset/elephants-alive/images.csv",
    )
    photo = dataset.get_photo(args.photo)
    analysis = PhotoAnalyzer(dataset=dataset).analyze(photo)
    if analysis is None:
        raise RuntimeError(f"Photo analyzer returned no usable evidence for {photo.identifier}")

    image = dataset.read_image(photo)
    shared_data = analysis["shared_data"]
    body = shared_data["body"]
    annotated = image.copy()
    body_mask = decode_rle_mask(body.rle_mask)
    annotated[body_mask] = (
        0.78 * annotated[body_mask] + 0.22 * np.array((255, 130, 70))
    ).astype(np.uint8)
    draw_box(annotated, body.xyxy, "body", (70, 130, 255))

    for trunk in shared_data["trunks"]:
        draw_box(annotated, trunk.xyxy, f"trunk {trunk.confidence:.0%}", (220, 220, 80))
    for tusk in analysis["tusks"]:
        draw_box(
            annotated,
            (tusk["x1"], tusk["y1"], tusk["x2"], tusk["y2"]),
            f"tusk: {tusk['side']} {tusk['confidence']:.0%}",
            (255, 105, 65),
        )
    for ear_data in analysis["ears"]:
        ear = ear_data["ear"]
        draw_box(annotated, ear.xyxy, f"{ear.side} ear", (80, 220, 130))
        contour = np.round(ear.contour).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(annotated, [contour], False, (130, 220, 80), 2, cv2.LINE_AA)
        for point, color in zip(
            ear.anchor_points,
            ((255, 255, 255), (255, 220, 80)),
            strict=True,
        ):
            cv2.circle(
                annotated,
                tuple(np.round(point).astype(int)),
                5,
                color[::-1],
                -1,
                cv2.LINE_AA,
            )

    figure = plt.figure(figsize=(18, 9), facecolor="white")
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    panes = figure.add_gridspec(1, 2, wspace=0)
    left = panes[0].subgridspec(2, 1, height_ratios=(8, 1), hspace=0)
    right = panes[1].subgridspec(2, 1, hspace=0)
    image_ax = figure.add_subplot(left[0])
    status_ax = figure.add_subplot(left[1])

    image_ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    image_ax.set_anchor("W")
    image_ax.margins(0)
    image_ax.text(
        0.01,
        0.98,
        f"Photo analysis: {photo.identifier}",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        transform=image_ax.transAxes,
        bbox={"facecolor": "white", "alpha": 0.85, "pad": 2, "edgecolor": "none"},
    )
    image_ax.set_xticks([])
    image_ax.set_yticks([])
    for spine in image_ax.spines.values():
        spine.set_visible(False)

    tusks = ", ".join(
        f"{tusk['side']} {tusk['confidence']:.0%}" for tusk in analysis["tusks"]
    ) or "none"
    status_ax.add_patch(
        Rectangle(
            (0.25, 0.25),
            0.5,
            0.5,
            transform=status_ax.transAxes,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
    )
    status_ax.text(
        0.5,
        0.5,
        f"View: {analysis['view'].title()}   |   Tusks: {tusks}",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="semibold",
        transform=status_ax.transAxes,
    )
    status_ax.set_xticks([])
    status_ax.set_yticks([])
    for spine in status_ax.spines.values():
        spine.set_visible(False)

    ears_by_side = {ear_data["ear"].side: ear_data for ear_data in analysis["ears"]}
    for index, side in enumerate(("left", "right")):
        row = right[index].subgridspec(1, 2, wspace=0.04)
        crop_ax = figure.add_subplot(row[0])
        profile_layout = row[1].subgridspec(3, 1, height_ratios=(0.08, 0.72, 0.20), hspace=0)
        profile_ax = figure.add_subplot(profile_layout[1])
        if ear_data := ears_by_side.get(side):
            plot_ear_diagnostic(crop_ax, profile_ax, image, ear_data)
        else:
            crop_ax.set_axis_off()
            profile_ax.set_axis_off()

    figure.add_artist(
        Line2D((0.5, 0.5), (0, 1), transform=figure.transFigure, color="black", linewidth=2)
    )
    figure.add_artist(
        Line2D((0.5, 1), (0.5, 0.5), transform=figure.transFigure, color="black", linewidth=2)
    )

    output_dir = repo_root / "outputs" / "analyzer"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{photo.identifier}.png"
    figure.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
