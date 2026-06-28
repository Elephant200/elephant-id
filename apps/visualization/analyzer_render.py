"""Render full-photo analyzer diagnostics as a Matplotlib dashboard."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from elephant_id.image import BgrImage
from elephant_id.image.boxes import clip_xyxy
from elephant_id.image.masks import decode_rle_mask
from elephant_id.visualize import plot_tear_profile, plot_tear_profile_geometry


def draw_box(
    image: BgrImage,
    xyxy: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a labeled half-open box into a BGR image in place."""
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


def _feature_area(feature: Any) -> float:
    """Return an area from either a detection method or geometry property."""
    area = feature.area
    if callable(area):
        return float(area())
    return float(area)


def plot_ear_diagnostic(
    crop_ax: Axes,
    profile_ax: Axes,
    image: BgrImage,
    ear_data: dict,
) -> None:
    """Plot one ear crop beside its tear-depth profile."""
    ear = ear_data["ear"]
    profile = ear_data["tear_profile"]
    plot_tear_profile_geometry(crop_ax, image, ear, profile)
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
    plot_tear_profile(profile_ax, profile)


def build_analyzer_figure(
    analysis: dict,
    photo_identifier: str,
    image: BgrImage,
) -> Figure:
    """Build the dashboard figure for one completed photo analysis."""
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
        f"Photo analysis: {photo_identifier}",
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
    raw_ears = shared_data.get("raw_ears", shared_data.get("ears", []))
    ear_area_ratio_text = ""
    if len(raw_ears) == 2:
        larger_area, smaller_area = sorted(
            (_feature_area(ear) for ear in raw_ears),
            reverse=True,
        )
        ear_area_ratio_text = f"Ear area ratio: {larger_area / smaller_area:.2f}   |   "
    status_ax.add_patch(
        Rectangle(
            (0.18, 0.25),
            0.64,
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
        f"{ear_area_ratio_text}View: {analysis['view'].title()}   |   Tusks: {tusks}",
        ha="center",
        va="center",
        fontsize=10,
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
        profile_layout = row[1].subgridspec(
            3,
            1,
            height_ratios=(0.08, 0.72, 0.20),
            hspace=0,
        )
        profile_ax = figure.add_subplot(profile_layout[1])
        ear_data = ears_by_side.get(side)
        if ear_data is not None:
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
    return figure


def dashboard_png(analysis: dict, photo_identifier: str, image: BgrImage) -> bytes:
    """Render one analyzer dashboard into PNG bytes."""
    figure = build_analyzer_figure(analysis, photo_identifier, image)
    output = BytesIO()
    try:
        figure.savefig(output, format="png", dpi=150)
    finally:
        plt.close(figure)
    return output.getvalue()
