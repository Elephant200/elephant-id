"""Visualization utilities for model predictions and tear-profile diagnostics.

OpenCV image helpers take and return a `BgrImage`. `color` arguments and
the palette are authored in RGB and flipped to BGR only when written into the
image buffer.
"""

from typing import Any

import cv2
import numpy as np
from matplotlib.axes import Axes

from elephant_id.image import BgrImage
from elephant_id.image.masks import decode_rle_mask
from elephant_id.inference import Detection
from elephant_id.matching.alphaphant.extraction import DEFAULT_VERSION
from elephant_id.matching.alphaphant.profile import TearProfile

_PROFILE_TRIM_DEGREES = DEFAULT_VERSION.config.trim_degrees


def _blend_bgr(
    image: BgrImage,
    mask: np.ndarray,
    bgr: tuple[int, int, int],
    alpha: float,
) -> None:
    """Alpha-blend a solid BGR color into image, in place.

    Raises:
        ValueError: If alpha, mask shape, or BGR color is invalid.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image size {image.shape[:2]}")
    if len(bgr) != 3 or any(not 0 <= c <= 255 for c in bgr):
        raise ValueError(f"bgr must be three values in [0, 255]: {bgr}")

    color = np.array(bgr, dtype=np.float32)
    image[mask] = ((1.0 - alpha) * image[mask] + alpha * color).astype(np.uint8)


def apply_alpha_mask(
    image: BgrImage,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> BgrImage:
    """Return a copy with a semi-transparent overlay where mask is True.

    Args:
        image: A BGR image to overlay the mask on.
        mask: A boolean mask to overlay on the image.
        color: The RGB color to use for the mask.
        alpha: The opacity of the mask in interval [0, 1].

    Returns:
        A new BGR image with the mask overlaid.
    """
    output = image.copy()
    _blend_bgr(output, mask.astype(bool), color[::-1], alpha)
    return output


def draw_rle_mask_overlay(
    image: BgrImage,
    rle: dict[str, Any],
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> BgrImage:
    """
    Overlay a single COCO RLE mask on an image.

    Args:
        image: A BGR image to overlay the mask on.
        rle: A COCO RLE mask to overlay on the image.
        color: The RGB color to use for the mask.
        alpha: The opacity of the mask in interval [0, 1].

    Returns:
        A new BGR image with the mask overlaid.
    """
    mask = decode_rle_mask(rle)
    return apply_alpha_mask(image, mask, color=color, alpha=alpha)


# Palette authored in RGB (with readable colour names); flipped to BGR at use.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (255, 127, 14),  # orange
    (31, 119, 180),  # blue
    (44, 160, 44),  # green
    (214, 39, 40),  # red
    (148, 103, 189),  # purple
    (140, 86, 75),  # brown
)


def visualize_predictions(
    image: BgrImage,
    detections: list[Detection],
    mask_alpha: float = 0.35,
) -> BgrImage:
    """Draw detections (RLE masks + boxes + labels) on an image.

    Args:
        image: Source BGR image.
        detections: Detections to render (see `Detection`).
        mask_alpha: Blend factor for the mask overlay in [0, 1].

    Returns:
        A new BGR image with all detections rendered.
    """
    output = image.copy()
    image_height, image_width = output.shape[:2]

    for detection in detections:
        class_id = detection.class_id
        class_name = detection.class_name
        confidence = detection.confidence
        bgr = _PALETTE[class_id % len(_PALETTE)][::-1]  # RGB palette -> BGR

        if detection.rle_mask is not None:
            mask = decode_rle_mask(detection.rle_mask)
            if mask.shape != (image_height, image_width):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            _blend_bgr(output, mask, bgr, mask_alpha)

        if detection.keypoints:
            for keypoint in detection.keypoints:
                x, y = int(keypoint[0]), int(keypoint[1])
                cv2.circle(output, (x, y), 5, bgr, -1)

        clipped = detection.clip(image_width, image_height)
        x1, y1, x2, y2 = int(clipped.x1), int(clipped.y1), int(clipped.x2), int(clipped.y2)
        # Detection boxes are half-open (x2/y2 exclusive); cv2.rectangle treats
        # its second corner as inclusive, so step back one pixel.
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), bgr, 2)

        label = f"{class_name} {confidence:.2f}"
        text_anchor = (x1, max(20, y1 - 8))
        cv2.putText(
            output,
            label,
            text_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            text_anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            bgr,
            1,
            cv2.LINE_AA,
        )

    return output


def _tear_profile_angles(profile_length: int) -> np.ndarray:
    """Return the angular x-axis for a tear profile."""
    return np.linspace(0.0, 180.0, profile_length)


def plot_tear_profile(
    axis: Axes,
    tear_profile: TearProfile | np.ndarray,
    *,
    color: str = "tab:red",
    y_max: float = 0.4,
    title: str = "Tear profile",
    trim_degrees: float = _PROFILE_TRIM_DEGREES,
) -> None:
    """Plot one tear-depth profile with trimmed angle bands."""
    profile = (
        np.asarray(tear_profile.depths, dtype=np.float64)
        if isinstance(tear_profile, TearProfile)
        else np.asarray(tear_profile, dtype=np.float64)
    )
    angles = _tear_profile_angles(len(profile))
    axis.plot(angles, profile, color=color, linewidth=1.4)
    axis.axvspan(0, trim_degrees, color="0.85")
    axis.axvspan(180.0 - trim_degrees, 180.0, color="0.85")
    axis.axvline(
        trim_degrees,
        color="tab:orange",
        linestyle="--",
        lw=1.0,
    )
    axis.axvline(
        180.0 - trim_degrees,
        color="tab:orange",
        linestyle="--",
        lw=1.0,
    )
    axis.set(xlim=(0, 183), ylim=(-0.03, y_max))
    axis.set_xticks(np.arange(0, 181, 30))
    axis.set_xlabel("angle (degrees)", fontsize=8, labelpad=2)
    axis.set_ylabel("depth / R", fontsize=8, labelpad=2)
    axis.tick_params(axis="both", labelsize=8, pad=2)
    axis.grid(alpha=0.3)
    axis.text(
        0.5,
        0.96,
        title,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        transform=axis.transAxes,
    )


def align_tear_profile_for_plot(
    profile: np.ndarray,
    shift_fraction: float,
    stretch: float,
) -> np.ndarray:
    """Return a clipped tear profile after the matcher's plot-space alignment."""
    if stretch <= 0:
        raise ValueError("stretch must be positive")
    profile = np.maximum(np.asarray(profile, dtype=np.float64), 0.0)
    grid = np.linspace(0.0, 1.0, len(profile))
    source_arc = (grid - 0.5) / stretch + 0.5
    stretched = np.interp(source_arc, grid, profile, left=0.0, right=0.0)

    shift = round(shift_fraction * len(profile))
    shifted = np.zeros_like(stretched)
    if shift > 0:
        shifted[shift:] = stretched[:-shift]
    elif shift < 0:
        shifted[:shift] = stretched[-shift:]
    else:
        shifted[:] = stretched
    return shifted


def plot_aligned_tear_profiles(
    axis: Axes,
    candidate_profile: np.ndarray,
    aligned_query_profile: np.ndarray,
    *,
    candidate_label: str,
    color: str,
    candidate_color: str = "black",
    y_max: float,
    shift_fraction: float | None = None,
    shift_bins: int | None = None,
    stretch: float | None = None,
    penalty: float | None = None,
    overlap_score: float | None = None,
    score: float | None = None,
    ylabel: str = "tear depth / R",
) -> None:
    """Plot one candidate profile against an already aligned query profile."""
    candidate_profile = np.maximum(np.asarray(candidate_profile, dtype=np.float64), 0.0)
    aligned_query_profile = np.maximum(
        np.asarray(aligned_query_profile, dtype=np.float64),
        0.0,
    )
    angles = _tear_profile_angles(len(candidate_profile))
    axis.fill_between(
        angles,
        np.minimum(candidate_profile, aligned_query_profile),
        color=color,
        alpha=0.25,
        label="overlap",
    )
    axis.plot(
        angles,
        candidate_profile,
        color=candidate_color,
        linewidth=1.6,
        label=candidate_label,
    )
    axis.plot(
        angles,
        aligned_query_profile,
        color=color,
        linewidth=1.6,
        label="query aligned",
    )

    title = candidate_label
    if shift_fraction is not None and stretch is not None:
        title = f"{candidate_label}: shift {shift_fraction * 180:+.1f}°"
        if shift_bins is not None:
            title += f" (bin {shift_bins:+d})"
        title += f", stretch x{stretch:.2f}"
        if penalty is not None:
            title += f", penalty x{penalty:.2f}"
    if overlap_score is not None and score is not None:
        title += f"\nIoU {overlap_score:.3f} -> score {score:.3f}"

    axis.set(title=title, xlim=(0, 180), ylim=(0, y_max), xlabel="ear angle (deg)", ylabel=ylabel)
    axis.tick_params(labelsize=7)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, loc="upper right")


def tear_profile_ymax(profiles: np.ndarray, *, minimum: float = 0.08) -> float:
    """Return a robust shared y-axis limit for tear-profile plots."""
    profile_rows = np.asarray(profiles, dtype=np.float64)
    if profile_rows.ndim == 1:
        profile_rows = profile_rows[None, :]
    peaks = np.maximum(profile_rows, 0.0).max(axis=1)
    return max(minimum, 1.1 * float(np.quantile(peaks, 0.95)))
