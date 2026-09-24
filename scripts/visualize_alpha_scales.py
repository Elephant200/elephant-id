"""Render alpha shapes and tear profiles across many reference scales.

Run with `uv run python scripts/visualize_alpha_scales.py`.
Figures are saved under `outputs/alpha_scales/`.
Selection is limited to Gemini curation scores of 9 or 10.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from elephant_id.cache import CacheManager
from elephant_id.composition import build_preparer
from elephant_id.dataset import Dataset
from elephant_id.domain import SightingEarPair
from elephant_id.image import BgrImage, decode_image
from elephant_id.image.boxes import BoundingBox, clip_xyxy
from elephant_id.image.geometry import resample_polyline
from elephant_id.log import configure_logging
from elephant_id.matching.alphaphant.extraction import (
    AlphaTearConfig,
    TearProfile,
    _alpha_shape,
    _densify,
    _opened_contour,
    _polar_directions,
    _reference_crossings,
    _signed_depths,
)
from elephant_id.preparation.ear import EarSide, PreparedEar, ear_flap_path
from elephant_id.visualize import tear_profile_ymax

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "alpha_scales"
CURATION_RESULTS = REPO_ROOT / "outputs/curation/archive/results.json"
PROFILE_PRODUCER = "alpha-tear-v3-multiscale"
PRODUCTION_SCALES = AlphaTearConfig().alpha_fractions
REFERENCE_SPECS: tuple[tuple[float | None, str], ...] = (
    (0.05, "lower"),
    (0.11, "current"),
    (0.22, "current"),
    (0.35, "old single"),
    (0.50, "current"),
    (1.10, "current"),
    (2.50, "current"),
    (5.00, "current"),
    (12.00, "current"),
    (None, "convex"),
)
_ROLE_COLORS = {
    "current": "#2f6b3a",
    "old single": "#b35c1e",
    "lower": "#5c5c5c",
    "convex": "#1a1a1a",
}
_BLOCKED_FACTORS = {"occlusion", "vegetation"}
_REFERENCE_COLORS = (
    "#00e5ff",
    "#76ff03",
    "#ffea00",
    "#ff9100",
    "#ff1744",
    "#d500f9",
    "#651fff",
    "#2979ff",
    "#304ffe",
    "#111111",
)


@dataclass(frozen=True, slots=True)
class QualityEar:
    """One Gemini-assessed ear that can be prepared as part of a sighting pair."""

    name: str
    side: EarSide
    sighting_id: UUID
    photo_id: UUID
    other_photo_id: UUID
    gemini_score: int


@dataclass(frozen=True, slots=True)
class RankedEar:
    """A cached high-quality ear selected for an identifiable tear pattern."""

    ear: QualityEar
    reason: str
    peak_fine: float
    peak_mid: float
    peak_coarse: float
    fine_peaks: int


@dataclass(frozen=True, slots=True)
class ScaleView:
    """One reference shape with its measuring path and extracted profile."""

    scale: float | None
    role: str
    reference_path: np.ndarray
    opened: np.ndarray
    profile: TearProfile

    @property
    def label(self) -> str:
        """Return the panel title for this reference."""
        return "convex hull" if self.scale is None else f"scale {self.scale:g}"


def _load_quality_ears(path: Path) -> dict[tuple[UUID, EarSide], QualityEar]:
    """Index Gemini 9/10 ears that have a complementary photo on the other side."""
    records = json.loads(path.read_text(encoding="utf-8"))
    by_sighting: dict[UUID, dict[EarSide, list[tuple[int, UUID]]]] = {}
    assessed: list[tuple[UUID, EarSide, int, str, UUID]] = []
    for record in records:
        sighting_id = UUID(record["sighting_id"])
        name = str(record["identity"])
        photo_id = UUID(record["photo_id"])
        for ear in record.get("ears", ()):
            assessment = ear.get("assessment") or {}
            fields = assessment.get("ear") or {}
            score = fields.get("reid_quality")
            side = ear.get("side")
            if side not in ("left", "right") or type(score) is not int:
                continue
            by_sighting.setdefault(sighting_id, {}).setdefault(side, []).append(
                (score, photo_id)
            )
            factors = set(fields.get("limiting_factors") or [])
            if (
                score >= 9
                and not factors & _BLOCKED_FACTORS
                and fields.get("contour_visibility") == "complete"
            ):
                assessed.append((photo_id, side, score, name, sighting_id))

    ears: dict[tuple[UUID, EarSide], QualityEar] = {}
    for photo_id, side, score, name, sighting_id in assessed:
        other_side: EarSide = "right" if side == "left" else "left"
        others = by_sighting.get(sighting_id, {}).get(other_side, [])
        if not others:
            continue
        other_photo_id = max(others, key=lambda item: item[0])[1]
        ears[(photo_id, side)] = QualityEar(
            name=name,
            side=side,
            sighting_id=sighting_id,
            photo_id=photo_id,
            other_photo_id=other_photo_id,
            gemini_score=score,
        )
    return ears


def _artifact_profile(depths: np.ndarray) -> bool:
    """True when a cached profile is dominated by an implausible edge spike."""
    values = np.maximum(np.asarray(depths, dtype=np.float64), 0.0)
    if values.size < 20:
        return True
    trim = max(int(0.05 * len(values)), 5)
    interior = values[trim:-trim]
    peak = float(values.max())
    interior_peak = float(interior.max()) if interior.size else 0.0
    return peak > 0.65 or (peak > 0.25 and peak > 1.5 * interior_peak + 1e-9)


def _count_peaks(depths: np.ndarray, min_height: float = 0.025) -> int:
    """Count well-separated positive local maxima in one profile."""
    peaks, _ = find_peaks(np.maximum(depths, 0.0), height=min_height, distance=12)
    return len(peaks)


def _rank_cached_ears(
    cache: CacheManager,
    ears: dict[tuple[UUID, EarSide], QualityEar],
) -> list[RankedEar]:
    """Score cached multiscale profiles that belong to high-quality ears."""
    ranked: list[RankedEar] = []
    producer_dir = cache.cache_root / PROFILE_PRODUCER
    for path in producer_dir.glob("*.json"):
        stem = path.stem
        try:
            photo_id = UUID(stem.split("__", 1)[0])
            side = stem.split("__side_")[1].split("__", 1)[0]
        except (IndexError, ValueError):
            continue
        if side not in ("left", "right"):
            continue
        quality_ear = ears.get((photo_id, side))
        if quality_ear is None:
            continue
        record = cache.load(PROFILE_PRODUCER, stem)
        profiles = record.get("profiles")
        if not isinstance(profiles, list) or len(profiles) != len(PRODUCTION_SCALES):
            continue
        rows = [np.asarray(row, dtype=np.float64) for row in profiles]
        if any(row.ndim != 1 or len(row) == 0 for row in rows):
            continue
        if _artifact_profile(rows[0]) or _artifact_profile(rows[3]):
            continue
        ranked.append(
            RankedEar(
                ear=quality_ear,
                reason="",
                peak_fine=float(np.max(rows[0])),
                peak_mid=float(np.max(rows[3])),
                peak_coarse=float(np.max(rows[6])),
                fine_peaks=_count_peaks(rows[0]),
            )
        )
    return ranked


def _pick_interesting(ranked: list[RankedEar], count: int) -> list[RankedEar]:
    """Choose complementary tear patterns from different high-quality elephants."""
    if not ranked:
        raise ValueError("No cached profiles match Gemini scores of 9 or 10")

    def take(
        reason: str,
        candidates: list[RankedEar],
        chosen: list[RankedEar],
        used_names: set[str],
    ) -> None:
        """Keep the first unused elephant from an already ordered list."""
        if len(chosen) >= count:
            return
        for candidate in candidates:
            if candidate.ear.name in used_names:
                continue
            chosen.append(
                RankedEar(
                    ear=candidate.ear,
                    reason=reason,
                    peak_fine=candidate.peak_fine,
                    peak_mid=candidate.peak_mid,
                    peak_coarse=candidate.peak_coarse,
                    fine_peaks=candidate.fine_peaks,
                )
            )
            used_names.add(candidate.ear.name)
            return

    chosen: list[RankedEar] = []
    used_names: set[str] = set()
    take(
        "large tear that only coarser references bridge",
        sorted(
            [row for row in ranked if row.peak_mid >= 0.20 and row.peak_fine < 0.10],
            key=lambda row: (row.ear.gemini_score, row.peak_mid),
            reverse=True,
        ),
        chosen,
        used_names,
    )
    take(
        "one large, clearly visible notch",
        sorted(
            [row for row in ranked if row.fine_peaks <= 2 and row.peak_mid >= 0.20],
            key=lambda row: (row.ear.gemini_score, row.peak_mid),
            reverse=True,
        ),
        chosen,
        used_names,
    )
    take(
        "many small nicks along the margin",
        sorted(
            [row for row in ranked if row.fine_peaks >= 4 and row.peak_mid < 0.16],
            key=lambda row: (row.ear.gemini_score, row.fine_peaks, row.peak_fine),
            reverse=True,
        ),
        chosen,
        used_names,
    )
    take(
        "irregular torn margin, small and large together",
        sorted(
            [row for row in ranked if row.fine_peaks >= 4],
            key=lambda row: (row.ear.gemini_score, row.peak_mid, row.fine_peaks),
            reverse=True,
        ),
        chosen,
        used_names,
    )
    leftovers = sorted(
        ranked,
        key=lambda row: (row.ear.gemini_score, row.peak_mid, row.fine_peaks),
        reverse=True,
    )
    while len(chosen) < count:
        before = len(chosen)
        take("additional distinctive high-quality profile", leftovers, chosen, used_names)
        if len(chosen) == before:
            break
    return chosen[:count]


def _reference_hull(opened: np.ndarray, alpha_radius: float | None) -> shapely.Polygon:
    """Return one alpha shape, or the convex hull when no radius is given."""
    if alpha_radius is None:
        hull = shapely.MultiPoint(opened).convex_hull
        if hull.geom_type != "Polygon":
            raise ValueError("Convex reference must be a polygon")
        return hull
    return _alpha_shape(opened, alpha_radius)


def _extract_scale_views(
    ear: PreparedEar,
    specs: tuple[tuple[float | None, str], ...],
) -> tuple[ScaleView, ...]:
    """Share contour preparation while retaining each reference path."""
    config = AlphaTearConfig(
        alpha_fractions=tuple(scale for scale, _ in specs if scale is not None)
    )
    contour = resample_polyline(ear.contour, config.contour_points)
    radius = float(np.sqrt(2.0 * ear.cleaned_area / np.pi))
    opened = _opened_contour(contour, config.morph_opening_fraction * radius)
    angles = np.linspace(0.0, 180.0, config.profile_bins)
    coded = (angles > config.trim_degrees) & (angles < 180.0 - config.trim_degrees)
    upper = np.asarray(ear.original_landmarks[0], dtype=float)
    lower = np.asarray(ear.original_landmarks[1], dtype=float)
    midpoint, directions = _polar_directions(
        upper, lower, ear.inferred_side, angles[coded]
    )
    views: list[ScaleView] = []
    for scale, role in specs:
        hull = _reference_hull(opened, None if scale is None else scale * radius)
        boundary = np.asarray(hull.exterior.coords)[:-1]
        reference_path = _densify(ear_flap_path(boundary, opened[0], opened[-1]))
        points, normals = _reference_crossings(midpoint, directions, hull)
        hit = np.isfinite(points).all(axis=1)
        coded_depths = np.zeros(len(points))
        if hit.any():
            coded_depths[hit] = (
                _signed_depths(points[hit], normals[hit], contour) / radius
            )
        depths = np.zeros(config.profile_bins)
        smoothed = gaussian_filter1d(coded_depths, sigma=config.smoothing_sigma)
        smoothed[~hit] = 0.0
        depths[coded] = smoothed
        views.append(
            ScaleView(
                scale=scale,
                role=role,
                reference_path=reference_path,
                opened=opened,
                profile=TearProfile(depths),
            )
        )
    return tuple(views)


def _window_from_points(
    image: BgrImage,
    xs: list[float],
    ys: list[float],
    pad_fraction: float,
) -> BoundingBox:
    """Clip a padded bounding box of the supplied points to the image."""
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    pad = pad_fraction * max(width, height, 1.0)
    return BoundingBox(
        *clip_xyxy(
            min(xs) - pad,
            min(ys) - pad,
            max(xs) + pad,
            max(ys) + pad,
            image_width=image.shape[1],
            image_height=image.shape[0],
        )
    )


def _shared_window(
    image: BgrImage,
    ear: PreparedEar,
    views: tuple[ScaleView, ...],
) -> BoundingBox:
    """Return one crop window framed on the torn margin, not the full ear disk."""
    xs = list(ear.contour[:, 0])
    ys = list(ear.contour[:, 1])
    for view in views:
        xs.extend(view.reference_path[:, 0])
        ys.extend(view.reference_path[:, 1])
        xs.extend(view.opened[:, 0])
        ys.extend(view.opened[:, 1])
    return _window_from_points(image, xs, ys, 0.10)


def _focus_window(
    image: BgrImage,
    ear: PreparedEar,
    views: tuple[ScaleView, ...],
) -> BoundingBox:
    """Zoom to the largest pocket where references peel away from the ear."""
    opened = views[0].opened
    tree = cKDTree(opened)
    radius = float(np.sqrt(2.0 * ear.cleaned_area / np.pi))
    coarsest = views[-1]
    distances, _ = tree.query(coarsest.reference_path)
    focus = coarsest.reference_path[int(np.argmax(distances))]
    window_r = max(0.20 * radius, 2.8 * float(distances.max()))
    xs = [float(focus[0])]
    ys = [float(focus[1])]
    for view in views:
        points = np.vstack([view.reference_path, view.opened])
        keep = np.linalg.norm(points - focus, axis=1) <= window_r
        if keep.any():
            xs.extend(points[keep, 0])
            ys.extend(points[keep, 1])
    return _window_from_points(image, xs, ys, 0.18)


def _draw_photo(
    axis: Axes,
    image: BgrImage,
    window: BoundingBox,
    ear: PreparedEar,
) -> np.ndarray:
    """Show one shared ear crop and return the window offset."""
    crop = image[window.y1 : window.y2, window.x1 : window.x2]
    offset = np.asarray((window.x1, window.y1), dtype=float)
    axis.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    contour = ear.contour - offset
    axis.plot(contour[:, 0], contour[:, 1], color="white", linewidth=1.0, zorder=3)
    for point in ear.original_landmarks:
        axis.scatter(
            point[0] - offset[0],
            point[1] - offset[1],
            s=22,
            c="white",
            edgecolors="black",
            linewidths=0.6,
            zorder=7,
        )
    axis.set_axis_off()
    axis.set_xlim(0, crop.shape[1])
    axis.set_ylim(crop.shape[0], 0)
    return offset


def _draw_scale_geometry(
    axis: Axes,
    image: BgrImage,
    window: BoundingBox,
    ear: PreparedEar,
    view: ScaleView,
) -> None:
    """Draw one reference path and the depth band it measures."""
    offset = _draw_photo(axis, image, window, ear)
    opened = view.opened - offset
    reference = view.reference_path - offset
    band = np.vstack([reference, opened[::-1]])
    axis.fill(band[:, 0], band[:, 1], color="#ff7f0e", alpha=0.45, zorder=2)
    axis.plot(opened[:, 0], opened[:, 1], color="#dddddd", linewidth=1.05, zorder=5)
    axis.plot(reference[:, 0], reference[:, 1], color="#e67e22", linewidth=2.25, zorder=6)


def _plot_compact_profile(
    axis: Axes,
    profile: TearProfile,
    *,
    y_max: float,
    labeled: bool,
) -> None:
    """Plot one depth profile with a shared y-limit."""
    depths = np.asarray(profile.depths, dtype=np.float64)
    angles = np.linspace(0.0, 180.0, len(depths))
    axis.plot(angles, depths, color="#c0392b", linewidth=1.2)
    axis.axhline(0.0, color="#bbbbbb", linewidth=0.6)
    axis.set(xlim=(0.0, 180.0), ylim=(-0.02, y_max))
    axis.set_xticks((0, 90, 180))
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.tick_params(labelsize=7 if labeled else 6, pad=1)
    if labeled:
        axis.set_xlabel("angle (deg)", fontsize=7, labelpad=1)
        axis.set_ylabel("depth / R", fontsize=7, labelpad=1)
    peak = float(np.max(depths))
    axis.text(
        0.98,
        0.90,
        f"{peak:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=7 if labeled else 6,
        color="#333333",
    )


def _label_scale(axis: Axes, view: ScaleView) -> None:
    """Write the reference name and its role onto one geometry panel."""
    axis.text(
        0.03,
        0.97,
        view.label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2.0, "edgecolor": "none"},
    )
    axis.text(
        0.97,
        0.97,
        view.role,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="white",
        bbox={
            "facecolor": _ROLE_COLORS[view.role],
            "alpha": 0.9,
            "pad": 2.0,
            "edgecolor": "none",
        },
    )


def _draw_overlaid_references(
    axis: Axes,
    image: BgrImage,
    window: BoundingBox,
    ear: PreparedEar,
    views: tuple[ScaleView, ...],
) -> None:
    """Draw every reference path on one shared crop."""
    offset = _draw_photo(axis, image, window, ear)
    for index, (view, color) in enumerate(zip(views, _REFERENCE_COLORS, strict=True)):
        path = view.reference_path - offset
        width = 3.0 if view.scale is None else 1.8 + 0.12 * index
        linestyle = (0, (4, 2)) if view.role == "old single" else "-"
        axis.plot(
            path[:, 0],
            path[:, 1],
            color="black",
            linewidth=width + 1.4,
            linestyle=linestyle,
            zorder=5,
        )
        axis.plot(
            path[:, 0],
            path[:, 1],
            color=color,
            linewidth=width,
            linestyle=linestyle,
            zorder=6,
        )


def _plot_overlaid_profiles(
    axis: Axes,
    views: tuple[ScaleView, ...],
    y_max: float,
) -> None:
    """Draw every depth profile on one shared angular axis."""
    for view, color in zip(views, _REFERENCE_COLORS, strict=True):
        depths = np.asarray(view.profile.depths, dtype=np.float64)
        angles = np.linspace(0.0, 180.0, len(depths))
        width = 2.2 if view.scale is None else 1.3
        linestyle = (0, (4, 2)) if view.role == "old single" else "-"
        axis.plot(angles, depths, color=color, linewidth=width, linestyle=linestyle)
    axis.axhline(0.0, color="#bbbbbb", linewidth=0.6)
    axis.set(xlim=(0.0, 180.0), ylim=(-0.02, y_max))
    axis.set_xticks((0, 45, 90, 135, 180))
    axis.set_xlabel("angle along the ear (deg)", fontsize=9)
    axis.set_ylabel("tear depth / ear radius", fontsize=9)
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.tick_params(labelsize=8)


def _legend_handles() -> list[Line2D]:
    """Return one legend handle per reference, in display order."""
    handles: list[Line2D] = []
    for view_spec, color in zip(REFERENCE_SPECS, _REFERENCE_COLORS, strict=True):
        scale, role = view_spec
        label = "convex hull" if scale is None else f"{scale:g}"
        if role == "old single":
            label += "  old"
        linestyle = (0, (4, 2)) if role == "old single" else "-"
        handles.append(
            Line2D([0], [0], color=color, linewidth=2.0, linestyle=linestyle, label=label)
        )
    return handles


def _render_ear_figure(
    image: BgrImage,
    ranked: RankedEar,
    ear: PreparedEar,
    views: tuple[ScaleView, ...],
) -> Figure:
    """Build one page of separate reference panels for a single ear."""
    window = _shared_window(image, ear, views)
    y_max = tear_profile_ymax(np.vstack([view.profile.depths for view in views]))
    columns = 5
    rows = int(np.ceil(len(views) / columns))
    figure = plt.figure(figsize=(20.0, 4.4 * rows + 0.85), facecolor="white")
    grid = figure.add_gridspec(
        rows,
        columns,
        left=0.03,
        right=0.99,
        top=0.90,
        bottom=0.06,
        wspace=0.12,
        hspace=0.28,
    )
    figure.suptitle(
        f"{ranked.ear.name}  ·  {ranked.ear.side} ear  ·  Gemini {ranked.ear.gemini_score}/10  ·  {ranked.reason}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.935,
        "Orange line: reference used for depth. Orange band: nicks that reference skipped. "
        "White: opened ear contour. A peak appears where the reference bridges a notch.",
        ha="center",
        va="top",
        fontsize=8,
        color="#444444",
    )
    for index, view in enumerate(views):
        row, column = divmod(index, columns)
        cell = grid[row, column].subgridspec(2, 1, height_ratios=(2.15, 1.0), hspace=0.08)
        geometry_ax = figure.add_subplot(cell[0])
        profile_ax = figure.add_subplot(cell[1])
        _draw_scale_geometry(geometry_ax, image, window, ear, view)
        _label_scale(geometry_ax, view)
        _plot_compact_profile(profile_ax, view.profile, y_max=y_max, labeled=True)
    return figure


def _render_comparison(
    pages: list[tuple[RankedEar, BgrImage, PreparedEar, tuple[ScaleView, ...]]],
) -> Figure:
    """Build the overlay figure that shows how references nest and what they measure."""
    n_rows = len(pages)
    figure = plt.figure(figsize=(14.4, 4.15 * n_rows + 1.7), facecolor="white")
    grid = figure.add_gridspec(
        n_rows,
        2,
        width_ratios=(1.15, 1.0),
        left=0.04,
        right=0.98,
        top=0.80,
        bottom=0.06,
        wspace=0.12,
        hspace=0.32,
    )
    figure.suptitle(
        "Smaller references follow nicks. Larger ones, and the convex hull, skip across them.",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    figure.text(
        0.5,
        0.935,
        "Each colored line is a reference hugging the same ear, zoomed to the nicks they disagree about. "
        "Depth is the inward gap from that line to the ear edge. "
        "A line that dives into a nick stays flat there; a line that bridges it makes a peak.",
        ha="center",
        va="top",
        fontsize=9,
        color="#333333",
    )
    figure.legend(
        handles=_legend_handles(),
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.905),
    )
    for row, (ranked, image, ear, views) in enumerate(pages):
        window = _focus_window(image, ear, views)
        y_max = tear_profile_ymax(np.vstack([view.profile.depths for view in views]))
        geometry_ax = figure.add_subplot(grid[row, 0])
        profile_ax = figure.add_subplot(grid[row, 1])
        _draw_overlaid_references(geometry_ax, image, window, ear, views)
        geometry_ax.set_title(
            f"{ranked.ear.name}  ·  {ranked.ear.side}  ·  Gemini {ranked.ear.gemini_score}/10\n{ranked.reason}",
            fontsize=10,
            loc="left",
            pad=6,
        )
        _plot_overlaid_profiles(profile_ax, views, y_max)
    return figure


def _slug(value: str) -> str:
    """Return a filesystem-safe token."""
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def _pair_for(ear: QualityEar, dataset: Dataset) -> SightingEarPair:
    """Build the sighting pair needed to prepare the selected ear."""
    if ear.side == "left":
        left_id, right_id = ear.photo_id, ear.other_photo_id
    else:
        left_id, right_id = ear.other_photo_id, ear.photo_id
    return SightingEarPair(
        sighting_id=ear.sighting_id,
        left_photo=dataset.photo(left_id),
        right_photo=dataset.photo(right_id),
    )


def main() -> None:
    """Select high-quality ears and write alpha-scale comparison figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=4)
    args = parser.parse_args()
    configure_logging()

    dataset = Dataset(
        REPO_ROOT / "dataset/elephants-alive/coded",
        REPO_ROOT / "dataset/elephants-alive/images.csv",
    )
    cache = CacheManager()
    ears = _load_quality_ears(CURATION_RESULTS)
    selected = _pick_interesting(_rank_cached_ears(cache, ears), args.count)
    preparer = build_preparer(dataset.photo_store, cache)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.png"):
        stale.unlink()

    pages: list[tuple[RankedEar, BgrImage, PreparedEar, tuple[ScaleView, ...]]] = []
    summary_rows: list[dict[str, object]] = []
    for ranked in selected:
        left_ear, right_ear = preparer.prepare(_pair_for(ranked.ear, dataset))
        prepared = left_ear if ranked.ear.side == "left" else right_ear
        image = decode_image(dataset.photo_store.read(prepared.source_photo))
        views = _extract_scale_views(prepared, REFERENCE_SPECS)
        pages.append((ranked, image, prepared, views))

        figure = _render_ear_figure(image, ranked, prepared, views)
        stem = f"{_slug(ranked.ear.name)}-{ranked.ear.side}"
        path = output_dir / f"{stem}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        print(f"Saved {path}")
        summary_rows.append(
            {
                "name": ranked.ear.name,
                "side": ranked.ear.side,
                "gemini_score": ranked.ear.gemini_score,
                "reason": ranked.reason,
                "peak_fine": f"{ranked.peak_fine:.4f}",
                "peak_mid": f"{ranked.peak_mid:.4f}",
                "peak_coarse": f"{ranked.peak_coarse:.4f}",
                "fine_peaks": ranked.fine_peaks,
                "path": str(path),
            }
        )

    comparison = _render_comparison(pages)
    comparison_path = output_dir / "comparison.png"
    comparison.savefig(comparison_path, dpi=150)
    plt.close(comparison)
    print(f"Saved {comparison_path}")

    summary_path = output_dir / "selection.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
