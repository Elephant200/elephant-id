"""Render desktop visualizations as local Matplotlib PNG files."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from elephant_id.visualize import (
    plot_aligned_tear_profiles,
    plot_tear_profile,
    tear_profile_ymax,
)

FIGURE_DPI = 150
PANEL_RGB = "#f6f1e3"
PLOT_RGB = "#f3eedd"
INK_RGB = "#2e2b21"
TEXT_DIM_RGB = "#615c48"
LINE_RGB = "#d2c8ac"
LEFT_RGB = "#3d6379"
RIGHT_RGB = "#8a5a2c"
QUERY_RGB = "#3e5a41"
CATALOG_RGB = "#b07c26"

_RENDER_LOCK = threading.Lock()


def render_tear_profile_png(
    profile: np.ndarray,
    output_path: Path,
    *,
    side: str,
    y_max: float = 0.4,
) -> Path:
    """Render one ear profile to a fixed-size desktop PNG.

    Raises:
        ValueError: If the profile, side, or y-axis limit is invalid.
    """
    values = _validated_profile(profile, "profile")
    if side not in ("left", "right"):
        raise ValueError(f"side must be left or right: {side}")
    if not np.isfinite(y_max) or y_max <= 0:
        raise ValueError("y_max must be a positive finite number")
    resolved_y_max = max(float(y_max), 1.08 * float(values.max(initial=0.0)))

    with _RENDER_LOCK:
        figure = _figure((5.0, 2.2))
        axis = figure.subplots()
        plot_tear_profile(
            axis,
            values,
            color=LEFT_RGB if side == "left" else RIGHT_RGB,
            y_max=resolved_y_max,
            title=f"{side.title()} ear tear profile",
        )
        _style_axis(axis)
        figure.tight_layout(pad=0.7)
        _write_png(figure, output_path)
    return output_path


def render_aligned_profiles_png(
    query_profile: np.ndarray,
    catalog_profile: np.ndarray,
    output_path: Path,
    *,
    side: str,
    shift_degrees: float,
    stretch: float,
    score: float,
    y_max: float | None = None,
) -> Path:
    """Render one aligned sighting-to-catalog comparison as a desktop PNG.

    Raises:
        ValueError: If the profiles or plot metadata are invalid.
    """
    query = _validated_profile(query_profile, "query_profile")
    catalog = _validated_profile(catalog_profile, "catalog_profile")
    if query.shape != catalog.shape:
        raise ValueError("query_profile and catalog_profile must have the same shape")
    if side not in ("left", "right"):
        raise ValueError(f"side must be left or right: {side}")
    metadata = np.asarray((shift_degrees, stretch, score), dtype=np.float64)
    if not np.isfinite(metadata).all() or stretch <= 0:
        raise ValueError("alignment metadata must be finite and stretch must be positive")
    requested_y_max = (
        tear_profile_ymax(np.vstack((query, catalog))) if y_max is None else float(y_max)
    )
    if not np.isfinite(requested_y_max) or requested_y_max <= 0:
        raise ValueError("y_max must be a positive finite number")
    resolved_y_max = max(
        requested_y_max,
        1.08 * float(max(query.max(initial=0.0), catalog.max(initial=0.0))),
    )

    with _RENDER_LOCK:
        figure = _figure((6.4, 2.55))
        axis = figure.subplots()
        plot_aligned_tear_profiles(
            axis,
            catalog,
            query,
            candidate_label="known-elephant catalog",
            candidate_color=CATALOG_RGB,
            color=QUERY_RGB,
            y_max=resolved_y_max,
            shift_fraction=shift_degrees / 180.0,
            stretch=stretch,
            score=score,
            ylabel="tear depth / R",
        )
        axis.set_title(
            f"{side.title()} ear · score {score:.3f}\n{axis.get_title()}",
            fontsize=9,
            color=INK_RGB,
        )
        _style_axis(axis)
        figure.tight_layout(pad=0.8)
        _write_png(figure, output_path)
    return output_path


def shared_profile_ymax(profile_pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Return one robust y-axis limit for a group of profile comparisons."""
    if not profile_pairs:
        return 0.08
    rows = [
        _validated_profile(profile, "profile")
        for pair in profile_pairs
        for profile in pair
    ]
    lengths = {len(profile) for profile in rows}
    if len(lengths) != 1:
        raise ValueError("all profiles must have the same length")
    profile_rows = np.vstack(rows)
    return max(
        tear_profile_ymax(profile_rows),
        1.08 * float(profile_rows.max(initial=0.0)),
    )


def _validated_profile(profile: np.ndarray, name: str) -> np.ndarray:
    """Return one finite, nonnegative profile suitable for plotting."""
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"{name} must be a one-dimensional array with at least two values")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.maximum(values, 0.0)


def _figure(size: tuple[float, float]) -> Figure:
    """Return an Agg-backed figure with the desktop panel background."""
    figure = Figure(figsize=size, dpi=FIGURE_DPI, facecolor=PANEL_RGB)
    FigureCanvasAgg(figure)
    return figure


def _style_axis(axis: Axes) -> None:
    """Apply the desktop application's savannah visual tokens to an axis."""
    axis.set_facecolor(PLOT_RGB)
    axis.grid(color=LINE_RGB, alpha=0.75, linewidth=0.7)
    axis.tick_params(colors=TEXT_DIM_RGB, labelsize=8)
    axis.xaxis.label.set_color(TEXT_DIM_RGB)
    axis.yaxis.label.set_color(TEXT_DIM_RGB)
    for spine in axis.spines.values():
        spine.set_color(LINE_RGB)


def _write_png(figure: Figure, output_path: Path) -> None:
    """Atomically write a Matplotlib figure as a fixed-DPI PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        figure.savefig(
            temp_path,
            format="png",
            dpi=FIGURE_DPI,
            facecolor=figure.get_facecolor(),
        )
        temp_path.replace(output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
