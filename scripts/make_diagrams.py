"""Create publication diagrams for the high-quality tear matching evaluation."""

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "diagrams"

COLORS = {
    "ink": "#374B2E",
    "accent": "#4F9F19",
    "deep": "#617F52",
    "soft": "#AEBFA2",
    "pale": "#BDCC8E",
}

TOP_K = np.asarray([1, 3, 5, 10, 15])
TOP_K_ACCURACY = np.asarray([0.660, 0.768, 0.816, 0.872, 0.909])
TOP_K_SPREAD = np.asarray([0.010, 0.008, 0.008, 0.009, 0.008])
BAR_COLORS = [
    COLORS["deep"],
    COLORS["accent"],
    COLORS["soft"],
    COLORS["pale"],
    COLORS["ink"],
]


def top_k_accuracy_figure() -> Figure:
    """Return a minimal top-k retrieval accuracy figure."""
    fig, ax = plt.subplots(figsize=(3.5, 2.55), constrained_layout=True)
    x_positions = np.arange(len(TOP_K))

    ax.bar(
        x_positions,
        100.0 * TOP_K_ACCURACY,
        color=BAR_COLORS,
        edgecolor=COLORS["ink"],
        linewidth=0.7,
        width=0.68,
        zorder=3,
    )
    ax.errorbar(
        x_positions,
        100.0 * TOP_K_ACCURACY,
        yerr=100.0 * TOP_K_SPREAD,
        fmt="none",
        ecolor=COLORS["ink"],
        elinewidth=1.15,
        capsize=4.0,
        capthick=1.15,
        zorder=4,
    )
    for x_position, accuracy in zip(x_positions, TOP_K_ACCURACY, strict=True):
        ax.text(
            x_position,
            100.0 * accuracy + 2.2,
            f"{100.0 * accuracy:.1f}%",
            ha="center",
            va="bottom",
            color=COLORS["ink"],
            fontsize=8,
        )

    ax.set_xlabel("Rank cutoff (k)")
    ax.set_ylabel("Correct identity in top-k (%)")
    ax.set_xticks(x_positions, [f"top-{k}" for k in TOP_K])
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.55, len(TOP_K) - 0.45)
    ax.grid(axis="y", color=COLORS["soft"], linewidth=0.7, alpha=0.55)
    ax.grid(axis="x", visible=False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLORS["ink"])
        ax.spines[spine].set_linewidth(0.8)

    ax.tick_params(colors=COLORS["ink"], labelsize=8)
    ax.xaxis.label.set_color(COLORS["ink"])
    ax.yaxis.label.set_color(COLORS["ink"])
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)

    return fig


def save_figure(fig: Figure, stem: str) -> None:
    """Save a figure as vector and high-resolution raster outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", bbox_inches="tight", dpi=600)


def main() -> None:
    """Build and save all paper diagrams."""
    figure = top_k_accuracy_figure()
    save_figure(figure, "top_k_retrieval_accuracy")
    plt.close(figure)


if __name__ == "__main__":
    main()
