"""Visual QA for the ear-margin embedding: photo next to its profile.

For each manifest photo: the ear image with the margin (white), the alpha
reference (cyan), and the deepest scan ray (red) overlaid -- beside the 1-D
profile, with detected tear events marked. This is how embedding accuracy is
checked against the actual image: every bump in the profile should point at
a visible tear, and the profile should read as the ear unrolled flat.

Also writes an all-photo overlay (same individual = same color) where
same-ear profiles should visibly align.

Outputs (outputs/ear_embedding/): profiles.png, <label>.png per photo.

Run:  uv run python -m scripts.ear_embedding
"""
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from elephant_id.coding.tears import PROFILE_GRID, tear_profile
from elephant_id.constants import TEAR_TRIM_HI, TEAR_TRIM_LO
from scripts.evaluate import (
    PHOTOS,
    base_name,
    make_extractor,
    out_dir,
    tear_events,
)

LINESTYLES = ["-", "--", "-.", ":"]


def main() -> None:
    extractor = make_extractor()
    out = out_dir("ear_embedding")

    results = {}
    for label, ident in PHOTOS.items():
        P = extractor.contour(ident)
        if P is None:
            print(f"{label}: no contour")
            continue
        results[label] = (P, tear_profile(P))
        print(f"{label}: embedded")

    # all-photo overlay: one color per individual, linestyle per photo
    bases = sorted({base_name(label) for label in results})
    palette = plt.get_cmap("tab10").colors
    colors = {b: palette[i % len(palette)] for i, b in enumerate(bases)}
    fig, ax = plt.subplots(figsize=(15, 5))
    style_count: dict[str, int] = {}
    for label, (_, res) in results.items():
        base = base_name(label)
        ls = LINESTYLES[style_count.get(base, 0) % len(LINESTYLES)]
        style_count[base] = style_count.get(base, 0) + 1
        ax.plot(PROFILE_GRID, res.profile, ls=ls, color=colors[base],
                lw=1.1, label=label)
    ax.set_xlabel("normalized reference arclength (0 = anchor P[0], 1 = P[-1])")
    ax.set_ylabel("tear depth / S")
    ax.set_title("ear-margin tear profiles", fontsize=12)
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "profiles.png", dpi=110)
    plt.close(fig)
    print(f"saved {out}/profiles.png")

    # per-photo: ear image with overlays | profile with events
    for label, ident in PHOTOS.items():
        if label not in results:
            continue
        crop = extractor.crop(ident)
        if crop is None:
            continue
        img, off = crop
        P, res = results[label]
        events = tear_events(res.profile)
        k = int(np.argmax(res.profile))

        fig, (axi, axp) = plt.subplots(
            1, 2, figsize=(16, 7), width_ratios=[1, 1.25])
        axi.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axi.plot(*(P - off).T, "w", lw=1.1, alpha=0.9, label="margin")
        axi.plot(*(res.reference - off).T, "tab:cyan", lw=1.5,
                 label="alpha reference")
        hit = res.origins[k] + res.profile[k] * res.scale * res.normals[k]
        axi.plot([res.origins[k, 0] - off[0], hit[0] - off[0]],
                 [res.origins[k, 1] - off[1], hit[1] - off[1]],
                 "r-", lw=2.0, label="deepest tear")
        axi.legend(fontsize=9, loc="lower right")
        axi.set_title(f"{label} ({ident})", fontsize=12)
        axi.axis("off")

        axp.plot(PROFILE_GRID, res.profile, color=colors[base_name(label)],
                 lw=1.4)
        for x, d in events:
            axp.plot(x, d, "rv", ms=7)
        axp.axvspan(0, TEAR_TRIM_LO, color="0.85")
        axp.axvspan(1 - TEAR_TRIM_HI, 1, color="0.85")
        axp.set_xlim(0, 1)
        axp.set_ylim(min(1.3 * res.profile.min(), -0.002),
                     max(1.3 * res.profile.max(), 0.02))
        axp.set_xlabel("normalized reference arclength")
        axp.set_ylabel("tear depth / S")
        axp.set_title(f"tear profile ({len(events)} events)", fontsize=12)
        axp.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"{label}.png", dpi=110)
        plt.close(fig)
    print(f"saved {out}/<label>.png ({len(results)} photos)")


if __name__ == "__main__":
    main()
