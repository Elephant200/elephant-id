"""Convex-hull margin-depth profiles (SUPERSEDED exploration).

Kept as the hull-only ablation; the production pipeline is
elephant_id.margins (3-channel embedding) with scripts/evaluate.py as the
harness.

Pipeline (no alpha shape, no pole, no angles):
  1. reference = convex hull of the margin points (parameter-free, unique;
     no bay-follow/bridge bistability across poses, unlike the alpha hull),
  2. x = normalized arclength along the ear-side hull path between the two
     anchors (P[0] -> 0, P[-1] -> 1; trivially monotonic),
  3. depth(x) = first margin crossing along the hull's inward normal
     (parallel-beam scan; on intact margin the hull touches the margin and
     depth reads ~0; convexity makes "inward" unambiguous).

Unlike the alpha-20 envelope, the hull does NOT follow gentle bays, so the
profile contains bays + scallops + tears. That is a feature for matching:
on the 14-photo / 7-individual test set this profile gives top-1
identification 10/11 (best correlating photo is the same individual), mean
within-individual corr 0.88 vs between 0.56, beating every alpha-envelope
variant (best 9/11, within 0.57). The alpha envelope erases the gentle
scallop structure that is individually distinctive (nile); keep it for
tear-vs-bay CLASSIFICATION, use the hull profile for matching.

Outputs tear_hull_<label>.png (ear image + profile side by side),
tear_hull.png (all profiles), and prints within-individual pair
correlations plus top-1 identification over the photo set.

Run:  uv run python -m scripts.tear_hull
"""
import itertools

import cv2
import matplotlib
import numpy as np
import shapely
from scipy.ndimage import uniform_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from elephant_id.margins import densify, ear_side_path
from elephant_id.margins.geometry import ray_crossings
from elephant_id.margins.matching import base_name
from scripts.common import COLORS, LINESTYLES, PHOTOS, make_extractor, out_dir

BINS = 720
TRIM = 20           # bins (5% of length per end) excluded from stats
NORMAL_PROBE = 3.0  # px along the normal for the inward test


def hull_profile(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalized-arclength hull depth profile.

    Returns (depth, origins, normals, path); depth is in px.
    """
    hull = shapely.MultiPoint(P).convex_hull
    ring = np.asarray(hull.exterior.coords)[:-1]
    path = densify(ear_side_path(ring, P[0], P[-1]))
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    s /= s[-1]

    grid = (np.arange(BINS) + 0.5) / BINS
    origins = np.c_[np.interp(grid, s, path[:, 0]), np.interp(grid, s, path[:, 1])]
    sm = uniform_filter1d(path, size=9, axis=0)
    tan = np.gradient(sm, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    normals = np.c_[np.interp(grid, s, tan[:, 1]), np.interp(grid, s, -tan[:, 0])]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    probe = origins + NORMAL_PROBE * normals
    inside = shapely.contains_xy(hull, probe[:, 0], probe[:, 1])
    normals[~inside] *= -1

    t, ok = ray_crossings(origins, normals, P)
    ok &= t > -1e-6
    depth = np.where(ok, t, np.inf).min(axis=1)
    depth[~ok.any(axis=1)] = 0.0
    return np.clip(depth, 0, None), origins, normals, path


def pair_corr(d1: np.ndarray, d2: np.ndarray) -> tuple[float, float]:
    """Best correlation over +-15%-of-length shifts; returns (corr, shift_frac)."""
    shifts = range(-60, 61)
    corr = [float(np.corrcoef(d1[TRIM:-TRIM], np.roll(d2, sh)[TRIM:-TRIM])[0, 1])
            for sh in shifts]
    k = int(np.argmax(corr))
    return corr[k], shifts[k] / BINS


# -------------------------------- main ----------------------------------- #
def main() -> None:
    extractor = make_extractor()
    out = out_dir("tear_hull")

    grid = (np.arange(BINS) + 0.5) / BINS
    fig_all, ax_all = plt.subplots(figsize=(15, 5.5))
    profiles: dict[str, np.ndarray] = {}
    style_count: dict[str, int] = {}

    for label, ident in PHOTOS.items():
        P = extractor.margin(ident)
        if P is None:
            print(f"{label}: no ears")
            continue
        scale = float(np.max(P.max(0) - P.min(0)))
        depth, origins, normals, path = hull_profile(P)
        d = depth / scale
        profiles[label] = d
        k = int(np.argmax(d[TRIM:-TRIM])) + TRIM
        print(f"{label:8s} deepest {100 * d[k]:5.1f}% at x={grid[k]:.3f}")

        base = base_name(label)
        ls = LINESTYLES[style_count.get(base, 0) % len(LINESTYLES)]
        style_count[base] = style_count.get(base, 0) + 1
        ax_all.plot(grid, d, ls=ls, color=COLORS[base], lw=1.2, label=label)

        crop = extractor.crop(ident)
        if crop is None:
            continue
        img, off = crop
        fig, (axi, axp) = plt.subplots(
            1, 2, figsize=(16, 7.5), width_ratios=[1, 1.25])
        axi.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axi.plot(*(P - off).T, "w", lw=1.2, alpha=0.9, label="margin")
        axi.plot(*(path - off).T, "tab:cyan", lw=1.6, label="convex hull")
        hit = origins[k] + depth[k] * normals[k]
        axi.plot([origins[k, 0] - off[0], hit[0] - off[0]],
                 [origins[k, 1] - off[1], hit[1] - off[1]],
                 "r-", lw=2.0, label=f"deepest scan (x={grid[k]:.2f})")
        axi.legend(fontsize=9, loc="lower right")
        axi.set_title(f"{label} ({ident})", fontsize=12)
        axi.axis("off")

        axp.plot(grid, d, color=COLORS[base], lw=1.4)
        axp.axvspan(0, TRIM / BINS, color="0.85")
        axp.axvspan(1 - TRIM / BINS, 1, color="0.85")
        axp.plot(grid[k], d[k], "rv", ms=8)
        axp.set_xlim(0, 1)
        axp.set_ylim(0, max(1.3 * d[k], 0.05))
        axp.set_xlabel("normalized hull arclength (0 = anchor P[0], 1 = P[-1])")
        axp.set_ylabel("depth / scale")
        axp.set_title("hull depth profile", fontsize=12)
        axp.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"{label}.png", dpi=110)
        plt.close(fig)
        print(f"  saved {out}/{label}.png")

    # within-individual consistency + top-1 identification
    by_base: dict[str, list[str]] = {}
    for label in profiles:
        by_base.setdefault(base_name(label), []).append(label)
    for labels in by_base.values():
        for a, b in itertools.combinations(labels, 2):
            c, sh = pair_corr(profiles[a], profiles[b])
            print(f"pair {a}/{b}: corr {c:.3f} at shift {sh:+.3f}")
    multi = [x for x in profiles if len(by_base[base_name(x)]) > 1]
    hits = 0
    for a in multi:
        ranked = sorted((x for x in profiles if x != a),
                        key=lambda x: -pair_corr(profiles[a], profiles[x])[0])
        ok = base_name(ranked[0]) == base_name(a)
        hits += ok
        if not ok:
            print(f"top-1 MISS {a}: best match {ranked[0]}")
    print(f"top-1 identification: {hits}/{len(multi)}")

    ax_all.set_xlabel("normalized hull arclength")
    ax_all.set_ylabel("depth / scale")
    ax_all.set_title("convex-hull margin depth profiles", fontsize=12)
    ax_all.legend(fontsize=8, ncol=3)
    ax_all.grid(alpha=0.3)
    fig_all.tight_layout()
    fig_all.savefig(out / "summary.png", dpi=110)
    print(f"saved {out}/summary.png")


if __name__ == "__main__":
    main()
