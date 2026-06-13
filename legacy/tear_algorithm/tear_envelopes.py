"""Evaluate tear-detection envelopes on three real ears.

Compares, per ear, the depth signal produced by four "intact-margin" references:
  * convex hull          -- bridges by connecting shoulders (over-bridges gentle bays)
  * alpha concave hull   -- Delaunay alpha shape; rolling disk of radius r,
                            bridges gaps narrower than the disk
  * arPLS                -- smooth baseline; good for small tears, bleeds on big ones

depth[i] = distance from margin point i to the envelope contour (0 on intact edge).

The alpha-shape construction now lives in elephant_id.margins.geometry;
this script keeps the envelope COMPARISON (hull vs alpha radii vs arPLS).

Run:  uv run python -m scripts.tear_envelopes
"""
import cv2
import matplotlib
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from elephant_id.margins import alpha_shape
from scripts.common import PHOTOS as ALL_PHOTOS
from scripts.common import make_extractor, out_dir
from scripts.tear_baseline import _signed_dev, baseline_arpls

# original five-ear comparison set
PHOTOS = {k: ALL_PHOTOS[k] for k in ("ripley", "adam", "les", "larson", "delani")}


# ------------------------------ envelopes -------------------------------- #
def envelope_depth(P: np.ndarray, env_xy: np.ndarray) -> np.ndarray:
    """Distance from each margin point to a densely sampled envelope contour."""
    seg = np.linalg.norm(np.diff(env_xy, axis=0, append=env_xy[:1]), axis=1)
    dense = [env_xy]
    for i in range(len(env_xy)):
        n = int(seg[i] // 2) + 1
        a, b = env_xy[i], env_xy[(i + 1) % len(env_xy)]
        dense.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    tree = cKDTree(np.vstack(dense))
    return tree.query(P)[0]


def mad(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))) + 1e-9)


def spans(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def convex_env(P: np.ndarray) -> np.ndarray:
    h = ConvexHull(P)
    return P[h.vertices]


def alpha_env(P: np.ndarray, radius: float) -> np.ndarray:
    """Exterior ring of the alpha-shape envelope (see :func:`alpha_shape`)."""
    shape = alpha_shape(P, radius)
    if shape.is_empty:
        return convex_env(P)
    return np.asarray(shape.exterior.coords)[:-1]


# -------------------------------- main ----------------------------------- #
def main() -> None:
    extractor = make_extractor()
    out = out_dir("tear_envelopes")

    for label, ident in PHOTOS.items():
        P = extractor.margin(ident)
        if P is None:
            print(f"{label}: no ears")
            continue
        scale = float(np.max(P.max(0) - P.min(0)))

        # Alpha concave hull (rolling-ball envelope, point-space) swept across
        # radii, plus convex hull and arPLS for reference.
        radii_frac = [0.10, 0.20]
        envs = {"convex hull": convex_env(P)}
        for fr in radii_frac:
            envs[f"alpha {int(fr * 100)}%"] = alpha_env(P, fr * scale)
        depths = {k: envelope_depth(P, v) for k, v in envs.items()}
        arpls = -_signed_dev(P, baseline_arpls(P))  # inward positive

        print(f"\n=== {label} ({ident})  ear scale {scale:.0f}px ===")
        for k, d in depths.items():
            print(f"  {k:<14} max inward depth {d.max():6.1f}px ({100*d.max()/scale:4.1f}%)")
        print(f"  {'arPLS':<14} max inward depth {arpls.max():6.1f}px ({100*arpls.max()/scale:4.1f}%)")

        crop = extractor.crop(ident)
        if crop is None:
            continue
        img, off = crop
        # convex hull, 4 alpha radii, arPLS
        colors = ["tab:red", "tab:cyan", "tab:green", "tab:blue", "tab:purple"]
        fig = plt.figure(figsize=(20, 9))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

        axi = fig.add_subplot(gs[0, 0])
        axi.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axi.plot(*(P - off).T, color="w", lw=1.2, alpha=0.85, label="ear margin")
        for (k, v), c in zip(envs.items(), colors, strict=False):
            ring = np.vstack([v, v[:1]]) - off
            axi.plot(*ring.T, c, lw=1.8, label=k)
        axi.legend(fontsize=9, loc="lower right")
        axi.set_title(f"{label}: convex hull vs alpha radii", fontsize=13)
        axi.axis("off")

        axd = fig.add_subplot(gs[0, 1])
        for (k, d), c in zip(depths.items(), colors, strict=False):
            axd.plot(d, c, lw=1.5, label=k)
        axd.plot(arpls, color="0.35", lw=1.1, ls="--", label="arPLS")
        axd.set_ylim(0, max(0.30 * scale, max(d.max() for d in depths.values()) * 1.08))
        axd.set_title(f"{label}: inward depth by method", fontsize=13)
        axd.set_xlabel("contour index")
        axd.set_ylabel("inward depth (px)")
        axd.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(out / f"{label}.png", dpi=110)
        print(f"  saved {out}/{label}.png")


if __name__ == "__main__":
    main()
