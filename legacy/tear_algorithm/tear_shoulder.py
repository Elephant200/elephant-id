"""Prototype: shoulder-bounded envelope tear detection (width-independent).

Core idea (validated against arPLS bleed):
  * A tear is NOT "a deviation from a smooth baseline" -- a wide tear is itself
    low-frequency, so any smoothness-penalized baseline sags into it.
  * A tear IS "a concavity bounded by two sharp shoulders." Measuring its depth
    against the chord between those shoulders (an outer envelope) is independent
    of how wide the tear is.

Hierarchical detector:
  1. Outer envelope via convex hull of the cut (anchor-to-anchor) margin.
     -> convexity defects = candidate tears; depth = floor-to-chord distance.
  2. Shoulder validation: a real tear has high-curvature corners at its mouth;
     a natural ear lobe has gentle shoulders. Discriminate on shoulder sharpness.
  3. Refinement: arPLS residual on the de-trended margin recovers shallow nicks
     the hull misses. Hull handles WIDE tears; arPLS handles NARROW ones.

Run:  uv run python scripts/tear_shoulder.py
"""
import json

import matplotlib
import numpy as np
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.tear_baseline import (
    _mad_band,
    _signed_dev,
    baseline_arpls,
    tear_segments,
)


def hull_defects(P: np.ndarray) -> np.ndarray:
    """Signed depth of every margin point INTO the convex hull (>=0 inside).

    Width-independent: the hull edge is a chord between two shoulders, so a point
    on a wide tear floor is just as deep as on a narrow one.
    """
    hull = ConvexHull(P)
    return np.min([-(P @ eq[:2] + eq[2]) for eq in hull.equations], axis=0)


def corner_sharpness(P: np.ndarray, span: int = 6) -> np.ndarray:
    """Turning angle at each point over +/- span samples. High = sharp corner."""
    n = len(P)
    a = P[np.arange(n) - span]
    b = P
    c = P[(np.arange(n) + span) % n]
    v1 = a - b
    v2 = c - b
    cos = np.einsum("ij,ij->i", v1, v2) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-12
    )
    return np.pi - np.arccos(np.clip(cos, -1, 1))  # 0 = straight, large = sharp


def detect(P: np.ndarray, k_mad: float = 3.0, min_px: float = 3.0,
           shoulder_drop: float = 0.4) -> dict:
    """Hierarchical tear detection. Returns hull tears + refinement nicks."""
    depth = hull_defects(P)
    band = _mad_band(depth[depth < np.percentile(depth, 75)])
    thr = max(k_mad * band, min_px)

    # candidate concavity spans where margin is meaningfully inside the hull
    is_concave = depth > thr
    sharp = corner_sharpness(P)

    tears = []
    for s, e in tear_segments(is_concave):
        idx = np.arange(s, e + 1)
        floor = int(idx[np.argmax(depth[idx])])
        d_floor = float(depth[floor])
        # shoulders: sharpest corners just outside the concavity mouth
        ls = sharp[max(0, s - 8):s + 4].max() if s > 0 else 0.0
        rs = sharp[e:min(len(P), e + 12)].max() if e < len(P) - 1 else 0.0
        shoulder = 0.5 * (ls + rs)
        tears.append(dict(start=s, end=e, floor=floor, depth_px=d_floor,
                          shoulder=float(shoulder),
                          is_tear=bool(shoulder > shoulder_drop)))
    tears.sort(key=lambda t: -t["depth_px"])
    return dict(depth=depth, thr=thr, sharp=sharp, tears=tears)


def main() -> None:
    P = np.array(json.load(open("data/contour.json")), float)
    res = detect(P)

    print(f"{len(P)} pts; hull-defect threshold {res['thr']:.1f}px\n")
    print(f"{'idx span':>14} {'depth':>8} {'shoulder':>9}  verdict")
    for t in res["tears"][:8]:
        v = "TEAR" if t["is_tear"] else "natural curve / noise"
        print(f"{t['start']:5d}-{t['end']:<5d}  {t['depth_px']:7.1f}px "
              f"{t['shoulder']:7.2f}rad  {v}")

    # arPLS depth on the same big tear, for the bleed comparison
    d_arpls = _signed_dev(P, baseline_arpls(P))
    big = res["tears"][0]
    seg = slice(big["start"], big["end"] + 1)
    print(f"\ndeepest tear (idx {big['start']}-{big['end']}):")
    print(f"  shoulder-envelope depth : {big['depth_px']:.1f}px")
    print(f"  arPLS smooth-baseline   : {float((-d_arpls[seg]).max()):.1f}px  (bled)")

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].plot(*P.T, color="0.6", lw=1.2, label="cut margin")
    hull = ConvexHull(P)
    for sx in hull.simplices:
        ax[0].plot(*P[sx].T, color="tab:green", lw=0.8, alpha=0.5)
    ax[0].plot([], [], color="tab:green", label="outer envelope (hull)")
    for t in res["tears"]:
        if t["is_tear"]:
            ax[0].plot(*P[t["start"]:t["end"] + 1].T, "m-", lw=4, alpha=0.8)
            f = t["floor"]
            ax[0].plot(*P[f], "r.", ms=12)
    ax[0].plot([], [], "m-", lw=4, label="validated tear")
    ax[0].set_aspect("equal"); ax[0].invert_yaxis()
    ax[0].legend(loc="lower left"); ax[0].set_title("shoulder-bounded tears")

    ax[1].plot(res["depth"], color="tab:blue", label="hull-defect depth")
    ax[1].axhline(res["thr"], color="r", ls="--", lw=0.8, label="threshold")
    ax2 = ax[1].twinx()
    ax2.plot(res["sharp"], color="tab:orange", lw=0.7, alpha=0.6,
             label="corner sharpness")
    ax[1].set_xlabel("contour index"); ax[1].set_ylabel("depth into hull (px)")
    ax2.set_ylabel("corner sharpness (rad)")
    ax[1].legend(loc="upper left"); ax2.legend(loc="upper right")
    ax[1].set_title("depth (tear floor) + sharpness (shoulders)")
    plt.tight_layout(); plt.savefig("outputs/legacy/tear_shoulder.png", dpi=110)
    print("\nsaved tear_shoulder.png")


if __name__ == "__main__":
    main()
