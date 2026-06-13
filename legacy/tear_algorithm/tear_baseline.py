"""Prototype: robust, deterministic tear detection via an outer-envelope baseline.

This is an exploration script, not package code. It compares three baselines on
the cut ear margin:

    1. arPLS (current approach in scripts/arPLS.py): asymmetric reweighted
       Whittaker. Tear points keep a small but NONZERO weight, so a wide tear
       pulls the baseline inward ("bleed").
    2. Hard-exclusion + lambda annealing (proposed): tear points get EXACTLY
       zero weight, and we anneal the smoothness from stiff -> flexible so the
       baseline can never fall into a wide tear, then hugs the intact margin to
       expose narrow ones.
    3. Convex-hull seed (safeguard): guarantees the baseline is pinned to the
       outer margin on both shoulders of any wide tear.

Run:  uv run python scripts/tear_baseline.py
"""
import json

import matplotlib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Shared Whittaker machinery (open contour: endpoints are the anchor points).
# --------------------------------------------------------------------------- #
def _DtD(n: int) -> sparse.csc_matrix:
    """Second-difference operator D^T D for an open polyline."""
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    return (D.T @ D).tocsc()


def _solve(y: np.ndarray, w: np.ndarray, lam: float, DtD) -> np.ndarray:
    """Weighted Whittaker solve: argmin sum w (y-z)^2 + lam ||D z||^2."""
    A = (sparse.diags(w) + lam * DtD).tocsc()
    return spsolve(A, w * y)


def _fit_xy(P: np.ndarray, w: np.ndarray, lam: float, DtD) -> np.ndarray:
    return np.stack([_solve(P[:, 0], w, lam, DtD),
                     _solve(P[:, 1], w, lam, DtD)], axis=1)


def _normals(S: np.ndarray) -> np.ndarray:
    """Unit outward normals along the baseline (outward = away from centroid)."""
    t = np.gradient(S, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    nrm = np.stack([t[:, 1], -t[:, 0]], axis=1)
    flip = np.sign(np.einsum("ij,ij->i", nrm, S - S.mean(0)))
    flip[flip == 0] = 1.0
    return nrm * flip[:, None]


def _signed_dev(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Signed normal deviation of margin from baseline. Inward (tear) < 0."""
    return np.einsum("ij,ij->i", P - S, _normals(S))


def _mad_band(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))) + 1e-9)


# --------------------------------------------------------------------------- #
# 1. arPLS reference (asymmetric reweighting; soft tear handling).
# --------------------------------------------------------------------------- #
def baseline_arpls(P: np.ndarray, lam: float = 1e5, iters: int = 30) -> np.ndarray:
    n = len(P)
    DtD = _DtD(n)
    w = np.ones(n)
    for _ in range(iters):
        S = _fit_xy(P, w, lam, DtD)
        d = _signed_dev(P, S)
        r = -d
        pk = r > 0
        dn = r[pk]
        m = dn.mean() if dn.size else 0.0
        s = dn.std() if dn.size else 1.0
        w_new = 1.0 / (1.0 + np.exp(2.0 * (r - (-m + 2 * s)) / (s + 1e-12)))
        if np.linalg.norm(w_new - w) / (np.linalg.norm(w) + 1e-12) < 1e-2:
            w = w_new
            break
        w = w_new
    return _fit_xy(P, w, lam, DtD)


# --------------------------------------------------------------------------- #
# 2+3. Proposed: hard-exclusion baseline with lambda annealing + hull seed.
# --------------------------------------------------------------------------- #
def _hull_seed(P: np.ndarray, tol_frac: float = 0.01) -> np.ndarray:
    """Boolean mask of margin points on/near the convex hull -> guaranteed intact.

    A wide tear's floor lies strictly inside the hull, so it is never seeded;
    its two shoulders lie on the hull, pinning the baseline outside the tear.
    """
    hull = ConvexHull(P)
    diag = np.linalg.norm(P.max(0) - P.min(0))
    tol = tol_frac * diag
    seed = np.zeros(len(P), bool)
    # distance from every point to every hull facet (2D: edges)
    for eq in hull.equations:  # a*x + b*y + c = 0, outward normalized
        dist = np.abs(P @ eq[:2] + eq[2])
        seed |= dist < tol
    return seed


def baseline_hard(
    P: np.ndarray,
    lam_hi: float = 3e6,
    lam_lo: float = 3e4,
    n_anneal: int = 6,
    k_mad: float = 3.0,
    min_px: float = 3.0,
    inner_iters: int = 12,
    use_hull: bool = True,
) -> dict:
    """Outer-envelope baseline: tear points get exactly zero data weight.

    Annealing lam_hi -> lam_lo means the first baseline is too stiff to enter
    any tear (so every tear is exposed and excluded), and later baselines relax
    to follow the intact margin's natural curvature while the already-excluded
    tear spans stay excluded. Tear width is irrelevant: 0 weight x any width = 0.
    """
    n = len(P)
    DtD = _DtD(n)
    seed = _hull_seed(P) if use_hull else np.zeros(n, bool)
    intact = np.ones(n, bool)

    for lam in np.geomspace(lam_hi, lam_lo, n_anneal):
        for _ in range(inner_iters):
            w = intact.astype(float)
            w[0] = w[-1] = 1.0  # anchors are always intact
            S = _fit_xy(P, w, lam, DtD)
            d = _signed_dev(P, S)
            band = _mad_band(d[intact])
            thr = -max(k_mad * band, min_px)
            new_intact = (d > thr) | seed
            new_intact[0] = new_intact[-1] = True
            if np.array_equal(new_intact, intact):
                intact = new_intact
                break
            intact = new_intact

    d = _signed_dev(P, S)
    band = _mad_band(d[intact])
    thr = -max(k_mad * band, min_px)
    is_tear = (d < thr) & ~seed
    return dict(S=S, dev=d, is_tear=is_tear, thr=thr, band=band, intact=intact)


def tear_segments(is_tear: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous (start, end) runs of True in an open boolean mask."""
    out, start = [], None
    for i, v in enumerate(is_tear):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(is_tear) - 1))
    return out


def measure(P: np.ndarray, dev: np.ndarray, segs: list[tuple[int, int]],
            ear_scale: float) -> list[dict]:
    ds = np.linalg.norm(np.diff(P, axis=0), axis=1)
    ds = np.append(ds, ds[-1])
    out = []
    for s, e in segs:
        idx = np.arange(s, e + 1)
        depth = float(-dev[idx].min())
        out.append(dict(
            start=s, end=e, n=len(idx),
            depth_px=depth,
            width_px=float(ds[idx].sum()),
            area_px2=float(np.sum(-dev[idx] * ds[idx])),
            depth_frac=depth / ear_scale,  # for SEEK 1/4 "extreme" test
        ))
    return sorted(out, key=lambda t: -t["depth_px"])


# --------------------------------------------------------------------------- #
# Synthetic wide tear: carve a broad inward notch into a smooth margin span.
# --------------------------------------------------------------------------- #
def carve_wide_tear(P: np.ndarray, center: int, half_width: int,
                    depth_px: float) -> np.ndarray:
    Q = P.copy().astype(float)
    n = len(Q)
    lo, hi = center - half_width, center + half_width
    idx = np.arange(lo, hi)
    # inward direction = toward centroid
    inward = Q.mean(0) - Q[center]
    inward /= np.linalg.norm(inward) + 1e-12
    # smooth bump (raised cosine) so the notch has shoulders, like a real bite
    bump = 0.5 * (1 + np.cos(np.linspace(-np.pi, np.pi, len(idx))))
    Q[idx % n] += np.outer(depth_px * bump, inward)
    return Q


# --------------------------------------------------------------------------- #
def ear_scale_estimate(P: np.ndarray) -> float:
    """Medial extent of the ear: max distance from the anchor chord. Proxy for
    'inner ear' used by the SEEK extreme (>= 1/4) test."""
    a, b = P[0], P[-1]
    ab = b - a
    L = np.linalg.norm(ab) + 1e-12
    nrm = np.array([ab[1], -ab[0]]) / L
    return float(np.abs((P - a) @ nrm).max())


def main() -> None:
    P = np.array(json.load(open("data/contour.json")), float)
    ear_scale = ear_scale_estimate(P)
    print(f"contour: {len(P)} pts   ear medial scale: {ear_scale:.0f}px\n")

    cases = {
        "real margin": P,
        "synthetic WIDE tear (180pts, depth 90px)":
            carve_wide_tear(P, center=300, half_width=90, depth_px=90.0),
    }

    fig, axes = plt.subplots(len(cases), 2, figsize=(15, 6 * len(cases)))
    if len(cases) == 1:
        axes = axes[None]

    for row, (name, Pc) in enumerate(cases.items()):
        S_arpls = baseline_arpls(Pc)
        d_arpls = _signed_dev(Pc, S_arpls)
        res = baseline_hard(Pc)

        # tears from each method, measured against ITS OWN baseline
        band_a = _mad_band(d_arpls[20:-20])
        thr_a = -max(3 * band_a, 3.0)
        tear_a = d_arpls < thr_a
        tear_a[:8] = tear_a[-8:] = False

        segs_h = tear_segments(res["is_tear"])
        stats_h = measure(Pc, res["dev"], segs_h, ear_scale)

        print(f"=== {name} ===")
        print(f"  arPLS    : deepest inward dev = {(-d_arpls).max():6.1f}px  "
              f"(baseline bled if << injected depth)")
        print(f"  proposed : {len(stats_h)} tears")
        for t in stats_h[:4]:
            flag = "  <EXTREME>" if t["depth_frac"] >= 0.25 else ""
            print(f"      depth {t['depth_px']:6.1f}px "
                  f"({t['depth_frac']*100:4.1f}% of ear)  "
                  f"width {t['width_px']:6.1f}px  idx {t['start']}-{t['end']}{flag}")
        print()

        axc = axes[row][0]
        axc.plot(*Pc.T, color="0.6", lw=1.0, label="raw margin")
        axc.plot(*S_arpls.T, "--", color="tab:orange", lw=1.8, label="arPLS baseline")
        axc.plot(*res["S"].T, "-", color="tab:blue", lw=2.0, label="proposed baseline")
        for s, e in segs_h:
            axc.plot(*Pc[s:e + 1].T, "m-", lw=4, alpha=0.7)
        axc.plot(*Pc[[0, -1]].T, "kx", ms=9)
        axc.set_aspect("equal"); axc.invert_yaxis()
        axc.legend(loc="lower left", fontsize=8)
        axc.set_title(f"{name}\nmargin vs baselines")

        axd = axes[row][1]
        axd.plot(d_arpls, color="tab:orange", lw=1.0, label="arPLS deviation")
        axd.plot(res["dev"], color="tab:blue", lw=1.0, label="proposed deviation")
        axd.axhline(res["thr"], color="r", ls="--", lw=0.8, label="proposed threshold")
        axd.axhline(0, color="0.5", lw=0.5)
        axd.fill_between(np.arange(len(Pc)), res["dev"], 0,
                         where=res["is_tear"], color="m", alpha=0.3)
        axd.set_title("signed normal deviation (neg = inward = tear)")
        axd.set_xlabel("contour index"); axd.set_ylabel("px")
        axd.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/legacy/tear_baseline_compare.png", dpi=110)
    print("saved tear_baseline_compare.png")


if __name__ == "__main__":
    main()
