"""Regenerate the diagrams for docs/tear-embedding.md (decision record).

Each figure validates one key design decision with real data:
  tear_embedding_pipeline.png    the settled pipeline, end to end (nile2)
  tear_embedding_references.png  why alpha hull, not convex hull or arPLS
  tear_embedding_pole.png        why normal scan, not a polar raster
  tear_embedding_opening.png     why the light morphological opening

Run:  uv run python -m scripts.tear_doc_figures
"""
import importlib.util

import cv2
import matplotlib
import numpy as np
import shapely

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from elephant_id.coding.tears import PROFILE_GRID, hull_arclength, tear_profile
from elephant_id.constants import (
    TEAR_ALPHA_FRAC,
    TEAR_OPEN_FRAC,
    TEAR_TRIM_HI,
    TEAR_TRIM_LO,
)
from elephant_id.geometry import (
    alpha_shape,
    densify,
    ear_side_path,
    inward_normals,
    nearest_crossing,
    opened_contour,
)
from scripts.evaluate import PHOTOS, REPO_ROOT, make_extractor, tear_events

ASSETS = REPO_ROOT / "docs" / "assets"


def _legacy_arpls():
    """Import baseline_arpls from the frozen exploration snapshot."""
    spec = importlib.util.spec_from_file_location(
        "tear_baseline",
        REPO_ROOT / "legacy/tear_algorithm/tear_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.baseline_arpls


def fig_pipeline(extractor) -> None:
    """The settled pipeline on nile2: reference, scan rays, profile."""
    ident = PHOTOS["nile2"]
    P = extractor.contour(ident)
    img, off = extractor.crop(ident)
    res = tear_profile(P)
    events = tear_events(res.profile)

    fig, (axi, axp) = plt.subplots(1, 2, figsize=(15, 6.5),
                                   width_ratios=[1, 1.3])
    axi.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axi.plot(*(P - off).T, "w", lw=1.2, label="margin (input)")
    axi.plot(*(res.reference - off).T, "tab:cyan", lw=1.8,
             label="intact-margin reference (alpha hull)")
    for k in range(0, len(res.origins), 24):
        d = res.profile[k] * res.scale
        if d < 1:
            continue
        hit = res.origins[k] + d * res.normals[k]
        axi.plot([res.origins[k, 0] - off[0], hit[0] - off[0]],
                 [res.origins[k, 1] - off[1], hit[1] - off[1]],
                 "r-", lw=0.9, alpha=0.8)
    axi.plot([], [], "r-", lw=0.9, label="depth scan rays")
    axi.legend(fontsize=9, loc="lower right")
    axi.set_title("margin -> opening -> alpha reference -> normal depth scan",
                  fontsize=11)
    axi.axis("off")

    axp.plot(PROFILE_GRID, res.profile, "tab:red", lw=1.4)
    for x, d in events:
        axp.plot(x, d, "kv", ms=6)
    axp.axvspan(0, TEAR_TRIM_LO, color="0.88")
    axp.axvspan(1 - TEAR_TRIM_HI, 1, color="0.88")
    axp.set_xlim(0, 1)
    axp.set_xlabel("normalized position along the intact margin")
    axp.set_ylabel("tear depth (fraction of ear scale)")
    axp.set_title("output: the ear unrolled flat -- one bump per tear",
                  fontsize=11)
    axp.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "tear_embedding_pipeline.png", dpi=110)
    plt.close(fig)


def fig_references(extractor) -> None:
    """Convex hull over-bridges bays (delani); arPLS bleeds (ripley)."""
    baseline_arpls = _legacy_arpls()
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    P = extractor.contour(PHOTOS["delani"])
    hull = np.asarray(shapely.MultiPoint(P).convex_hull.exterior.coords)[:-1]
    hpath = ear_side_path(hull, P[0], P[-1])
    apath = tear_profile(P).reference
    axes[0].plot(*P.T, "0.4", lw=1.0, label="margin (intact, gently bowed)")
    axes[0].plot(*hpath.T, "tab:red", lw=1.6,
                 label="convex hull: bridges the bay -> false tear")
    axes[0].plot(*apath.T, "tab:cyan", lw=1.6,
                 label="alpha hull: follows the bay")
    axes[0].set_title("delani: why not the convex hull", fontsize=11)

    P = extractor.contour(PHOTOS["ripley"])
    apath = tear_profile(P).reference
    arpls = baseline_arpls(P)
    axes[1].plot(*P.T, "0.4", lw=1.0, label="margin (one massive tear)")
    axes[1].plot(*arpls.T, "tab:orange", lw=1.6,
                 label="arPLS baseline: bleeds into the tear")
    axes[1].plot(*apath.T, "tab:cyan", lw=1.6,
                 label="alpha hull: bridges it cleanly")
    axes[1].set_title("ripley: why not a smooth baseline (arPLS)", fontsize=11)

    for ax in axes:
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.legend(fontsize=9, loc="lower right")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ASSETS / "tear_embedding_references.png", dpi=110)
    plt.close(fig)


def fig_pole(extractor) -> None:
    """Polar rasters fail: non-star-convex margin, jagged radial profiles."""
    P = extractor.contour(PHOTOS["ripley"])
    S = hull_arclength(P)
    shape = alpha_shape(opened_contour(P, TEAR_OPEN_FRAC * S),
                        TEAR_ALPHA_FRAC * S)
    pole = np.asarray(shape.centroid.coords[0])
    mid = (P[0] + P[-1]) / 2

    # find a midpoint-pole ray with >= 3 margin crossings (non-star-convexity)
    def crossings(pole_pt, direction):
        A, E = P[:-1], P[1:] - P[:-1]
        rel = A - pole_pt
        den = direction[0] * E[:, 1] - direction[1] * E[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (rel[:, 0] * E[:, 1] - rel[:, 1] * E[:, 0]) / den
            u = (rel[:, 0] * direction[1] - rel[:, 1] * direction[0]) / den
        ok = (u >= 0) & (u <= 1) & np.isfinite(t) & (t > 0)
        return np.sort(t[ok])

    fig, (axg, axp) = plt.subplots(1, 2,figsize=(13, 6.5))

    ang = np.pi * 1.98
    best_dir = np.array([np.cos(ang), np.sin(ang)])
    best_n = len(crossings(mid, best_dir))

    axg.plot(*P.T, "0.4", lw=1.0, label="margin")
    ts = crossings(mid, best_dir)
    axg.plot([mid[0], mid[0] + best_dir[0] * ts[-1] * 1.05],
             [mid[1], mid[1] + best_dir[1] * ts[-1] * 1.05],
             "tab:red", lw=1.4,
             label=f"one ray, {best_n} margin crossings")
    hits = mid + best_dir * ts[:, None]
    axg.plot(hits[:, 0], hits[:, 1], "ro", ms=6)
    axg.plot(*mid, "y*", ms=16, markeredgecolor="k",
             label="pole at anchor midpoint")
    axg.plot(*pole, "g*", ms=16, markeredgecolor="k",
             label="pole at centroid")
    axg.set_aspect("equal")
    axg.invert_yaxis()
    axg.legend(fontsize=9, loc="lower right")
    axg.set_title("ripley: not star convex from the anchor midpoint",
                  fontsize=11)
    axg.axis("off")

    ang = np.pi * 1.9
    best_dir = np.array([np.cos(ang), np.sin(ang)])
    best_n = len(crossings(pole, best_dir))

    axp.plot(*P.T, "0.4", lw=1.0, label="margin")
    ts = crossings(pole, best_dir)
    axp.plot([pole[0], pole[0] + best_dir[0] * ts[-1] * 1.05],
             [pole[1], pole[1] + best_dir[1] * ts[-1] * 1.05],
             "tab:red", lw=1.4,
             label=f"one ray, {best_n} margin crossings")
    hits = pole + best_dir * ts[:, None]
    axp.plot(hits[:, 0], hits[:, 1], "ro", ms=6)
    axp.plot(*mid, "y*", ms=16, markeredgecolor="k",
             label="pole at anchor midpoint")
    axp.plot(*pole, "g*", ms=16, markeredgecolor="k",
             label="pole at centroid")
    axp.set_aspect("equal")
    axp.invert_yaxis()
    axp.legend(fontsize=9, loc="lower right")
    axp.set_title("ripley: not star convex from the centroid",
                  fontsize=11)
    axp.axis("off")

    fig.tight_layout()
    fig.savefig(ASSETS / "tear_embedding_pole.png", dpi=110)
    plt.close(fig)


def fig_opening(extractor) -> None:
    """A synthetic outward spur perturbs the reference unless opened away."""
    P0 = extractor.contour(PHOTOS["les"])
    S = hull_arclength(P0)
    # inject a small outward spur on an intact stretch
    i0, w, amp = 300, 5, 0.012 * S
    tang = np.gradient(P0, axis=0)
    nrm = np.c_[tang[:, 1], -tang[:, 0]]
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    inside = shapely.contains_xy(shapely.Polygon(P0), *(P0 + 3 * nrm).T)
    nrm[inside] *= -1
    P1 = P0.copy()
    for j in range(-w, w + 1):
        P1[i0 + j] += nrm[i0 + j] * amp * (1 - abs(j) / (w + 1))

    def profile_of(P, opening: bool):
        src = opened_contour(P, TEAR_OPEN_FRAC * S) if opening else P
        shape = alpha_shape(src, TEAR_ALPHA_FRAC * S)
        path = densify(ear_side_path(
            np.asarray(shape.exterior.coords)[:-1], src[0], src[-1]))
        o, n = inward_normals(path, shape, PROFILE_GRID)
        d = nearest_crossing(o, n, P) / S
        d[:int(TEAR_TRIM_LO * len(d))] = 0   # standard anchor-zone trims
        d[-int(TEAR_TRIM_HI * len(d)):] = 0
        return d, path

    d_clean, _ = profile_of(P0, opening=True)
    d_no, path_no = profile_of(P1, opening=False)
    d_yes, path_yes = profile_of(P1, opening=True)

    fig, (axg, axp) = plt.subplots(1, 2, figsize=(15, 6.5),
                                   width_ratios=[1, 1.3])
    axg.plot(*P1.T, "0.4", lw=1.0, label="margin with a segmentation spur")
    axg.plot(*path_no.T, "tab:red", lw=1.4,
             label="reference without opening: lifted by the spur")
    axg.plot(*path_yes.T, "tab:cyan", lw=1.4,
             label="reference with opening: unaffected")
    z = P0[i0]
    axg.set_xlim(z[0] - 0.25 * S, z[0] + 0.25 * S)
    axg.set_ylim(z[1] + 0.25 * S, z[1] - 0.25 * S)
    axg.set_aspect("equal")
    axg.legend(fontsize=9, loc="lower right")
    axg.set_title("les + synthetic spur (zoom)", fontsize=11)
    axg.axis("off")

    axp.plot(PROFILE_GRID, d_clean, "0.6", lw=2.4, label="clean margin")
    axp.plot(PROFILE_GRID, d_no, "tab:red", lw=1.0,
             label="spurred, no opening: collateral readings")
    axp.plot(PROFILE_GRID, d_yes, "tab:cyan", lw=1.0, ls="--",
             label="spurred, with opening")
    axp.set_xlim(0, 1)
    axp.set_xlabel("normalized position")
    axp.set_ylabel("tear depth (fraction of ear scale)")
    axp.legend(fontsize=9)
    axp.grid(alpha=0.3)
    axp.set_title("one outward spur perturbs depths elsewhere", fontsize=11)
    fig.tight_layout()
    fig.savefig(ASSETS / "tear_embedding_opening.png", dpi=110)
    plt.close(fig)


def main() -> None:
    extractor = make_extractor()
    for fig in (fig_pipeline, fig_references, fig_pole, fig_opening):
        fig(extractor)
        print(f"{fig.__name__} done")
    print(f"saved to {ASSETS}/tear_embedding_*.png")


if __name__ == "__main__":
    main()
