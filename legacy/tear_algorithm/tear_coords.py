"""Anchor-normalized margin depth profiles against two references.

The raw contour index shifts whenever a tear adds perimeter, so "index 400"
is not the same place on two photos. Here every margin is reduced to depth
profiles d(x), where x in [0, 1] is normalized arclength along a reference
envelope between the two anchors (P[0] -> 0, P[-1] -> 1) and d is measured
by a normal scan (below). Both references are built from the OPENED margin
(morphological opening, radius 1.5% of scale) so a single outward
segmentation spur cannot lift the reference and perturb readings elsewhere;
depth is always measured against the original margin. Two references serve
two jobs:

  * convex hull  -- matching. Parameter-free and unique, with no
    bay-follow/bridge bistability across poses. Its profile keeps bays and
    scallops, which are individually distinctive: on the 14-photo /
    7-individual test set it gives top-1 identification 10/11, mean
    within-individual corr 0.88 vs between 0.56. The alpha envelope erases
    exactly that structure (nile's scallops, snap's mound) and scores
    markedly worse for matching at every radius tried (best 0.68 within).
  * alpha hull (radius 20% of ear scale) -- tear measurement. Follows broad
    gentle concavity, so wide shallow bows (delani) stay near zero and a
    peak can be read as "a tear of depth d at x" without bay false
    positives. 20% gives the qualitatively best tear sizes/shapes (nile2,
    delani); values 20-40% are all reasonable.

Coordinate: x is SAMPLED by arclength -- sampling by the pinned pole angle
was tested and costs ~0.2 within-individual correlation, because pose moves
the pole and locally stretches the angle grid. The pole (centroid) and the
pinned angle survive as an ANNOTATION: a secondary axis marks where 0..180
deg falls, so "90" still reads as roughly perpendicular to the anchor
chord. Pinned angle = unwrapped centroid-ray angle of an envelope point,
linearly rescaled so the anchors read exactly 0 and 180.

Depth ("normal scan"), exactly:
  1. sample the envelope path at each grid x (linear interpolation),
  2. local tangent (path smoothed ~18px, central differences) rotated 90
     deg, oriented INWARD by probing 3px along the normal for polygon
     containment (not by pointing at the pole, which is ill-conditioned
     near the anchors),
  3. cast the ray, keep the FIRST margin crossing (smallest t >= 0).
  On intact margin the envelope lies on the margin so depth reads ~0;
  across a bridge the ray drops perpendicular to the tear silhouette --
  parallel-beam rasterization, indifferent to the ear outline's
  non-star-convexity.

Matching recipe (what scored best across the experiments in /tmp): TWO
complementary similarities, plus their fusion.
  1. hull shift-correlation (pair_corr): max Pearson correlation over a
     +-SHIFT_FRAC window. Beat DTW (free warping aligns distractor noise),
     |FFT| (shift-invariance discards the locality that carries identity),
     population mean-removal, and template registration.
  2. gated tear-event matching (tear_events / event_sim): the alpha profile
     reduced to discrete (x, depth) peaks above the GATE noise floor --
     nothing real lives below ~0.005 of scale -- greedily paired within
     EVENT_XTOL. The sparse gated form is what makes the qualitatively-best
     alpha-20 profile also score best on the PILOT SET (17 photos, 8
     individuals -- hypothesis-generating, not validated): top-1 14/14,
     AUC 0.985 with exact assignment (hull corr: 13/14, AUC 0.94; dense
     corr 10/14; DTW 11/14). Gate swept: 0.003-0.005 plateau, 0.008
     destroys real small tears (7/14).
  3. fusion (z-scored mean of 1 + 2; scores share a contour so independence
     is NOT assumed -- weights must be calibrated on held-out data).
Each ear also gets a matchability score (RMS tear depth); ears below the
noise floor are flagged so a match can abstain instead of guessing.

Outputs tear_coords.png (all profiles, hull and alpha panels; same
individual = same color), tear_coords_<label>.png per photo (ear image
with margin + both envelopes + deepest scan ray, beside its profiles), and
prints per-photo match-RMS plus within-individual pair correlations and
top-1 identification (all, and among matchable ears) for both references.
The first/last 5% of x (anchor-region segmentation spurs) are excluded from
stats and shaded in the figures.

NOTE: this script works in bbox-scale units (depth / bounding-box max side);
the production embedding (elephant_id.margins) migrated to rotation-invariant
hull-path-length units. Constants here keep their original bbox calibration.

Run:  uv run python -m scripts.tear_coords
"""
import itertools

import cv2
import matplotlib
import numpy as np
import shapely
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from elephant_id.margins import alpha_shape, densify, ear_side_path, opened_margin
from elephant_id.margins.geometry import ray_crossings
from elephant_id.margins.matching import base_name
from scripts.common import COLORS, LINESTYLES, PHOTOS, make_extractor, out_dir

ALPHA_FRAC = 0.20         # rolling-ball radius for the measurement reference
OPEN_FRAC = 0.015         # morphological opening radius (outward-spur removal)
BINS = 720                # profile resolution
TRIM = 36                 # bins (5% of length per end) excluded from stats
NORMAL_PROBE = 3.0        # px along the normal for the inward test
GATE = 0.005              # pilot-estimated noise floor (17 photos); re-validate
EVENT_XTOL = 0.08         # event-pairing window; covers p90 of measured
                          # matched-event displacement (median .031, max .138)
SHIFT_FRAC = 0.10         # matching shift-search half-window (frac of length)
MATCH_RMS_FLOOR = 0.008   # RMS tear depth below which an ear is "unmatchable"


class EnvelopeFrame:
    """Reference envelope with arclength coordinate and pinned-angle labels.

    ``radius=None`` uses the convex hull (matching reference); otherwise the
    alpha hull with that rolling-ball radius (measurement reference).
    """

    def __init__(self, P: np.ndarray, radius: float | None = None):
        if radius is None:
            shape = shapely.MultiPoint(P).convex_hull
        else:
            shape = alpha_shape(P, radius)
            if shape.is_empty:
                shape = shapely.MultiPoint(P).convex_hull
        self.shape = shape
        env = np.asarray(shape.exterior.coords)[:-1]
        self.path = densify(ear_side_path(env, P[0], P[-1]))
        s = np.concatenate(
            [[0], np.cumsum(np.linalg.norm(np.diff(self.path, axis=0), axis=1))])
        self.s = s / s[-1]
        # pinned pole angle per path sample, for axis annotation only:
        # unwrapped centroid-ray angle rescaled so the anchors read 0 / 180.
        self.pole = np.asarray(shape.centroid.coords[0])
        ref = (P[0] - self.pole) / np.linalg.norm(P[0] - self.pole)
        perp = np.array([-ref[1], ref[0]])
        rel = self.path - self.pole
        th = np.unwrap(np.arctan2(rel @ perp, rel @ ref))
        pinned = (th - th[0]) / (th[-1] - th[0]) * 180.0
        self.backtrack_deg = float(np.clip(-np.diff(pinned), 0, None).sum())
        self.pinned_path = np.maximum.accumulate(pinned)

    def angle_to_arc(self, deg: np.ndarray) -> np.ndarray:
        return np.interp(deg, self.pinned_path, self.s)


def normal_scan(P: np.ndarray, frame: EnvelopeFrame, grid_arc: np.ndarray,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Margin depth below the envelope along its inward normal, per grid x.

    Depth is the NEAREST margin crossing along the ray line (minimum |t|),
    clipped at zero. Where the margin lies on or slightly outside the
    reference (intact stretches; outward spurs shaved off by an opened
    reference) the nearest crossing sits at t <= 0 and depth reads 0;
    across a tear bridge it is the perpendicular drop to the silhouette.
    First-crossing with t >= 0 would be wrong for references that do not
    pass exactly through margin points: origins a hair inside the margin
    would reject the nearby crossing and shoot across the whole ear.
    Returns (depth, origins, normals) so callers can draw the scan rays.
    """
    origins = np.c_[np.interp(grid_arc, frame.s, frame.path[:, 0]),
                    np.interp(grid_arc, frame.s, frame.path[:, 1])]
    sm = uniform_filter1d(frame.path, size=9, axis=0)
    tan = np.gradient(sm, axis=0)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-12
    normals = np.c_[np.interp(grid_arc, frame.s, tan[:, 1]),
                    np.interp(grid_arc, frame.s, -tan[:, 0])]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    probe = origins + NORMAL_PROBE * normals
    inside = shapely.contains_xy(frame.shape, probe[:, 0], probe[:, 1])
    normals[~inside] *= -1

    t, ok = ray_crossings(origins, normals, P)
    tt = np.where(ok, t, np.inf)
    nearest = tt[np.arange(len(tt)), np.argmin(np.abs(tt), axis=1)]
    nearest[~np.isfinite(nearest)] = 0.0
    return np.clip(nearest, 0, None), origins, normals


def pair_corr(d1: np.ndarray, d2: np.ndarray) -> tuple[float, float]:
    """Best correlation over +-SHIFT_FRAC shifts; returns (corr, shift).

    The shift window absorbs anchor-placement offset between photos (measured
    spread +-0.10 of length; a no-shift match scores only 7/14 top-1). It is
    NOT widened past that: a wider window lets a distractor's noise align to
    the query and inflates between-individual similarity -- swept on the test
    set, top-1 peaks at this window (14/14) and falls to 13/14 by +-0.15.
    Content-based registration to a population template was tried and hurts:
    individuals share no common landmark to align to.
    """
    m = int(SHIFT_FRAC * BINS)
    corr = [float(np.corrcoef(d1[TRIM:-TRIM], np.roll(d2, sh)[TRIM:-TRIM])[0, 1])
            for sh in range(-m, m + 1)]
    k = int(np.argmax(corr))
    return corr[k], (k - m) / BINS


def tear_events(d: np.ndarray) -> list[tuple[float, float]]:
    """Discrete tears from the alpha profile interior: (x, depth) peaks.

    Peaks below GATE are segmentation noise and are ignored entirely;
    prominence GATE/2 merges shoulder wiggles into their parent tear.
    """
    core = np.where(d[TRIM:-TRIM] < GATE, 0.0, d[TRIM:-TRIM])
    pk, _ = find_peaks(core, height=GATE, prominence=GATE / 2)
    x = (np.arange(BINS) + 0.5)[TRIM:-TRIM] / BINS
    return [(float(x[p]), float(core[p])) for p in pk]


def event_sim(e1: list[tuple[float, float]], e2: list[tuple[float, float]],
              xtol: float = EVENT_XTOL) -> float:
    """Depth-weighted optimal assignment of two tear-event sets, in [0, 1].

    Maximum-weight one-to-one assignment (Hungarian) with gain
    min(da, db) for pairs within ``xtol``; score = matched depth mass /
    total depth mass (Dice form). Exact assignment beats greedy
    deepest-first on the pilot set (AUC .985 vs .930; per-pair scores
    diverge up to .32). The window is binary on purpose: a smooth
    distance-decaying kernel was tested and HURT (AUC .84-.90), because
    true-match displacement is systematically biased by anchor jitter, so
    distance decay taxes true pairs more than coincidentally-near
    impostor events -- revisit once anchors are fixed. Pilot-set numbers
    throughout (17 photos / 8 individuals): hypothesis-generating, not
    validated.
    """
    t1, t2 = sum(e[1] for e in e1), sum(e[1] for e in e2)
    if t1 + t2 < 1e-9 or not e1 or not e2:
        return 0.0
    gain = np.zeros((len(e1), len(e2)))
    for i, (xa, da) in enumerate(e1):
        for j, (xb, db) in enumerate(e2):
            if abs(xa - xb) <= xtol:
                gain[i, j] = min(da, db)
    ri, ci = linear_sum_assignment(-gain)
    return 2 * float(gain[ri, ci].sum()) / (t1 + t2)


def matchability(depth: np.ndarray) -> float:
    """RMS tear depth over the trimmed profile -- a confidence score.

    Ears below MATCH_RMS_FLOOR have tears at the segmentation noise floor
    (adam, snap2): their profile is noise, no similarity metric recovers
    identity, and a match should ABSTAIN rather than guess. The score is
    monotonic with reliability on the test set; gating at the floor trades
    coverage for precision predictably.
    """
    core = depth[TRIM:-TRIM]
    return float(np.sqrt((core ** 2).mean()))


# -------------------------------- main ----------------------------------- #
def main() -> None:
    extractor = make_extractor()
    out = out_dir("tear_coords")

    grid = (np.arange(BINS) + 0.5) / BINS
    fig_all, (ax_h, ax_a) = plt.subplots(2, 1, figsize=(15, 9))
    prof_h: dict[str, np.ndarray] = {}
    prof_a: dict[str, np.ndarray] = {}
    style_count: dict[str, int] = {}

    for label, ident in PHOTOS.items():
        P = extractor.margin(ident)
        if P is None:
            print(f"{label}: no ears")
            continue
        scale = float(np.max(P.max(0) - P.min(0)))
        src = opened_margin(P, OPEN_FRAC * scale)
        frame_h = EnvelopeFrame(src)
        frame_a = EnvelopeFrame(src, ALPHA_FRAC * scale)
        for f, nm in [(frame_h, "hull"), (frame_a, "alpha")]:
            if f.backtrack_deg > 0.1:
                print(f"  WARNING {label}: {nm} pinned angle backtracks "
                      f"{f.backtrack_deg:.2f}deg (annotation only)")

        d_h, origins, normals = normal_scan(P, frame_h, grid)
        d_a = normal_scan(P, frame_a, grid)[0] / scale
        d_hn = d_h / scale
        prof_h[label], prof_a[label] = d_hn, d_a
        k = int(np.argmax(d_a[TRIM:-TRIM])) + TRIM   # tear peak: alpha ref
        kh = int(np.argmax(d_hn[TRIM:-TRIM])) + TRIM
        rms = matchability(d_hn)
        gate = "" if rms >= MATCH_RMS_FLOOR else "  [below match floor]"
        print(f"{label:8s} tear peak {100 * d_a[k]:5.1f}% at x={grid[k]:.3f} "
              f"({frame_a.pinned_path[np.searchsorted(frame_a.s, grid[k])]:.0f}deg)"
              f"  match-RMS {100 * rms:.2f}%{gate}")

        base = base_name(label)
        ls = LINESTYLES[style_count.get(base, 0) % len(LINESTYLES)]
        style_count[base] = style_count.get(base, 0) + 1
        ax_h.plot(grid, d_hn, ls=ls, color=COLORS[base], lw=1.2, label=label)
        ax_a.plot(grid, d_a, ls=ls, color=COLORS[base], lw=1.2, label=label)

        # per-photo figure: ear image with overlays | both profiles
        crop = extractor.crop(ident)
        if crop is None:
            continue
        img, off = crop
        fig, (axi, axp) = plt.subplots(
            1, 2, figsize=(16, 7.5), width_ratios=[1, 1.25])
        axi.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axi.plot(*(P - off).T, "w", lw=1.2, alpha=0.9, label="margin")
        axi.plot(*(frame_h.path - off).T, "tab:cyan", lw=1.5, label="convex hull")
        axi.plot(*(frame_a.path - off).T, "yellow", lw=1.1, alpha=0.9,
                 label=f"alpha-{int(100 * ALPHA_FRAC)}")
        hit = origins[kh] + d_h[kh] * normals[kh]
        axi.plot([origins[kh, 0] - off[0], hit[0] - off[0]],
                 [origins[kh, 1] - off[1], hit[1] - off[1]],
                 "r-", lw=2.0, label=f"deepest scan (x={grid[kh]:.2f})")
        axi.legend(fontsize=9, loc="lower right")
        axi.set_title(f"{label} ({ident})", fontsize=12)
        axi.axis("off")

        axp.plot(grid, d_hn, color=COLORS[base], lw=1.4, label="hull (matching)")
        axp.plot(grid, d_a, color="0.35", lw=1.2, label="alpha (tear measurement)")
        axp.axvspan(0, TRIM / BINS, color="0.85")
        axp.axvspan(1 - TRIM / BINS, 1, color="0.85")
        axp.set_xlim(0, 1)
        axp.set_ylim(0, max(1.3 * max(d_hn[kh], d_a[k]), 0.05))
        axp.set_xlabel("normalized envelope arclength")
        axp.set_ylabel("depth / scale")
        axp.grid(alpha=0.3)
        axp.legend(fontsize=9)
        # pinned-angle annotation: where 0..180deg falls on the arclength axis
        ticks = np.arange(0, 181, 30)
        top = axp.secondary_xaxis("top")
        top.set_xticks(frame_h.angle_to_arc(ticks))
        top.set_xticklabels([f"{t:d}" for t in ticks], fontsize=8)
        top.set_xlabel("pinned pole angle (deg)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{label}.png", dpi=110)
        plt.close(fig)
        print(f"  saved {out}/{label}.png")

    # within-individual consistency + top-1 identification, both references
    by_base: dict[str, list[str]] = {}
    for label in prof_h:
        by_base.setdefault(base_name(label), []).append(label)
    multi = [x for x in prof_h if len(by_base[base_name(x)]) > 1]
    matchable = [x for x in multi if matchability(prof_h[x]) >= MATCH_RMS_FLOOR]
    for name, prof in [("hull", prof_h), ("alpha", prof_a)]:
        for labels in by_base.values():
            for a, b in itertools.combinations(labels, 2):
                c, sh = pair_corr(prof[a], prof[b])
                print(f"{name:5s} pair {a}/{b}: corr {c:.3f} at shift {sh:+.3f}")
        hits = gated_hits = 0
        for a in multi:
            ranked = sorted((x for x in prof if x != a),
                            key=lambda x: -pair_corr(prof[a], prof[x])[0])
            ok = base_name(ranked[0]) == base_name(a)
            hits += ok
            gated_hits += ok and a in matchable
            if not ok:
                print(f"{name} top-1 MISS {a}: best match {ranked[0]}")
        print(f"{name} top-1: {hits}/{len(multi)} all, "
              f"{gated_hits}/{len(matchable)} above match floor")

    # gated tear-event matching on the alpha profile, and fusion with hull
    events = {lab: tear_events(prof_a[lab]) for lab in prof_a}
    pairs = list(itertools.combinations(prof_a, 2))
    s_ev = {p: event_sim(events[p[0]], events[p[1]]) for p in pairs}
    s_hc = {p: pair_corr(prof_h[p[0]], prof_h[p[1]])[0] for p in pairs}
    s_fused = {}
    for S, out_w in [(s_ev, 1.0), (s_hc, 1.0)]:
        v = np.array([S[p] for p in pairs])
        mu, sd = float(v.mean()), float(v.std()) + 1e-12
        for p in pairs:
            s_fused[p] = s_fused.get(p, 0.0) + out_w * (S[p] - mu) / sd / 2
    for labels in by_base.values():
        for a, b in itertools.combinations(labels, 2):
            p = (a, b) if (a, b) in s_ev else (b, a)
            print(f"event pair {a}/{b}: sim {s_ev[p]:.3f} "
                  f"({len(events[a])}/{len(events[b])} events)")
    for name, S in [("event", s_ev), ("fused", s_fused)]:
        sym = dict(S)
        sym.update({(b, a): v for (a, b), v in S.items()})
        hits = gated_hits = 0
        for a in multi:
            ranked = sorted((x for x in prof_a if x != a),
                            key=lambda x: -sym[(a, x)])
            ok = base_name(ranked[0]) == base_name(a)
            hits += ok
            gated_hits += ok and a in matchable
            if not ok:
                print(f"{name} top-1 MISS {a}: best match {ranked[0]}")
        print(f"{name} top-1: {hits}/{len(multi)} all, "
              f"{gated_hits}/{len(matchable)} above match floor")

    ax_h.set_title("convex hull reference (matching)", fontsize=12)
    ax_a.set_title(f"alpha-{int(100 * ALPHA_FRAC)} reference (tear measurement)",
                   fontsize=12)
    for ax in (ax_h, ax_a):
        ax.set_xlabel("normalized envelope arclength")
        ax.set_ylabel("depth / scale")
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.3)
    fig_all.tight_layout()
    fig_all.savefig(out / "summary.png", dpi=110)
    print(f"saved {out}/summary.png")


if __name__ == "__main__":
    main()
