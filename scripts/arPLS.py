"""Tear-free elephant-ear baseline via spectroscopy baseline correction.

The mapping (see module docstring notes):
    spectrum intensity y(lambda)   <->  signed normal deviation d(s) of the edge
    wavelength axis lambda          <->  arc length s along the closed contour
    smooth fluorescence baseline    <->  intact (tear-free) ear edge
    sharp one-sided peaks           <->  tears (always INWARD excursions)
    peak detection / integration    <->  tear localisation / deficit area

Engine is the Whittaker smoother (Eilers 2003). On top of it we provide the
three canonical asymmetric reweighting schemes from the Raman/IR literature:
    AsLS   - Eilers & Boelens 2005   (manual asymmetry p)
    airPLS - Zhang, Chen & Liang 2010 (adaptive, magnitude-weighted)
    arPLS  - Baek, Park, Ahn & Choi 2015 (auto threshold from noise stats)

Two differences from a benchtop spectrometer, both handled here:
  1. A spectrum has endpoints; a closed contour is PERIODIC -> circulant D.
  2. A spectrum is a 1-D function of lambda; the ear is a 2-D curve. We keep
     the weighting decision 1-D (on the normal-deviation signal, exactly as in
     spectroscopy) and apply the resulting weights when re-smoothing x(s),y(s).
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _DtD(n, closed):
    """Second-difference operator D^T D.

    closed=True  -> circulant (periodic), for a true closed loop.
    closed=False -> open with natural boundaries, exactly as in spectroscopy
                    (a spectrum has two endpoints; do NOT wrap them together).
    """
    if closed:
        idx = np.arange(n)
        rows = np.concatenate([idx, idx, idx])
        cols = np.concatenate([idx, (idx + 1) % n, (idx + 2) % n])
        vals = np.concatenate([np.ones(n), -2 * np.ones(n), np.ones(n)])
        D = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    else:
        D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    return (D.T @ D).tocsc()


def _whittaker(y, w, lam, DtD):
    """One weighted Whittaker solve: min  sum w(y-z)^2 + lam ||D z||^2."""
    A = (sparse.diags(w) + lam * DtD).tocsc()
    return spsolve(A, w * y)


def _normals(S, closed):
    if closed:
        t = np.roll(S, -1, 0) - np.roll(S, 1, 0)
    else:
        t = np.gradient(S, axis=0)            # one-sided at the two endpoints
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    nrm = np.stack([t[:, 1], -t[:, 0]], axis=1)
    flip = np.sign(np.einsum("ij,ij->i", nrm, S - S.mean(0)))
    flip[flip == 0] = 1.0
    return nrm * flip[:, None]  # outward-pointing


def _weights(d, scheme, it, p):
    """d = signed normal deviation (inward tears are NEGATIVE).
    Convert to spectroscopy convention r = -d (peaks POSITIVE above baseline)."""
    r = -d
    pk = r > 0  # candidate "peak" (inward) samples
    if scheme == "asls":
        w = np.where(pk, p, 1.0 - p)
    elif scheme == "airpls":
        if not pk.any():
            return np.ones_like(d)
        s = r[pk].sum()
        w = np.ones_like(d)
        w[pk] = np.exp(-it * r[pk] / (s + 1e-12))
        w[r >= r.max()] = 0.0
    elif scheme == "arpls":
        dn = r[pk]                       # positive residuals = above baseline
        m = dn.mean() if dn.size else 0.0
        s = dn.std() if dn.size else 1.0
        w = 1.0 / (1.0 + np.exp(2.0 * (r - (-m + 2 * s)) / (s + 1e-12)))
    else:
        raise ValueError(scheme)
    return w


def smooth_intact_ear(contour, lam=1e5, scheme="arpls", p=1e-2, iters=20,
                      tol=1e-2, closed=False):
    """Fit a smooth, tear-free baseline to a closed resampled ear contour.

    contour : (N,2) closed, arc-length-resampled points.
    lam     : Whittaker smoothness (larger = stiffer baseline).
    scheme  : 'arpls' (auto, recommended), 'airpls', or 'asls'.
    p       : asymmetry, used only by 'asls'.
    iters   : max reweighting iterations.
    tol     : convergence tol on relative weight change.

    Returns (intact, dev, weight):
      intact (N,2) tear-free baseline (rides the OUTER edge, bridges tears)
      dev    (N,)  signed normal deviation, input minus baseline (- = inward)
      weight (N,)  final per-point weight (low = treated as tear)
    """
    P = np.asarray(contour, float)
    n = len(P)
    DtD = _DtD(n, closed)
    w = np.ones(n)
    S = P.copy()
    w_prev = w
    for it in range(1, iters + 1):
        S = np.stack([_whittaker(P[:, 0], w, lam, DtD),
                      _whittaker(P[:, 1], w, lam, DtD)], axis=1)
        dev = np.einsum("ij,ij->i", P - S, _normals(S, closed))
        w = _weights(dev, scheme, it, p)
        if np.linalg.norm(w - w_prev) / (np.linalg.norm(w_prev) + 1e-12) < tol:
            break
        w_prev = w
    S = np.stack([_whittaker(P[:, 0], w, lam, DtD),
                  _whittaker(P[:, 1], w, lam, DtD)], axis=1)
    dev = np.einsum("ij,ij->i", P - S, _normals(S, closed))
    return S, dev, w


def tear_segments(is_tear):
    """Contiguous (start, end) index runs in a periodic boolean mask."""
    n = len(is_tear)
    if is_tear.all():
        return [(0, n - 1)]
    if not is_tear.any():
        return []
    start = np.where(is_tear & ~np.roll(is_tear, 1))[0]
    end = np.where(is_tear & ~np.roll(is_tear, -1))[0]
    out = []
    for s in start:
        e = end[np.searchsorted(end, s) % len(end)]
        out.append((int(s), int(e)))
    return out

# ---------------------------------------------------------------------------
# Multiscale extension: spatially adaptive lambda(s).
#
# A single global lambda cannot bridge a WIDE tear and still hug the edge to
# expose NARROW tears. The spectroscopy analogue is a sharp peak on a broad
# band: remove the broad part first, resolve the sharp part on the flattened
# residual. Here we do it in one final baseline by making the smoothness
# penalty LARGE only across wide-tear spans (found in a stiff coarse pass) and
# SMALL elsewhere. lambda becomes a function of arc length, lambda(s).
# ---------------------------------------------------------------------------

def _DtD_var(lam_point, closed):
    """Second-difference operator with per-point smoothness lambda(s).

    Penalty = z^T D^T diag(lam_row) D z, with lam_row taken from lam_point.
    lam_point : (N,) per-point smoothness weights.
    """
    n = len(lam_point)
    if closed:
        idx = np.arange(n)
        rows = np.concatenate([idx, idx, idx])
        cols = np.concatenate([idx, (idx + 1) % n, (idx + 2) % n])
        vals = np.concatenate([np.ones(n), -2 * np.ones(n), np.ones(n)])
        D = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        lam_row = lam_point
    else:
        D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n)).tocsr()
        lam_row = lam_point[1:-1]
    return (D.T @ sparse.diags(lam_row) @ D).tocsc()


def _arpls_weights(dev):
    r = -dev
    pk = r > 0
    dn = r[pk]
    m = dn.mean() if dn.size else 0.0
    s = dn.std() if dn.size else 1.0
    return 1.0 / (1.0 + np.exp(2.0 * (r - (-m + 2 * s)) / (s + 1e-12)))


def smooth_var(contour, lam_point, iters=30, tol=1e-2, closed=False):
    """arPLS baseline with a per-point smoothness array lam_point (N,)."""
    P = np.asarray(contour, float)
    n = len(P)
    DtD = _DtD_var(np.asarray(lam_point, float), closed)
    w = np.ones(n)
    w_prev = w
    for _ in range(iters):
        S = np.stack([_whittaker(P[:, 0], w, 1.0, DtD),
                      _whittaker(P[:, 1], w, 1.0, DtD)], axis=1)
        dev = np.einsum("ij,ij->i", P - S, _normals(S, closed))
        w = _arpls_weights(dev)
        if np.linalg.norm(w - w_prev) / (np.linalg.norm(w_prev) + 1e-12) < tol:
            break
        w_prev = w
    S = np.stack([_whittaker(P[:, 0], w, 1.0, DtD),
                  _whittaker(P[:, 1], w, 1.0, DtD)], axis=1)
    dev = np.einsum("ij,ij->i", P - S, _normals(S, closed))
    return S, dev, w


def detect_tears(contour, lam_lo=1e4, lam_hi=1e6, k_mad=3.0, min_px=4.0,
                 wide_pts=25, grow=12, closed=False, edge_guard=8):
    """Two-scale tear detection with a single adaptive-lambda final baseline.

    1. Coarse pass at lam_hi -> bridges and measures WIDE tears (true depth).
    2. Mark wide-tear spans; build lambda(s) = lam_hi inside (grown by `grow`),
       lam_lo elsewhere.
    3. Final pass with smooth_var -> baseline bridges wide tears yet hugs the
       edge so narrow tears surface with a small, uninflated noise band.
    4. Threshold the adaptive deviation once.

    Returns dict with baseline S, deviation dev, tear mask, and per-tear stats.
    """
    P = np.asarray(contour, float)
    n = len(P)

    # 1. coarse, stiff pass
    Sc, devc, _ = smooth_intact_ear(P, lam=lam_hi, scheme="arpls",
                                    iters=30, closed=closed)
    core = devc[20:-20]
    band_c = 1.4826 * np.median(np.abs(core - np.median(core)))
    wide_mask = devc < -max(k_mad * band_c, min_px)

    # keep only spans wider than `wide_pts` -> the genuinely broad tears
    lam_point = np.full(n, lam_lo)
    for s, e in tear_segments(wide_mask if closed else wide_mask):
        if e < s:  # wrap (closed only)
            length = (n - s) + e + 1
        else:
            length = e - s + 1
        if length >= wide_pts:
            a, b = max(0, s - grow), min(n - 1, e + grow)
            lam_point[a:b + 1] = lam_hi

    # 2-3. single adaptive-lambda baseline
    S, dev, w = smooth_var(P, lam_point, iters=30, closed=closed)

    # 4. detect on the (now clean) adaptive deviation
    core = dev[edge_guard:-edge_guard] if not closed else dev
    band = 1.4826 * np.median(np.abs(core - np.median(core)))
    thr = -max(k_mad * band, min_px)
    is_tear = dev < thr
    if not closed:
        is_tear[:edge_guard] = is_tear[-edge_guard:] = False

    ds = np.linalg.norm(np.diff(P, axis=0), axis=1)
    ds = np.append(ds, ds[-1])
    stats = []
    for s, e in tear_segments(is_tear):
        idx = np.arange(s, e + 1) if e >= s else np.r_[s:n, 0:e + 1]
        stats.append(dict(start=int(s), end=int(e), n=len(idx),
                          depth=float(-dev[idx].min()),
                          area=float(np.sum(-dev[idx] * ds[idx])),
                          wide=bool(lam_point[idx].max() == lam_hi)))
    return dict(S=S, dev=dev, weight=w, is_tear=is_tear, thr=thr,
                band=band, lam_point=lam_point, stats=stats)



if __name__ == "__main__":
    import json
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P = np.array(json.load(open("contour.json")))
    N = len(P)

    res = {}
    for scheme in ("asls", "airpls", "arpls"):
        kw = dict(lam=1e5, scheme=scheme, iters=30, closed=False)
        if scheme == "asls": kw["p"] = 0.02
        res[scheme] = smooth_intact_ear(P, **kw)

    S, dev, w = res["arpls"]
    # robust noise band, exclude the open endpoints from the threshold estimate
    core = dev[20:-20]
    band = 1.4826 * np.median(np.abs(core - np.median(core)))
    thr = -max(3 * band, 4.0)
    is_tear = dev < thr
    is_tear[:8] = is_tear[-8:] = False     # ignore the two open ends
    segs = tear_segments(is_tear)
    ds = np.linalg.norm(np.diff(P, axis=0), axis=1); ds = np.append(ds, ds[-1])
    print(f"noise band {band:.2f}px  threshold {thr:.1f}px")
    for s, e in segs:
        idx = np.arange(s, e + 1)
        print(f"  tear {s:4d}-{e:4d}  n={len(idx):2d}  depth={-dev[idx].min():4.1f}px  area={np.sum(-dev[idx]*ds[idx]):5.0f}px^2")

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    axc = fig.add_subplot(gs[:, 0])
    axc.plot(*P.T, color="0.6", lw=1.0, label="raw margin")
    axc.plot(*S.T, "b-", lw=2.0, label="intact baseline (arPLS, open)")
    axc.plot(P[is_tear, 0], P[is_tear, 1], "m.", ms=5, label="detected tear")
    axc.plot(*P[[0, -1]].T, "kx", ms=9, label="open endpoints")
    axc.set_aspect("equal"); axc.invert_yaxis(); axc.legend(loc="lower left")
    axc.set_title("open ear margin: raw vs tear-free baseline")

    axd = fig.add_subplot(gs[0, 1])
    for sch, c in [("asls","tab:orange"),("airpls","tab:green"),("arpls","tab:blue")]:
        axd.plot(res[sch][1], c, lw=1.0, label=sch, alpha=0.85)
    axd.axhline(thr, color="r", ls="--", lw=0.8); axd.axhline(0, color="0.5", lw=0.6)
    axd.set_title("normal deviation (neg = inward = tear)"); axd.legend(fontsize=8)
    axd.set_xlabel("index"); axd.set_ylabel("px")

    axw = fig.add_subplot(gs[1, 1])
    axw.plot(w, "b-", lw=1.0); axw.fill_between(np.arange(N), w, 1, color="m", alpha=0.25)
    axw.set_title("arPLS final weights"); axw.set_ylim(-0.05,1.05)
    axw.set_xlabel("index"); axw.set_ylabel("weight")
    plt.tight_layout(); plt.savefig("ear_open.png", dpi=115)
    print("saved")
