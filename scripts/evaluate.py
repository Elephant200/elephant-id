"""Evaluation harness: does the 1-D tear profile carry individual identity?

TEST code for elephant_id.coding.ears.tear_profile - the matchers here are drafts used
to validate the feature, not part of the pipeline.

For every photo in the manifest: extract the contour, compute the
profile, reduce it to gated tear events. Score every pair two ways:

  * event  -- depth-weighted optimal assignment of tear events within a
              displacement window (the sparse matcher the exploration
              found best: noise has no events to align). Two empty event
              sets score 0: no tear evidence is no identity evidence.
  * corr   -- Pearson correlation of the profiles at the best
              stretch+shift alignment (dense baseline for comparison).

plus their z-fused combination. Reports AUC, top-1/3/5, MRR,
within/between means, misses, and per-photo tear evidence. All numbers are
pilot-set (17 photos / 8 individuals): hypothesis-generating, not
validated.

Outputs (outputs/evaluate/): metrics.txt, metrics.json, similarity_matrix.png.

Run:  uv run python scripts/evaluate.py
"""
import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from elephant_id.ai import AnchorService, Sam3Service
from elephant_id.coding.ears import AnchoredEar
from elephant_id.coding.ears.tear_profile import PROFILE_GRID, embed
from elephant_id.constants import (
    TEAR_PROFILE_BINS,
    TEAR_TRIM_HI,
    TEAR_TRIM_LO,
)
from elephant_id.dataset import Dataset
from elephant_id.log import configure_logging

REPO_ROOT = Path(__file__).resolve().parent.parent
PHOTOS: dict[str, str] = {
    label: rec["id"]
    for label, rec in json.loads(
        (REPO_ROOT / "data/notable_photos.json").read_text()).items()
}


def base_name(label: str) -> str:
    """Individual name from a photo label (strip the trailing index)."""
    return label.rstrip("0123456789")


def out_dir(script_name: str) -> Path:
    d = REPO_ROOT / "outputs" / script_name
    d.mkdir(parents=True, exist_ok=True)
    return d


class ContourExtractor:
    """Research path: photo -> cached ear contour, bypassing the coder.

    Mirrors PhotoAnalyzer's shared processing (SAM3 -> anchors -> AnchoredEar) so
    scripts can iterate without the full coding pipeline; contours cache as
    .npy under .cache/contours/ (delete to re-extract). Production goes
    SeekCoder -> PhotoAnalyzer -> EarFieldAnalyzer.
    """

    def __init__(
        self,
        dataset: Dataset,
        sam3: Sam3Service,
        anchor: AnchorService,
    ) -> None:
        self.dataset = dataset
        self.sam3 = sam3
        self.anchor = anchor

    def ears(self, identifier: str) -> list[AnchoredEar]:
        photo = self.dataset.get_photo(identifier)
        dets = self.sam3.run(photo, "features")
        ears: list[AnchoredEar] = []
        for d in (x for x in dets if x.class_name == "ear"):
            ad = self.anchor.run(photo, crop_xyxy=d.xyxy)
            if not ad:
                continue
            best = max(ad, key=lambda a: a.confidence)
            try:
                ears.append(AnchoredEar(d, best))
            except ValueError:
                continue
        return ears

    def contour(self, identifier: str, n_points: int = 1024) -> np.ndarray | None:
        """Resampled cut contour of the largest ear, or None."""
        ears = self.ears(identifier)
        if not ears:
            logger.warning(f"{identifier}: no anchored ears")
            return None
        ear = max(ears, key=lambda e: e.area)
        P = ear.resampled_contour(n_points)
        return P

    def crop(self, identifier: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Ear crop image and its (x0, y0) offset, for figure overlays."""
        ears = self.ears(identifier)
        if not ears:
            return None
        ear = max(ears, key=lambda e: e.area)
        from elephant_id.image.transforms import apply_crop
        photo = self.dataset.get_photo(identifier)
        img = apply_crop(self.dataset.read_image(photo), ear.xyxy)
        return img, np.array([ear.xyxy[0], ear.xyxy[1]])


def make_extractor(log_level: str = "WARNING") -> ContourExtractor:
    load_dotenv(REPO_ROOT / ".env")
    configure_logging(level=log_level)
    dataset = Dataset(
        dataset_root=REPO_ROOT / "dataset/elephants-alive/coded",
        metadata_path=REPO_ROOT / "dataset/elephants-alive/images.csv",
    )
    return ContourExtractor(dataset, Sam3Service(dataset=dataset),
                            AnchorService(dataset=dataset))

# Test-matcher constants, in profile units (depth / hull arc length S).
EVENT_GATE = 0.0024   # noise floor (= 0.005 of bbox long side); 0.003-0.005
                      # bbox-units all worked, 0.008 deleted real small tears
EVENT_XTOL = 0.08     # pairing window; p90 of measured event displacement
STRETCHES = (0.96, 0.98, 1.0, 1.02, 1.04)
SHIFTS = np.linspace(-0.085, 0.085, 35)
_LO = int(TEAR_TRIM_LO * TEAR_PROFILE_BINS)
_HI = int(TEAR_TRIM_HI * TEAR_PROFILE_BINS)


# ------------------------------ matchers ---------------------------------- #
def tear_events(profile: np.ndarray) -> list[tuple[float, float]]:
    """Profile -> (x, depth) peaks above the noise gate."""
    core = profile[_LO:-_HI].copy()
    core[core < EVENT_GATE] = 0.0
    pk, _ = find_peaks(core, height=EVENT_GATE, prominence=EVENT_GATE / 2)
    x = PROFILE_GRID[_LO:-_HI]
    return [(float(x[p]), float(core[p])) for p in pk]


def event_sim(e1: list[tuple[float, float]],
              e2: list[tuple[float, float]]) -> float:
    """Matched tear-depth mass / total mass, optimal one-to-one pairing."""
    t1, t2 = sum(e[1] for e in e1), sum(e[1] for e in e2)
    if t1 + t2 < 1e-9 or not e1 or not e2:
        return 0.0
    gain = np.zeros((len(e1), len(e2)))
    for i, (xa, da) in enumerate(e1):
        for j, (xb, db) in enumerate(e2):
            if abs(xa - xb) <= EVENT_XTOL:
                gain[i, j] = min(da, db)
    ri, ci = linear_sum_assignment(-gain)
    return 2 * float(gain[ri, ci].sum()) / (t1 + t2)


def _z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / (v.std() + 1e-12)


def profile_corr(d1: np.ndarray, d2: np.ndarray) -> float:
    """Correlation at the best stretch+shift alignment of the x-axis."""
    best = -np.inf
    z1 = _z(d1)
    for a in STRETCHES:
        for b in SHIFTS:
            w = np.interp(a * PROFILE_GRID + b, PROFILE_GRID, d2,
                          left=0.0, right=0.0)
            best = max(best, float(np.corrcoef(z1, _z(w))[0, 1]))
    return best


# ------------------------------ metrics ----------------------------------- #
def retrieval_metrics(sim: dict, labels: list[str]) -> dict:
    multi = [x for x in labels
             if sum(base_name(y) == base_name(x) for y in labels) > 1]
    ranks, misses = [], []
    for a in multi:
        ranked = sorted((x for x in labels if x != a),
                        key=lambda x: -sim[(a, x)])
        r = next(i for i, x in enumerate(ranked)
                 if base_name(x) == base_name(a)) + 1
        ranks.append(r)
        if r > 1:
            misses.append(f"{a}->{ranked[0]}")
    ranks = np.array(ranks)
    within = [sim[p] for p in itertools.combinations(labels, 2)
              if base_name(p[0]) == base_name(p[1])]
    between = [sim[p] for p in itertools.combinations(labels, 2)
               if base_name(p[0]) != base_name(p[1])]
    w, b = np.array(within), np.array(between)
    auc = float(((w[:, None] > b[None, :]).sum()
                 + 0.5 * (w[:, None] == b[None, :]).sum()) / (len(w) * len(b)))
    return {"n_queries": len(multi), "top1": int((ranks == 1).sum()),
            "top3": int((ranks <= 3).sum()), "top5": int((ranks <= 5).sum()),
            "mrr": float((1 / ranks).mean()), "auc": auc,
            "within_mean": float(w.mean()), "between_mean": float(b.mean()),
            "misses": misses}


# -------------------------------- main ------------------------------------ #
def main() -> None:
    extractor = make_extractor()
    out = out_dir("evaluate")

    profiles: dict[str, np.ndarray] = {}
    events: dict[str, list] = {}
    for label, ident in PHOTOS.items():
        P = extractor.contour(ident)
        if P is None:
            print(f"{label}: no contour, skipped")
            continue
        profiles[label] = embed(P)
        events[label] = tear_events(profiles[label])
        mass = sum(e[1] for e in events[label])
        print(f"{label:8s} {len(events[label])} tears, "
              f"evidence mass {100 * mass:.2f}% of S")
    labels = list(profiles)
    pairs = list(itertools.combinations(labels, 2))

    sims = {"event": {p: event_sim(events[p[0]], events[p[1]]) for p in pairs},
            "corr": {p: profile_corr(profiles[p[0]], profiles[p[1]])
                     for p in pairs}}
    combo = {}
    for name in ("event", "corr"):
        v = np.array([sims[name][p] for p in pairs])
        m, s = float(v.mean()), float(v.std()) + 1e-12
        for p in pairs:
            combo[p] = combo.get(p, 0.0) + (sims[name][p] - m) / s / 2
    sims["combo"] = combo
    for table in sims.values():
        table.update({(b, a): v for (a, b), v in list(table.items())})

    report: dict = {"n_photos": len(labels)}
    lines = [f"photos: {len(labels)}", ""]
    for name, sim in sims.items():
        m = retrieval_metrics(sim, labels)
        report[name] = m
        lines.append(
            f"{name:6s} AUC {m['auc']:.3f}  top1 {m['top1']}/{m['n_queries']} "
            f"top3 {m['top3']} top5 {m['top5']}  MRR {m['mrr']:.3f}  "
            f"within {m['within_mean']:.3f} between {m['between_mean']:.3f}")
        if m["misses"]:
            lines.append(f"       misses: {', '.join(m['misses'])}")
    lines += ["", "within-individual pairs (event / corr / combo):"]
    for a, b in pairs:
        if base_name(a) == base_name(b):
            lines.append(f"  {a}/{b}: {sims['event'][(a, b)]:5.3f}  "
                         f"{sims['corr'][(a, b)]:5.3f}  "
                         f"{sims['combo'][(a, b)]:6.2f}")

    text = "\n".join(lines)
    print(text)
    (out / "metrics.txt").write_text(text + "\n")
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    order = sorted(labels, key=lambda x: (base_name(x), x))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, name in zip(axes, sims, strict=True):
        M = np.array([[sims[name][(a, b)] if a != b else np.nan
                       for b in order] for a in order])
        im = ax.imshow(M, cmap="viridis")
        ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
        ax.set_yticks(range(len(order)), order, fontsize=7)
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("pairwise similarities (diagonal masked)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "similarity_matrix.png", dpi=110)
    print(f"saved {out}/metrics.txt, metrics.json, similarity_matrix.png")


if __name__ == "__main__":
    main()
