"""EXTREMELY SLOW experimentation harness for detecting holes in elephant ears.

This is a *standalone experimentation script* (separate from ``holes.py``). It
loads the example ears, runs several candidate hole-detection methods, and writes
annotated outputs + zoomed crops to ``scripts/hole_experiments_out/`` so results
can be reviewed as files (instead of interactive ``cv2.imshow`` windows).

Usage:
    python scripts/hole_experiments.py dump            # dump ear crops to study
    python scripts/hole_experiments.py run [method...] # run detection methods
    python scripts/hole_experiments.py list            # list available methods

Each detection method is a function ``method(ctx) -> list[np.ndarray]`` returning
OpenCV contours (in ear-crop pixel coordinates). All methods are kept in place
for side-by-side comparison.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from elephant_id.ai import AnchorService, Sam3Service
from elephant_id.coding.ears import AnchoredEar
from elephant_id.dataset import Dataset
from elephant_id.image.transforms import apply_crop, apply_mask

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_ROOT = Path("outputs/holes")

# ROI geometry constants (ported from holes.py).
EDGE_MARGIN_FRACTION = 0.015
ANCHOR_LINE_MARGIN_FRACTION = 0.2

# The nine example photos with holes worth studying.
PHOTOS = {
    "Gap": "Gap_2019-11-20_02",
    "Centaures": "Centaures_2018-11-24_10",
    "Bloom": "Bloom_2016-06-06_08",
    "Intwandamela": "Intwandamela_2021-05-27_03",
    "Intwandamela2": "Intwandamela_2019-05-21_13",
    "Nguyen": "Nguyen_2012-08-02_07",
    "Scar": "Scar_2010-11-30_08",
    "Delani": "Delani_2017-10-09_06",
    "Fire": "Fire_2004-01-10_03",
}


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------


@dataclass
class EarContext:
    """Everything a detection method needs for one ear."""

    name: str
    ear: AnchoredEar
    image: np.ndarray  # BGR ear crop (full crop, background still present)
    ear_mask: np.ndarray  # bool, ear interior within the crop
    ear_only: np.ndarray  # BGR, background blacked out
    gray: np.ndarray  # uint8 LAB-L channel within the crop
    roi_mask: np.ndarray  # bool, plausible-hole interior (edges/crease excluded)
    # debug exclusion bands
    edge_exclusion: np.ndarray = field(repr=False, default=None)
    anchor_line_exclusion: np.ndarray = field(repr=False, default=None)
    top_line_exclusion: np.ndarray = field(repr=False, default=None)
    # Sobel gradient magnitude of gray (filled in by run_methods).
    grad: np.ndarray = field(repr=False, default=None)

    @property
    def area(self) -> float:
        return self.ear.area


def trim_binary_mask(mask: np.ndarray, *, margin_fraction: float) -> np.ndarray:
    """Return a binary mask with its nearest boundary band removed."""
    bool_mask = mask.astype(bool)
    if not bool_mask.any():
        raise ValueError("Cannot trim an empty mask")
    distances = cv2.distanceTransform(bool_mask.astype(np.uint8), cv2.DIST_L2, maskSize=5)
    edge_margin = margin_fraction * max(mask.shape)
    return distances > edge_margin


def build_roi_mask(
    ear: AnchoredEar,
    ear_mask: np.ndarray,
    crop_origin: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the plausible-hole interior mask (ported from holes.py).

    Excludes a band near the ear edge, a band around the anchor line, and the
    region above the top fold line. Returns (interior, edge_excl, anchor_excl,
    top_excl), all bool in crop coordinates.
    """
    interior_mask = trim_binary_mask(ear_mask, margin_fraction=EDGE_MARGIN_FRACTION)
    edge_exclusion = ear_mask & ~interior_mask

    (x1, y1), (x2, y2) = ear.anchor_points
    crop_x1, crop_y1 = crop_origin
    x1, x2 = x1 - crop_x1, x2 - crop_x1
    y1, y2 = y1 - crop_y1, y2 - crop_y1

    yy, xx = np.indices(ear_mask.shape)
    anchor_line_margin = ANCHOR_LINE_MARGIN_FRACTION * max(ear_mask.shape)

    anchor_line_mask = np.zeros(ear_mask.shape, dtype=np.uint8)
    cv2.line(anchor_line_mask, (round(x1), round(y1)), (round(x2), round(y2)), color=1, thickness=1)
    anchor_distances = cv2.distanceTransform(1 - anchor_line_mask, cv2.DIST_L2, maskSize=5)
    anchor_line_exclusion = anchor_distances <= anchor_line_margin
    interior_mask &= ~anchor_line_exclusion

    upper_x, upper_y = min(((x1, y1), (x2, y2)), key=lambda p: p[1])
    ear_ys, ear_xs = np.where(ear_mask)
    middle_x = round((ear_xs.min() + ear_xs.max()) / 2)
    top_x = ear_xs[np.argmin(np.abs(ear_xs - middle_x))]
    top_y = ear_ys[ear_xs == top_x].min()
    if abs(top_x - upper_x) < 1e-6:
        top_line_exclusion = yy <= min(top_y, upper_y) + anchor_line_margin
    else:
        line_y = upper_y + ((top_y - upper_y) / (top_x - upper_x)) * (xx - upper_x)
        line_distance = np.abs(
            (top_y - upper_y) * xx
            - (top_x - upper_x) * yy
            + top_x * upper_y
            - top_y * upper_x
        ) / np.hypot(top_y - upper_y, top_x - upper_x)
        top_line_exclusion = (yy < line_y) | (line_distance <= anchor_line_margin)
    interior_mask &= ~top_line_exclusion

    return interior_mask, edge_exclusion, anchor_line_exclusion, top_line_exclusion


def _build_ear_context(name: str, ear: AnchoredEar, full_image: np.ndarray) -> EarContext:
    """Build one EarContext from an anchored ear and the full photo image."""
    crop_origin = (int(ear.xyxy[0]), int(ear.xyxy[1]))
    image = apply_crop(full_image, ear.xyxy)
    ear_mask = ear.mask[
        int(ear.xyxy[1]):int(ear.xyxy[3]), int(ear.xyxy[0]):int(ear.xyxy[2])
    ].copy()
    ear_only = apply_mask(image, ear_mask)
    gray = cv2.split(cv2.cvtColor(ear_only, cv2.COLOR_BGR2LAB))[0]
    roi, edge_excl, anchor_excl, top_excl = build_roi_mask(ear, ear_mask, crop_origin)
    return EarContext(
        name=name,
        ear=ear,
        image=image,
        ear_mask=ear_mask,
        ear_only=ear_only,
        gray=gray,
        roi_mask=roi,
        edge_exclusion=edge_excl,
        anchor_line_exclusion=anchor_excl,
        top_line_exclusion=top_excl,
    )


def build_ears() -> dict[str, EarContext]:
    """Load all example photos and build an EarContext per anchored ear.

    Every anchored ear is kept (labeled ``Name#i`` by descending area), not just
    the largest — some photos (e.g. Centaures) carry the hole on the smaller ear.
    """
    dataset = Dataset(
        dataset_root=Path("dataset/elephants-alive/coded"),
        metadata_path=Path("dataset/elephants-alive/images.csv"),
    )
    sam3 = Sam3Service(dataset=dataset)
    anchor_model = AnchorService(dataset=dataset)

    contexts: dict[str, EarContext] = {}
    for name, identifier in PHOTOS.items():
        photo = dataset.get_photo(identifier)
        detections = sam3.run(photo, "features")
        ear_detections = [d for d in detections if d.class_name == "ear"]

        ears: list[AnchoredEar] = []
        for ear_detection in ear_detections:
            anchor_dets = anchor_model.run(photo, crop_xyxy=ear_detection.xyxy)
            if len(anchor_dets) == 0:
                continue
            anchor_dets = sorted(anchor_dets, key=lambda d: d.confidence, reverse=True)
            ears.append(AnchoredEar(ear_detection, anchor_dets[0]))
        if not ears:
            print(f"[{name}] no anchored ears found; skipping")
            continue

        full_image = dataset.read_image(photo)
        ears.sort(key=lambda e: e.area, reverse=True)
        for i, ear in enumerate(ears):
            label = name if len(ears) == 1 else f"{name}#{i}"
            contexts[label] = _build_ear_context(label, ear, full_image)
            print(f"[{label}] side={ear.side} area={ear.area:.0f} "
                  f"crop={contexts[label].image.shape[1]}x{contexts[label].image.shape[0]}")
    return contexts


# ---------------------------------------------------------------------------
# Dump mode: study what holes look like before tuning detectors
# ---------------------------------------------------------------------------


def _upscale(img: np.ndarray, target_long_edge: int = 900) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_long_edge / max(h, w)
    if scale <= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_NEAREST)


def background_subtract(ctx: EarContext, sigma_frac: float = 0.03) -> np.ndarray:
    """Signed local-mean subtraction: gray - local_mean, float32, ear interior.

    Positive => brighter than surroundings, negative => darker. The Gaussian
    sigma scales with ear size so the background estimate is roughly hole-scale
    independent across the very different crop resolutions.
    """
    sigma = max(5.0, sigma_frac * np.sqrt(ctx.area))
    gray_f = ctx.gray.astype(np.float32)
    mask_f = ctx.ear_mask.astype(np.float32)
    # Normalized box/Gaussian so the mean ignores the black background.
    blur = cv2.GaussianBlur(gray_f * mask_f, (0, 0), sigma)
    weight = cv2.GaussianBlur(mask_f, (0, 0), sigma)
    local_mean = blur / np.clip(weight, 1e-3, None)
    signed = (gray_f - local_mean)
    signed[~ctx.ear_mask] = 0.0
    return signed


def _diverging(signed: np.ndarray, scale: float = 25.0) -> np.ndarray:
    """Map signed deviation to BGR: red=darker-than-local, blue=brighter."""
    out = np.zeros((*signed.shape, 3), dtype=np.uint8)
    darker = np.clip(-signed / scale, 0, 1)  # holes that are darker
    brighter = np.clip(signed / scale, 0, 1)  # holes that are brighter
    out[..., 2] = (darker * 255).astype(np.uint8)  # R
    out[..., 0] = (brighter * 255).astype(np.uint8)  # B
    return out


def dump_ears(contexts: dict[str, EarContext]) -> None:
    """Save per-ear crops (ear-only, grayscale, ROI overlay, deviation) for study."""
    out = OUT_ROOT / "_crops"
    out.mkdir(parents=True, exist_ok=True)
    for name, ctx in contexts.items():
        roi_overlay = ctx.ear_only.copy()
        roi_overlay[ctx.edge_exclusion] = (0, 0, 255)
        roi_overlay[ctx.anchor_line_exclusion & ctx.ear_mask] = (255, 0, 0)
        roi_overlay[ctx.top_line_exclusion & ctx.ear_mask] = (0, 255, 0)

        gray_bgr = cv2.cvtColor(ctx.gray, cv2.COLOR_GRAY2BGR)

        signed = background_subtract(ctx)
        signed[~ctx.roi_mask] = 0.0  # only show deviations inside the ROI
        dev = _diverging(signed)

        cv2.imwrite(str(out / f"{name}_1_ear.png"), _upscale(ctx.ear_only))
        cv2.imwrite(str(out / f"{name}_2_gray.png"), _upscale(gray_bgr))
        cv2.imwrite(str(out / f"{name}_3_roi.png"), _upscale(roi_overlay))
        cv2.imwrite(str(out / f"{name}_4_dev.png"), _upscale(dev))
    print(f"Wrote crops to {out}")


# ---------------------------------------------------------------------------
# Contour metrics & shared acceptance filter
#
# The study of the example ears showed the key discriminator: holes are compact,
# roughly elliptical blobs, while the dominant confounder (wrinkles/folds) are
# elongated ridges. Tone contrast alone fails on Bloom (texture-only holes), so
# acceptance also allows a closed-rim cue (boundary gradient) or an interior
# texture cue. Methods are compared through this single filter.
# ---------------------------------------------------------------------------

# Acceptance thresholds (tunable). Areas are fractions of the ear area.
FILTER = {
    "area_min_frac": 8e-5,
    "area_max_frac": 6e-3,
    "circularity_min": 0.35,
    "solidity_min": 0.80,
    "elongation_max": 3.2,
    "tone_contrast_min": 6.0,  # |median(core) - median(ring)| in L (0..255)
    "edge_strength_min": 22.0,  # mean Sobel magnitude on the contour band
    "texture_contrast_min": 9.0,  # std(core) - std(ring) in L
}


@dataclass
class ContourMetrics:
    area_frac: float
    circularity: float
    solidity: float
    elongation: float
    tone_contrast: float
    edge_strength: float
    texture_contrast: float
    centroid: tuple[int, int]


def _filled(shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.drawContours(m, [contour], -1, 1, cv2.FILLED)
    return m.astype(bool)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def measure_contour(ctx: EarContext, contour: np.ndarray) -> ContourMetrics | None:
    """Compute shape + contrast/texture/edge metrics for a candidate contour."""
    area = cv2.contourArea(contour)
    if area < 4:
        return None
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    circularity = float(np.clip(4.0 * np.pi * area / (perimeter * perimeter), 0, 1))

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area / hull_area) if hull_area > 0 else 0.0

    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    elongation = float(max(w, h) / max(min(w, h), 1e-3))

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    mask = _filled(ctx.gray.shape, contour)
    radius = max(2.0, np.sqrt(area / np.pi))
    core_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(radius * 0.7)) | 1,) * 2)
    core = cv2.erode(mask.astype(np.uint8), core_k).astype(bool)
    if not core.any():
        core = mask
    inner_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(radius * 1.4)) | 1,) * 2)
    outer_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(radius * 3.0)) | 1,) * 2)
    ring = (
        cv2.dilate(mask.astype(np.uint8), outer_k).astype(bool)
        & ~cv2.dilate(mask.astype(np.uint8), inner_k).astype(bool)
        & ctx.ear_mask
    )
    if not ring.any():
        return None

    gray_f = ctx.gray.astype(np.float32)
    tone_contrast = float(abs(np.median(gray_f[core]) - np.median(gray_f[ring])))
    texture_contrast = float(np.std(gray_f[core]) - np.std(gray_f[ring]))

    band = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~core
    band &= ctx.ear_mask
    edge_strength = float(np.mean(ctx.grad[band])) if band.any() else 0.0

    return ContourMetrics(
        area_frac=area / ctx.area,
        circularity=circularity,
        solidity=solidity,
        elongation=elongation,
        tone_contrast=tone_contrast,
        edge_strength=edge_strength,
        texture_contrast=texture_contrast,
        centroid=(cx, cy),
    )


def classify(ctx: EarContext, contour: np.ndarray) -> tuple[str, ContourMetrics | None]:
    """Return ('accepted' | rejection-reason, metrics)."""
    m = measure_contour(ctx, contour)
    if m is None:
        return "degenerate", None
    cx, cy = m.centroid
    if not (0 <= cy < ctx.roi_mask.shape[0] and 0 <= cx < ctx.roi_mask.shape[1] and ctx.roi_mask[cy, cx]):
        return "outside_roi", m
    if not (FILTER["area_min_frac"] <= m.area_frac <= FILTER["area_max_frac"]):
        return "area", m
    if m.elongation > FILTER["elongation_max"]:
        return "elongated", m
    if m.solidity < FILTER["solidity_min"]:
        return "ragged", m
    if m.circularity < FILTER["circularity_min"]:
        return "noncircular", m
    cue = (
        m.tone_contrast >= FILTER["tone_contrast_min"]
        or m.edge_strength >= FILTER["edge_strength_min"]
        or m.texture_contrast >= FILTER["texture_contrast_min"]
    )
    if not cue:
        return "no_contrast", m
    return "accepted", m


def dedupe_contours(contours: list[np.ndarray], min_dist: float = 8.0) -> list[np.ndarray]:
    """Drop near-duplicate contours by centroid proximity, keeping larger ones."""
    scored = []
    for c in contours:
        mom = cv2.moments(c)
        if mom["m00"] == 0:
            continue
        cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
        scored.append((cv2.contourArea(c), cx, cy, c))
    scored.sort(key=lambda t: -t[0])
    kept: list[tuple[float, float, np.ndarray]] = []
    for _area, cx, cy, c in scored:
        if any(np.hypot(cx - kx, cy - ky) < min_dist for kx, ky, _ in kept):
            continue
        kept.append((cx, cy, c))
    return [c for _, _, c in kept]


# ---------------------------------------------------------------------------
# Candidate-center -> contour refinement (for blob/LoG center detections)
# ---------------------------------------------------------------------------


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes of a binary component (texture gaps)."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask, np.uint8)
    cv2.drawContours(out, cs, -1, 1, cv2.FILLED)
    return out


def refine_to_contour(
    ctx: EarContext, cx: int, cy: int, radius: float
) -> np.ndarray | None:
    """Segment a hole contour around a detected blob center via local Otsu.

    A window around the center is dominated by surrounding tissue plus the
    (darker or brighter) hole, so a local Otsu split adapts per-hole. The chosen
    component is closed at hole scale and its interior texture filled, yielding a
    solid contour even for textured holes (Bloom). Returns ear-crop coords.
    """
    h, w = ctx.gray.shape
    r = max(3, round(radius))
    pad = int(r * 3) + 6
    x0, y0 = max(0, cx - pad), max(0, cy - pad)
    x1, y1 = min(w, cx + pad), min(h, cy + pad)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None

    patch = ctx.gray[y0:y1, x0:x1].astype(np.float32)
    emask = ctx.ear_mask[y0:y1, x0:x1]
    if emask.sum() < 9:
        return None
    lcx, lcy = cx - x0, cy - y0
    yy, xx = np.indices(patch.shape)
    dist = np.hypot(xx - lcx, yy - lcy)
    inner = (dist < r) & emask
    if not inner.any():
        return None

    # Contrast-stretch the ear pixels, then Otsu within the window.
    ear_vals = patch[emask]
    lo, hi = np.percentile(ear_vals, [2, 98])
    if hi - lo < 1e-3:
        return None
    norm = np.clip((patch - lo) / (hi - lo), 0, 1)
    norm_filled = norm.copy()
    norm_filled[~emask] = np.median(norm[emask])  # neutralise background
    v8 = (norm_filled * 255).astype(np.uint8)
    otsu_t, _ = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    polarity_dark = float(np.median(patch[inner])) < float(np.median(ear_vals))
    binar = (v8 <= otsu_t) & emask if polarity_dark else (v8 > otsu_t) & emask
    binar = binar.astype(np.uint8)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(r * 0.7)) | 1,) * 2)
    binar = cv2.morphologyEx(binar, cv2.MORPH_CLOSE, close_k)
    binar = _fill_holes(binar)

    if binar[lcy, lcx] == 0:
        ys, xs = np.where(binar & (dist < r * 1.6))
        if len(xs) == 0:
            return None
        idx = int(np.argmin((xs - lcx) ** 2 + (ys - lcy) ** 2))
        lcx, lcy = int(xs[idx]), int(ys[idx])

    _n, labels = cv2.connectedComponents(binar)
    label = labels[lcy, lcx]
    if label == 0:
        return None
    comp = (labels == label).astype(np.uint8)
    # Reject components that bled into the surrounding tissue (fill the window).
    if comp.sum() > 0.6 * emask.sum():
        return None
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return contour + np.array([[x0, y0]], dtype=np.int32)


# ---------------------------------------------------------------------------
# Detection methods. Each returns raw candidate contours (ear-crop coords);
# the shared classify() then accepts/rejects them.
# ---------------------------------------------------------------------------


def _normalized(ctx: EarContext, sigma_frac: float = 0.03) -> np.ndarray:
    """Background-subtracted absolute deviation, uint8, ROI only."""
    signed = background_subtract(ctx, sigma_frac=sigma_frac)
    norm = np.clip(np.abs(signed), 0, 60) / 60.0 * 255.0
    return norm.astype(np.uint8)


def method_canny(ctx: EarContext) -> list[np.ndarray]:
    """Baseline: Canny edges on bg-subtracted L -> close/open -> contours."""
    norm = _normalized(ctx)
    edges = cv2.Canny(norm, 50, 125)
    edges[~ctx.roi_mask] = 0
    ck = round(np.sqrt(ctx.area) * 0.0025) * 2 + 1
    ok = round(np.sqrt(ctx.area) * 0.00125) * 2 + 1
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)), iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok)), iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def method_mser(ctx: EarContext) -> list[np.ndarray]:
    """MSER on both polarities of the bg-subtracted image."""
    norm = _normalized(ctx)
    norm[~ctx.roi_mask] = 0
    area = ctx.area
    mser = cv2.MSER_create()
    mser.setMinArea(int(FILTER["area_min_frac"] * area))
    mser.setMaxArea(int(FILTER["area_max_frac"] * area))
    mser.setDelta(5)
    out: list[np.ndarray] = []
    for img in (norm, cv2.bitwise_not(norm)):
        regions, _ = mser.detectRegions(img)
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            out.append(hull)
    return dedupe_contours(out)


def method_threshold_ccl(ctx: EarContext) -> list[np.ndarray]:
    """Adaptive + Otsu threshold on signed bg-subtraction, both polarities, CCL."""
    signed = background_subtract(ctx)
    out: list[np.ndarray] = []
    for pol in (-1, 1):  # -1 dark holes, +1 bright holes
        resp = np.clip(pol * signed, 0, None)
        resp = (resp / max(resp.max(), 1e-3) * 255).astype(np.uint8)
        resp[~ctx.roi_mask] = 0
        _, otsu = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, _stats, _ = cv2.connectedComponentsWithStats(otsu)
        for i in range(1, n):
            comp = (labels == i).astype(np.uint8)
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out.extend(cs)
    return dedupe_contours(out)


def method_tophat(ctx: EarContext) -> list[np.ndarray]:
    """Morphological black-hat (dark holes) + top-hat (bright holes) at hole scale."""
    radius = int(np.sqrt(FILTER["area_max_frac"] * ctx.area / np.pi))
    ksize = max(7, radius * 2 + 1) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    gray = ctx.gray
    out: list[np.ndarray] = []
    for op in (cv2.MORPH_BLACKHAT, cv2.MORPH_TOPHAT):
        resp = cv2.morphologyEx(gray, op, kernel)
        resp[~ctx.roi_mask] = 0
        _, binar = cv2.threshold(resp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binar = cv2.morphologyEx(binar, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cs, _ = cv2.findContours(binar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out.extend(cs)
    return dedupe_contours(out)


def method_blob(ctx: EarContext) -> list[np.ndarray]:
    """cv2.SimpleBlobDetector on bg-subtracted image, both polarities, refined."""
    area = ctx.area
    out: list[np.ndarray] = []
    for blob_color in (0, 255):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = blob_color
        params.filterByArea = True
        params.minArea = float(FILTER["area_min_frac"] * area)
        params.maxArea = float(FILTER["area_max_frac"] * area)
        params.filterByCircularity = True
        params.minCircularity = 0.4
        params.filterByInertia = True
        params.minInertiaRatio = 0.25  # rejects elongated wrinkle responses
        params.filterByConvexity = True
        params.minConvexity = 0.7
        params.minThreshold = 10
        params.maxThreshold = 220
        params.thresholdStep = 10
        detector = cv2.SimpleBlobDetector_create(params)
        norm = _normalized(ctx)
        keypoints = detector.detect(norm if blob_color == 255 else cv2.bitwise_not(norm))
        for kp in keypoints:
            cx, cy = round(kp.pt[0]), round(kp.pt[1])
            contour = refine_to_contour(ctx, cx, cy, kp.size / 2.0)
            if contour is not None:
                out.append(contour)
    return dedupe_contours(out)


def method_log_doh(ctx: EarContext) -> list[np.ndarray]:
    """Multi-scale Laplacian-of-Gaussian blobs with Hessian ridge rejection.

    Targets compact blobs and rejects elongated ridges (wrinkles) via the
    Hessian eigenvalue ratio, then refines each center to a contour.
    """
    gray = ctx.gray.astype(np.float32)
    r_min = max(2.0, np.sqrt(FILTER["area_min_frac"] * ctx.area / np.pi))
    r_max = np.sqrt(FILTER["area_max_frac"] * ctx.area / np.pi)
    sigmas = np.geomspace(r_min / np.sqrt(2), r_max / np.sqrt(2), 6)

    responses = []
    for s in sigmas:
        g = cv2.GaussianBlur(gray, (0, 0), s)
        lap = cv2.Laplacian(g, cv2.CV_32F) * (s ** 2)
        responses.append(lap)
    stack = np.stack(responses)  # (S, H, W), sign encodes polarity
    abs_stack = np.abs(stack)
    best_scale = np.argmax(abs_stack, axis=0)
    best_resp = np.max(abs_stack, axis=0)

    # Local maxima within the ROI. A low absolute floor lets subtle holes
    # through; the Hessian ridge test and shape/contrast filter remove the
    # wrinkle/noise peaks this admits.
    best_resp[~ctx.roi_mask] = 0
    thresh = max(2.0, 0.08 * best_resp.max())
    local_max = cv2.dilate(best_resp, np.ones((5, 5), np.float32))
    peaks = (best_resp >= local_max) & (best_resp >= thresh)
    ys, xs = np.where(peaks)

    out: list[np.ndarray] = []
    for y, x in zip(ys, xs, strict=True):
        s = sigmas[best_scale[y, x]]
        g = cv2.GaussianBlur(gray, (0, 0), s)
        # Hessian at this scale (scale-normalized) for ridge rejection.
        gxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        gyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        gxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        a, b, c = gxx[y, x], gyy[y, x], gxy[y, x]
        tmp = np.sqrt(max(((a - b) / 2.0) ** 2 + c ** 2, 0.0))
        l1, l2 = (a + b) / 2.0 + tmp, (a + b) / 2.0 - tmp
        if min(abs(l1), abs(l2)) < 1e-6:
            continue
        ratio = max(abs(l1), abs(l2)) / min(abs(l1), abs(l2))
        if ratio > 4.0:  # ridge-like -> reject
            continue
        contour = refine_to_contour(ctx, int(x), int(y), s * np.sqrt(2))
        if contour is not None:
            out.append(contour)
    return dedupe_contours(out)


def _sam3_hole(ctx: EarContext, confidence: float, query: str = "hole") -> list[np.ndarray]:
    """Run the SAM3 text prompt via run_workflow directly (no caching)."""
    import os

    from inference_sdk import InferenceHTTPClient

    from elephant_id.ai.sam3 import detection_from_prediction
    from elephant_id.constants import (
        ROBOFLOW_API_URL,
        ROBOFLOW_SAM3_WORKFLOW_ID,
        ROBOFLOW_WORKSPACE,
    )

    client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=os.getenv("ROBOFLOW_API_KEY"))
    response = client.run_workflow(
        workspace_name=ROBOFLOW_WORKSPACE,
        workflow_id=ROBOFLOW_SAM3_WORKFLOW_ID,
        images={"image": ctx.ear_only},
        parameters={"queries": query, "confidence_threshold": confidence,
                    "nms": True, "nms_iou_threshold": 0.2},
    )
    try:
        preds = response[0]["predictions"]["predictions"]
    except (KeyError, IndexError, TypeError):
        return []
    out: list[np.ndarray] = []
    for pred in preds:
        det = detection_from_prediction(pred)
        if det.rle_mask is None:
            continue
        mask = det.get_mask().astype(np.uint8)
        if mask.shape != ctx.gray.shape:
            mask = cv2.resize(mask, (ctx.gray.shape[1], ctx.gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out.extend(cs)
    return out


def method_sam3_hole(ctx: EarContext) -> list[np.ndarray]:
    """SAM3 'hole' prompt at the default confidence (high precision)."""
    return _sam3_hole(ctx, confidence=0.2)


def method_sam3_hole_lo(ctx: EarContext) -> list[np.ndarray]:
    """SAM3 'hole' prompt at low confidence (recover small/subtle holes)."""
    return _sam3_hole(ctx, confidence=0.03)


METHODS = {
    "canny": method_canny,
    "mser": method_mser,
    "threshold_ccl": method_threshold_ccl,
    "tophat": method_tophat,
    "blob": method_blob,
    "log_doh": method_log_doh,
    "sam3_hole": method_sam3_hole,
    "sam3_hole_lo": method_sam3_hole_lo,
}

LEARNED_METHODS = {"sam3_hole", "sam3_hole_lo"}

REJECTION_COLORS = {
    "area": (0, 0, 255),
    "elongated": (0, 255, 255),
    "ragged": (255, 0, 255),
    "noncircular": (255, 255, 0),
    "no_contrast": (255, 0, 0),
    "outside_roi": (128, 128, 128),
    "degenerate": (60, 60, 60),
}


def run_methods(contexts: dict[str, EarContext], method_names: list[str]) -> None:
    """Run methods on every ear, save annotated ears + per-hole zoom crops."""
    for ctx in contexts.values():
        ctx.grad = _gradient_magnitude(ctx.gray)

    for method_name in method_names:
        method = METHODS[method_name]
        out_dir = OUT_ROOT / method_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== method: {method_name} ===")
        learned = method_name in LEARNED_METHODS
        for name, ctx in contexts.items():
            try:
                raw = method(ctx)
            except Exception as exc:  # keep the sweep going
                print(f"  [{name}] ERROR: {exc}")
                continue

            # Learned detectors (SAM3) are shown RAW: their own confidence is the
            # filter. Applying classical shape rules here destroys real tears
            # (elongated) and large holes (area) -- so only classical methods are
            # passed through classify().
            accepted: list[np.ndarray] = []
            rejected: dict[str, list[np.ndarray]] = {}
            if learned:
                accepted = list(raw)
            else:
                for contour in raw:
                    verdict, _ = classify(ctx, contour)
                    if verdict == "accepted":
                        accepted.append(contour)
                    else:
                        rejected.setdefault(verdict, []).append(contour)

            annot = ctx.ear_only.copy()
            for reason, cs in rejected.items():
                cv2.drawContours(annot, cs, -1, REJECTION_COLORS.get(reason, (90, 90, 90)), 1)
            cv2.drawContours(annot, accepted, -1, (0, 255, 0), 2)
            cv2.imwrite(str(out_dir / f"{name}_annot.png"), _upscale(annot))

            # Zoomed full-resolution crops around each detection (geometry derived
            # straight from the contour so it works for filtered + raw output).
            for idx, contour in enumerate(accepted):
                mom = cv2.moments(contour)
                if mom["m00"] == 0:
                    continue
                cx, cy = int(mom["m10"] / mom["m00"]), int(mom["m01"] / mom["m00"])
                r = int(np.sqrt(cv2.contourArea(contour) / np.pi))
                pad = max(20, r * 4)
                x0, y0 = max(0, cx - pad), max(0, cy - pad)
                x1, y1 = min(ctx.gray.shape[1], cx + pad), min(ctx.gray.shape[0], cy + pad)
                crop = ctx.ear_only[y0:y1, x0:x1].copy()
                shifted = contour - np.array([[x0, y0]], dtype=np.int32)
                cv2.drawContours(crop, [shifted], -1, (0, 255, 0), 1)
                cv2.imwrite(str(out_dir / f"{name}_hole{idx}.png"), _upscale(crop, 360))

            rej_counts = {k: len(v) for k, v in rejected.items()}
            tag = "raw" if learned else "rejected"
            print(f"  [{name}] detections={len(accepted)} {tag}={rej_counts}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> None:
    from dotenv import load_dotenv

    from elephant_id.log import configure_logging

    load_dotenv()
    configure_logging()

    mode = argv[0] if argv else "dump"
    if mode == "list":
        print("Methods:", ", ".join(sorted(METHODS)) or "(none yet)")
        return

    contexts = build_ears()

    if mode == "dump":
        dump_ears(contexts)
    elif mode == "run":
        requested = argv[1:] or [m for m in METHODS if m not in ("sam3_hole", "sam3_hole_lo")]
        unknown = [m for m in requested if m not in METHODS]
        if unknown:
            raise SystemExit(f"Unknown method(s): {unknown}. Known: {sorted(METHODS)}")
        run_methods(contexts, requested)
    else:
        raise SystemExit(f"Unknown mode: {mode!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
