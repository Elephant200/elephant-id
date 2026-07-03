# Tear Profile Reference

This document preserves the substantive design history behind the current tear-profile representation. It merges the original tear embedding note and the later angular-coordinate update.

## Goal

Elephants are individually identifiable by tears along their ear margins. The goal of the tear-profile stage is to reduce a photographed ear margin to a one-dimensional signal where position is stable across photos and positive values correspond to inward tears.

Two photographs of the same ear, even years apart and from different angles, should produce profiles whose tear peaks approximately coincide.

## Earlier Reference-Path Design

The first settled pipeline used:

1. a light morphological opening of the margin,
2. an alpha-hull estimate of the intact margin,
3. a depth scan along the reference path's inward normals,
4. normalization.

The design was chosen after rejecting several alternatives.

## Rejected Smoothing Baselines

The first approach estimated the intact margin with asymmetric penalized-least-squares smoothing, treating tears as outliers below a smooth baseline.

This failed because smoothing separates signal from background by wavelength, but tears vary widely in size. A baseline stiff enough to bridge a large tear misses small nicks; a baseline soft enough to expose small nicks sags into large tears and truncates their measured depth.

An adaptive variant was also tried. It detected wide tears in a stiff first pass and locally re-weighted the smoother, but it became complicated, introduced shoulder artifacts, and still did not measure large and small tears consistently.

## Rejected Convex Hull Reference

The convex hull carried some useful signal but bridged normal anatomical concavities as if they were tears. It also reacted strongly to single outward segmentation errors, shifting reference edges far from the actual error.

The key distinction is geometric: tears are narrow-mouthed relative to their depth, while normal anatomical bays are wide-mouthed and shallow.

## Alpha-Hull Reference

A disk rolled along the outside of the margin captures that distinction. It cannot enter concavities whose mouths are narrow relative to the disk, so tears are bridged, while wide shallow bays are followed.

The original raster morphological closing produced the intended reference but was too slow at full image resolution. The alpha hull is the faster vector-geometry equivalent.

The alpha radius is an important parameter. Early sweeps found that roughly 20-40% of ear scale gave stable qualitative results under the original scale definition.

## Coordinate-System Attempts

Three coordinate systems were tried.

**Anchor-midpoint pole**:
Place a polar pole at the midpoint between anchors and parameterize by angle. This failed because the margin is not always star-convex; a single ray can cross a torn margin multiple times.

**Centroid pole**:
Moving the pole to the reference centroid improved the reference but retained star-convexity problems. Oblique rays also aliased thin flaps and steep tear walls.

**Reference normal scan**:
Sample the reference path at equal arc-length positions and cast rays along inward normals. This avoids a star-convexity assumption and measures the tear silhouette from the bridged intact-margin estimate.

The normal scan must use the nearest crossing along the ray line rather than the first crossing in front of the origin, because reference points do not always lie exactly on original margin points after opening.

## Opening for Segmentation Noise

Envelope-based references are vulnerable to outward segmentation excursions. A small spurious outward bump can lift the reference and corrupt depth readings away from the error.

A light morphological opening removes outward protrusions narrower than its disk before building the reference, while inward tears remain available for measurement against the original margin.

The opening should stay light. More aggressive opening can erase fine outward texture that may carry identity signal on nearly intact ears.

## Earlier Resolution and Coded Region

The original design sampled the profile at 1,024 positions and used very light Gaussian smoothing. Anchor-adjacent regions were zeroed because they were outside the useful coded region and tended to contain segmentation noise.

Those exact choices were superseded by the angular v2 profile, but the design lessons remain relevant.

## Current Angular Coordinate System

The current profile uses angular coordinates.

The input is:

- anchored outer ear margin,
- original anchor-model points,
- cached ear side.

The scale is the equal-area semicircle radius:

```text
R = sqrt(2A / pi)
```

`A` is the area enclosed by the anchored cut contour and its anchor chord. Opening and alpha-hull radii are fractions of `R`.

The ray pole and 0/180 degree directions come from the original anchor-model points before contour snapping. Snapped anchors still define the cut contour and reference geometry. Cached ear side chooses the left or right semicircle so both ears use the same upper-to-lower anatomical ordering.

## Current Reference and Depth

The pipeline lightly opens the contour, then computes the alpha-hull reference.

For each of 720 evenly spaced angles, the ray selects the furthest forward alpha-boundary intersection. This handles occasional additional crossings near an anchor and assumes the reference is mostly star-convex from the anchor midpoint.

A ray without a reference intersection returns zero instead of failing the ear analysis.

Depth remains local-normal depth, not radial depth. A tangent from the nearby reference path supplies an inward normal, and the nearest original-contour crossing along that normal is divided by `R`.

Positive values are inward tears.

The first and last 5 degrees are excluded from polar intersection, normal, and depth work, then returned as zero because anchor-adjacent readings are less reliable.

## Current Initial Parameters

The current parameters preserve the prior approximate pixel radii under the observed `S:R` ratio of 7:2:

- alpha radius: `0.35R`,
- opening radius: `0.020R`,
- smoothing: Gaussian sigma 2 bins,
- samples: 720 over 0 through 180 degrees.

Visual comparison output is written to `outputs/ear_embedding_v2/`, preserving earlier v1 images in `outputs/ear_embedding/`.

## Validation Set

The original ablation set was chosen so each individual stressed a different design choice:

| individual | photos | reason |
| --- | --- | --- |
| ripley | 2 (2008, 2016) | one massive tear; tests large-tear handling and long-term stability |
| nile | 4 (2014-2017) | train of medium scallops; tests multi-tear resolution and cross-pose stability |
| scar | 3 (2006-2010) | tears that accumulated over time; tests graceful behavior under real change |
| les | 1 | single medium tear; generic case |
| larson | 1 | deep notch beside a wide bay; tests bridged-versus-followed behavior |
| delani | 1 | intact but bowed margin with small nicks; false-positive control |
| adam | 2 (same day) | small tears near the noise floor, one nearly edge-on photo |
| snap | 3-4 (2007-2008) | nearly intact ear; negative control |
