# Tear Embedding v2: Angular Coordinates

This note describes the second tear-profile coordinate system. It supersedes
the implementation described in `tear-embedding.md`.

## Coordinate and scale

The input is the anchored outer ear margin, ordered from upper anchor to lower
anchor. The profile scale is the equal-area semicircle radius:

\[
R = \sqrt{2A / \pi}
\]

where `A` is the area enclosed by the anchored cut contour and its
anchor chord. Opening and alpha-hull radii are fractions of `R`.

The ray pole is the midpoint of the two anchors. The upper-anchor direction is
0 degrees and the lower-anchor direction is 180 degrees. Cached ear side
chooses the left or right semicircle, so both ears use the same upper-to-lower
anatomical ordering.

## Reference and depth

The pipeline lightly opens the contour, then computes the same alpha-hull
reference as v1. For each of 720 evenly spaced angles, the ray selects the
furthest forward alpha-boundary intersection. This handles the occasional
additional crossing near an anchor and assumes the reference is mostly star
convex from the anchor midpoint.

Depth remains local-normal depth, not radial depth. A tangent from the nearby
reference path supplies an inward normal; the nearest original-contour
crossing along that normal is divided by `R`. Positive values are inward tears.

The first and last 5 degrees are excluded from polar intersection, normal, and
depth work, then returned as zero because anchor-adjacent readings are outside
the coded region and are less reliable.

## Initial parameters

The initial parameters preserve the prior approximate pixel radii under the
observed `S:R` ratio of 7:2:

- alpha radius: `0.35R`
- opening radius: `0.025R`
- smoothing: Gaussian sigma 2 bins
- samples: 720 over 0 through 180 degrees

Visual comparison output is written to `outputs/ear_embedding_v2/`, preserving
the v1 images in `outputs/ear_embedding/`.
