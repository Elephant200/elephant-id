# AlphaTear geometry

AlphaTear turns a prepared outer ear contour into a one-dimensional tear-depth profile. Its angular coordinate aims to place the same tear at a similar position across photos. Positive values describe inward tears.

## Reference shape

The ear's scale is the equal-area semicircle radius `R = sqrt(2A / pi)`, where `A` is the cleaned contour area. Opening and alpha-shape radii are fractions of `R`, so their pixel sizes follow the ear's size in the photo.

A light morphological opening removes narrow outward segmentation bumps before the alpha shape is built. The alpha-shape reference behaves like a disk rolled along the outer margin: it bridges narrow openings while following broader bays. The seven radius fractions are specified in [pipeline.md](../pipeline.md).

## Position and depth

Original detector landmarks define the polar midpoint and angular directions. Contour-snapped anchors delimit the ear contour. Keeping these roles separate prevents contour snapping from changing the polar coordinate frame. Ear side selects the appropriate semicircle.

For each of 720 evenly spaced angles, the algorithm takes the furthest forward intersection with the reference boundary. A nearby reference tangent defines an inward normal. The nearest original-contour crossing along this normal gives the signed depth, divided by `R`. The angle selects a reference position; the depth measurement follows its local normal.

The first and last 5 degrees are zero. A missing reference intersection also returns zero. Gaussian smoothing uses sigma 2 bins, and missing intersections are reset to zero after smoothing. The opening radius is `0.020R`; the contour is resampled to 1,024 points before geometry computation.

## Limits

The reference assumes that the ear is mostly star-convex from the landmark midpoint. Pose, occlusion, segmentation errors, and landmark errors can therefore change the profile. Normalization removes image scale, not those sources of variation. Matching provides limited shift and stretch tolerance; it cannot recover an unseen contour.

Extraction settings and implementation identity travel together in an immutable producer slug. Any output-changing extraction change requires a new slug.
