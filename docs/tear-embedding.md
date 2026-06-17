# The Tear Embedding: Goals and Key Decisions

This document records the design of the tear-embedding stage (`src/elephant_id/coding/ears/tear_profile.py`) and the reasoning behind each major choice, in roughly the order the decisions were made. It is written to support an ablation study: every rejected alternative described here was implemented and tested before being set aside, and frozen snapshots of that code are preserved in `legacy/tear_algorithm/`. The companion figures referenced below live in `docs/assets/`.

## Goal

Elephants are individually identifiable by the tears along the margins of their ears. The goal of this stage is to reduce a photographed ear margin — an ordered sequence of contour points running between two anatomical anchor landmarks — to a single one-dimensional array in which position corresponds to a stable location along the intact margin, large values correspond directly to tears, and the overall pattern reads as a flattened version of the ear. Two photographs of the same ear, taken years apart and from different angles, should produce arrays whose tear bumps coincide; the array is the feature that both the SEEK coding stage and the matching stage consume.

![The settled pipeline](assets/tear_embedding_pipeline.png)

The settled pipeline is short: a light morphological opening of the margin, an alpha-hull estimate of the intact margin, a depth scan along that estimate's inward normals, and normalization. The remainder of this document explains why each of those steps survived and what they replaced.

## Decision 1: Reject smoothing baselines (arPLS)

My first approach estimated the intact margin with asymmetric penalized-least-squares smoothing (arPLS), treating tears as outliers below a smooth baseline. This fails for a structural reason: a smoothing baseline separates signal from background by *wavelength*, but tears come in very different sizes. A stiffness that holds the baseline taut across a massive tear is far too stiff to expose small nicks, and a stiffness that exposes small nicks lets the baseline sag — "bleed" — into any large tear, truncating its measured depth (right panel of the figure below). I attempted an adaptive variant that detected wide tears in a stiff first pass and locally re-weighted the smoother; it grew complicated, introduced its own failure modes at tear shoulders, and still did not measure large and small tears consistently. The lesson I drew is that tears and anatomical concavities are not distinguished by wavelength at all, which motivated the geometric approach below.

![Reference comparison](assets/tear_embedding_references.png)

## Decision 2: Reject the convex hull as the intact-margin estimate

The convex hull is the simplest geometric envelope, and a depth profile measured from it does carry useful signal. But the hull cannot follow any concavity, so the broad, shallow bays that are normal ear anatomy are bridged and read as wide false tears (left panel above; the lower margin of this ear is intact, merely bowed). The hull is also maximally sensitive to its supporting points: because it must contain every contour point, a single outward error anywhere relocates entire hull edges and perturbs depth readings far from the error. I therefore needed a reference that distinguishes features by their *shape* — and the observation that matters is that tears are narrow-mouthed relative to their depth, while anatomical bays are wide-mouthed and shallow.

## Decision 3: Adopt the rolling-ball reference (morphological closing / alpha hull)

A disk rolled along the outside of the margin realizes exactly that distinction: the disk cannot enter a concavity whose mouth is narrow relative to the disk, so tears are bridged, while it follows wide gentle bays without resistance. I first implemented this as a raster morphological closing, which produced exactly the reference I wanted but was far too slow at full image resolution. The alpha hull (computed from a Delaunay triangulation) is the fast vector-geometry equivalent of the same rolling-ball construction and runs in a fraction of a second. One practical note for reproducers: the widely used `alphashape` package on PyPI produced collapsed, fragmented output on this data due to an internal bookkeeping defect, and I implemented the construction directly instead.

The disk radius is the one genuinely important parameter. I swept it and inspected the resulting tear profiles against the photographs: at roughly 20% of ear scale the bridged-versus-followed split matched my judgment of tear versus anatomy on every test ear, and results were qualitatively stable between 20% and 40%. I state the radius — and every other length in the pipeline — as a fraction of the arc length of the convex hull between the two anchors, after finding that the initial bounding-box normalization changed when the same ear was merely rotated in the image. The hull arc length is invariant to rotation and, because the hull bridges tears, also invariant to tear formation.

## Decision 4: The coordinate system — three attempts

Assigning each depth reading a stable position was the hardest part of the design, and I went through three formulations.

**Pole at the anchor midpoint.** My first idea was polar: place a pole at the midpoint between the two anchors and parameterize the margin by angle, so that a given angle would name the same anatomical place in every photograph. This fails because the margin is, in general, non-star-convex: a single ray from the pole can cross a torn margin many times (the figure below shows a ray with three crossings), so "the depth at angle θ" is not even well defined.

**Pole at the centroid.** Moving the pole to the centroid of the reference shape improves the *reference* somewhat, but it is still not star-convex. I could ignore this and rasterize depths anyway, but then rays become increasingly oblique to the margin far from the pole and thin flaps and steep tear walls alias across neighboring rays. Moreover, the location of the tear becomes skewed, as it does not correspond to the actual opening of the tear.

![Coordinate-system failures](assets/tear_embedding_pole.png)

**Normal scan from the reference (adopted).** The resolution was to abandon poles entirely. I sample the reference path at evenly spaced arc-length positions and, at each, cast a ray along the reference's inward normal, reading the nearest margin crossing. Over an intact stretch the reference lies on the margin and the depth is zero; across a tear bridge the rays drop perpendicularly onto the tear, so the profile traces the tear's actual silhouette. This construction makes no star-convexity assumption about the ear at all, and arc length along the reference is the natural position coordinate: because the reference bridges tears, a tear does not shift the coordinates of features beyond it, which was the original defect of simply numbering contour points. One implementation detail proved important: the scan must take the *nearest* crossing along the ray line rather than the first crossing in front of the origin, because the reference does not pass exactly through margin points once the opening below is applied, and the naive rule occasionally fired rays across the entire ear.

## Decision 5: A light morphological opening for segmentation noise

Any envelope-based reference inherits a vulnerability: it must contain every contour point, so a single spurious *outward* excursion from the segmentation model lifts the reference and corrupts depth readings well away from the error. The remedy is the morphological dual of the closing already in use: a light opening (radius 0.7% of ear scale) shaves outward protrusions narrower than its disk before the reference is built, while tears — which are inward — are untouched, and depths are still measured against the original margin. In a controlled experiment with a synthetic spur, the opening reduced the spur's collateral effect on distant readings several-fold and eliminated its false local double-bump (figure below). I keep the opening deliberately light: an earlier, more aggressive variant measurably erased fine outward texture that, on nearly intact ears, itself carries identity information.

![Opening robustness](assets/tear_embedding_opening.png)

## Decision 6: Resolution, smoothing, and the coded region

Three smaller choices complete the pipeline. The profile is sampled at 1,024 positions — about 0.1% of margin length per bin, several pixels at typical photo resolution, with the smallest coded tears spanning roughly ten bins; anything within a factor of two behaves identically, and the number matches the contour resampling used elsewhere. The profile receives a very light Gaussian smoothing, which an ablation showed to measurably help. Finally, the first 20% and last 10% of the coordinate are zeroed: these anchor-adjacent regions are outside the coverage of the SEEK coding scheme, and on every ear examined, anything appearing there was segmentation noise rather than anatomy.

## Validation and the ablation set

I validated the embedding by checking, photograph against profile, that every bump corresponds to a visible tear and every visible tear to a bump, and by confirming that profiles of the same individual align across photographs — including pairs taken eight years apart in which the tears themselves had visibly grown. As a quantitative check, a draft matcher that reduces each profile to its significant peaks and pairs them across photographs recovers the correct individual for nearly every query in the pilot set; I treat that as evidence the feature carries identity, while the matching algorithm itself remains a separate concern outside this document's scope.

The pilot set was chosen so that each individual stresses a different decision, and I recommend the same set for the ablation study:

| individual | photos | why it is in the set |
|---|---|---|
| ripley | 2 (2008, 2016) | one massive tear; tests large-tear handling (arPLS bleed, radius) and long-term stability |
| nile | 4 (2014–2017) | a train of medium scallops; tests multi-tear resolution and cross-pose position stability |
| scar | 3 (2006–2010) | tears that accumulated over time; tests graceful behavior under real change |
| les | 1 | single medium tear; the generic case |
| larson | 1 | one deep, narrow-mouthed notch beside a wide bay; tests the bridged-versus-followed split directly |
| delani | 1 | an intact but gently bowed margin with two small nicks; the false-positive control for the reference choice |
| adam | 2 (same day) | small tears near the noise floor, one photo nearly edge-on; probes the lower limits (pose and tear size) |
| snap | 3–4 (2007–2008) | a nearly intact ear; the negative control — its profile should stay near zero |
