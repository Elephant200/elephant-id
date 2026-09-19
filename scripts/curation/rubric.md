Assess the centered elephant ear for re-identification usefulness. Return one JSON object.

You receive a padded crop with the original background intact. Judge whether the ear's natural shape and outer edge can be read reliably from the supplied pixels. A useful image separates the ear edge from its surroundings, preserves fine edge shape, and shows the ear broadly enough to interpret its outline. Sharp skin texture alone is not evidence of a clear contour.

## Visual criteria

- Assess the visible outer free boundary between the ear's upper and lower anatomical attachment points. These endpoints are `upper_anchor` and `lower_anchor`.
- Real tears are valid contour. Rate how clearly their shape is visible, independently of their distinctiveness.
- `contour_visibility` measures how much of that boundary is exposed: **none** = none exposed; **partial** = substantial sections hidden; **mostly** = most exposed with a localized section hidden; **complete** = exposed throughout. Count occlusion by vegetation, overlapping anatomy, folds, and truncation at the image edge. Interior obstructions are acceptable when the outer boundary remains exposed. An exposed but blurry boundary can be complete in visibility and poor in detail.
- `contour_detail` measures clarity of the exposed boundary, considering blur, resolution, contrast, and compression: **unreadable** = the ear edge cannot be reliably separated from its surroundings; **limited** = the broad edge is traceable, but smaller bends and notches merge or remain uncertain; **clear** = the edge and small changes in its shape are reliably distinguishable, with minor softness; **excellent** = fine edge shape is crisp and unambiguous throughout the exposed portion. A smooth, clearly resolved edge can be excellent without tears. Assess exposed sections here; record hidden sections under visibility.
- `upper_anchor` / `lower_anchor` describe each anatomical endpoint: **not_visible** = hidden or outside the image; **uncertain** = its approximate location is apparent but blur, overlap, or low contrast prevents precise placement; **clear** = the endpoint is directly visible and can be placed confidently.
- `pose_limitation` measures how much the viewing angle or folding compromises the outline: **none** = the broad natural shape is readable; **mild** = slight angle or curvature with little loss of shape evidence; **moderate** = noticeable compression or folding makes part of the shape difficult to interpret; **severe** = an edge-on view or major folding makes the broad shape unreliable despite a potentially sharp silhouette. Ordinary wrinkles and natural curvature are acceptable when the outline remains readable.

## Quality score

`reid_quality` is a holistic integer from 1–10 based on the amount, clarity, and interpretability of visible ear-shape evidence. Weigh the practical effect of limitations rather than averaging the fields:

1. No useful ear evidence.
2. Very poor: fragments, extreme distortion, or severe viewing-angle compression that makes the broad shape unreadable.
3. Poor: major pose, occlusion, or blur limitations, with some interpretable shape evidence.
4. Weak borderline: substantial evidence remains difficult to interpret.
5. Borderline: some useful shape evidence, with important limitations.
6. Usable with substantial limitations.
7. Good: useful shape and detail, with a meaningful localized limitation.
8. Very good: strong evidence with only modest limitations.
9. Excellent: nearly ideal evidence with a small observable limitation.
10. Excellent practical evidence: a broadly readable natural ear shape with clear fine contour, an exposed boundary, and clear endpoints, without any material limitations.

Severe viewing-angle compression belongs in 1–3, usually 2. Slight angle, natural curvature, or harmless interior obstructions can still merit 10 when they leave the relevant shape and detail fully readable. Use 9 when a small observable limitation actually reduces the evidence.

## Output

Return exactly `status`, `ear`, and `reason`:

- `assessed`: a recognizable target ear, including poor views; `ear` contains the fields below.
- `no_visible_ear`: no visible ear; `ear` is null.
- `unresolved`: the target ear cannot be distinguished from surrounding anatomy or another ear; `ear` is null.

`reason` is one short sentence explaining the status. For `assessed`, return exactly these ear fields:

| Field | Allowed values |
|---|---|
| reid_quality | integer 1–10 |
| contour_visibility | none / partial / mostly / complete |
| contour_detail | unreadable / limited / clear / excellent |
| upper_anchor | not_visible / uncertain / clear |
| lower_anchor | not_visible / uncertain / clear |
| pose_limitation | none / mild / moderate / severe |
| limiting_factors | array containing applicable labels: occlusion, vegetation, truncation, blur, folding, insufficient_detail, foreshortening; empty when none apply |
| main_limitation | one factual sentence naming the main visible limitation, or "No material limitation." |

In `main_limitation`, name the affected section and its practical effect: for example, "The lower edge blends into the shaded neck, making its fine shape uncertain." For occlusion, identify the hidden boundary section or endpoint. Include only limiting factors that reduce the visible evidence; use `insufficient_detail` for inadequate resolution or edge separation.
