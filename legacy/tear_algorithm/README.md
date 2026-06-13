# Ear Tear Extraction Testing

Snapshots of the exploration that produced `src/elephant_id/tears.py` (+ `geometry.py`). They are NOT maintained and most no longer run (they import script modules that have since moved); they are kept for the record of what was tried and why it was rejected or adopted.


| script              | what it explored                                                                                                                                           | conclusion                                                                                                                                                           |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `curvature.py`      | Implementation of `curvrank` integral curvature on ear contours (with `coding/curvature.py`, now removed; `resample2d` survives in `elephant_id.geometry`) | replaced by the tear-profile pipeline                                                                                                                                |
| `arPLS.py`          | asymmetric least-squares smoothing as the intact-margin baseline                                                                                           | rejected: frequency-domain separation bleeds into large tears; tears/bays separate by mouth *width*, not wavelength                                                  |
| `tear_baseline.py`  | arPLS / asls / airpls baseline comparison on a sample contour                                                                                              | same conclusion; kept the signed-deviation helpers used by tear_envelopes                                                                                            |
| `tear_shoulder.py`  | convex-hull defect depth + corner sharpness ("shoulder") tear detection                                                                                    | superseded by the envelope/normal-scan formulation                                                                                                                   |
| `tear_envelopes.py` | convex hull vs alpha-hull radii vs arPLS as references                                                                                                     | adopted the Delaunay alpha hull (the fast equivalent of morphological closing); found the PyPI `alphashape` package broken (ordered-tuple edge bookkeeping)          |
| `tear_coords.py`    | anchor-pinned coordinates (pole angle vs arclength), radial vs normal scans, dual hull/alpha profiles, gated tear events                                   | adopted: arclength coordinate, nearest-crossing normal scan, morphological opening, gated events with optimal assignment; pole angle kept only as an annotation idea |
| `tear_hull.py`      | hull-only profile (no alpha) for matching                                                                                                                  | the hull profile matches well but conflates ear shape with tear evidence; superseded by the single alpha-reference profile + test matchers                           |


The settled pipeline these converged on:

```
margin -> light morphological opening -> alpha hull (radius = 0.10 x hull arclength)
       -> inward-normal depth scan -> normalized 1-D tear profile
```

implemented in `src/elephant_id/tears.py`, validated by
`scripts/evaluate.py`, visualized by `scripts/ear_embedding.py`.