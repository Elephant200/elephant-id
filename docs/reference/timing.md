# Timing Notes

> **Historical.** Point-in-time performance numbers, kept for reference only; not maintained against the current code.

These timings are a point-in-time reference for local analysis performance; to be referenced when implementing performance optimizations.

## `analyze()` Per Phase


| phase                       | mean ms   | % of total |
| --------------------------- | --------- | ---------- |
| ear_tear_profile            | 122.2     | 79.0%      |
| tusk                        | 18.4      | 11.9%      |
| other (grouping/view/glue)  | 13.1      | 8.5%       |
| sam3_body (cached read)     | 0.28      | 0.2%       |
| age                         | 0.22      | 0.1%       |
| sam3_features (cached read) | 0.13      | 0.1%       |
| anchor (cached read)        | 0.13      | 0.1%       |
| gender                      | 0.10      | 0.1%       |
| ear_contour                 | 0.10      | 0.1%       |
| **TOTAL**                   | **154.7** | **100%**   |


## `tear_profile()` Internals


| step                                          | mean ms   | % of tear time |
| --------------------------------------------- | --------- | -------------- |
| alpha_shape                                   | 77.3      | 70.6%          |
| -> `shapely.unary_union` (within alpha_shape) | 52        | 47%            |
| inward_normals_at_origins                     | 10.4      | 9.5%           |
| opened_contour                                | 8.8       | 8.0%           |
| nearest_crossing                              | 7.0       | 6.4%           |
| furthest_ray_crossings                        | 4.4       | 4.0%           |
| densify                                       | 1.6       | 1.5%           |
| ear_side_path                                 | 0.11      | 0.1%           |
| gaussian_filter1d                             | 0.05      | 0.0%           |
| **TOTAL**                                     | **109.7** | **100%**       |


