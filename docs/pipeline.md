# AlphaPhant Pipeline

AlphaPhant receives one `SightingEarPair` and a candidate catalog. It returns one finite similarity score per candidate. Ranking is the descending view of those scores; the pipeline makes no identity decision. MiewID is a separate zero-shot comparison.

## Shared ear preparation

`SightingPreparer` reads each neutral Photo through the image-only PhotoStore and decodes its original bytes as BGR. Cached complete SAM3 features supply ear detections. Cached YOLO landmarks supply anatomical endpoints in full-image coordinates. Existing geometry resolves the declared side, cleans its contour, and produces an immutable `PreparedEar`.

A same-photo pair prepares that Photo once. Composition memoizes preparation and injects the same callable into all scale analyzers. CurvRank and MiewID receive an ear-preparation callable, rather than an AlphaTear analyzer. The comparison composition supplies `SightingAnalyzer.prepare`, which forwards to the same shared computation. Neither comparator receives Dataset or AlphaPhant scores.

`SightingAnalyzer` combines preparation with one profile extractor. It preserves structured failures and never silently drops an ear or substitutes another sighting.

## AlphaTear extraction

The default extraction version is `alpha-tear-v3`, with 1,024 contour points, 720 profile bins, 5-degree trimming, opening fraction 0.020, and smoothing sigma 2.0. The single-scale baseline uses alpha fraction 0.35.

AlphaPhant uses seven immutable versions with alpha fractions 0.11, 0.22, 0.50, 1.10, 2.50, 5.00, and 12.00. Their producer slugs are `alpha-tear-v3-a011`, `alpha-tear-v3-a022`, `alpha-tear-v3-a050`, `alpha-tear-v3-a110`, `alpha-tear-v3-a250`, `alpha-tear-v3-a500`, and `alpha-tear-v3-a1200`. Small rolling disks follow finer contour structure; large disks give broader geometric references. No scale subset is selected per pair.

Any future output-changing extraction change requires a new producer slug. Preparation and extraction identity are part of the version contract.

## Profile channels

Each ear has two channel scores:

- Depth uses the nonnegative tear depths.
- Signed depth change separates the positive and negative angular derivatives of the original profile. Rising and falling slopes cannot substitute for each other. This is not a negative-depth channel.

Each nonnegative row is raised to power 0.75, resampled to 240 bins, and multiplied by Gaussian angular weights with center 120 degrees and standard deviation 35 degrees in the resampled profile coordinate. The derivative is taken before compression and resampling.

The depth channel has one row per alpha-shape scale. Signed depth change has rising and falling rows per scale. All rows in a channel share one alignment. Depth and signed depth change may choose different alignments.

## Directional alignment

`TearMatcher.match_stack_many` compares one query stack with catalog stacks. It searches centered stretches 0.80 through 1.20 in steps of 0.025 and integer shifts within 15% of the 240-bin profile. Values shifted outside the profile are zero.

For each transformation, a row scores by Ruzicka overlap, `sum(min(q,c)) / sum(max(q,c))`. Row scores are averaged before selecting a transformation. The shared score is multiplied by `exp(-(abs(shift_fraction)/0.16)^4)`. The first best transformation wins; zero overlap has neutral alignment.

The implementation uses the equivalent nonnegative-vector formula `(T-D)/(T+D)`, where `T` is the sum of both vectors and `D` their L1 distance. Empty rows score zero. Compiled distance calculations and bounded batches preserve scores to high precision.

The selected comparison is directional: shifts and stretches apply to the query profile while the catalog profile stays fixed.

Depth contributes 0.55 and signed depth change contributes 0.45 to each ear similarity. No rank fusion, per-scale normalization, or ear-decisiveness weight is applied.

## Catalog scoring

For each side, the supplied catalog determines a background strength for each catalog ear: the mean of its ten strongest outgoing similarities to other catalog ears, or all available neighbors when fewer than ten exist. The ear itself is excluded. If no neighbor exists, background strength is zero. Other sightings of its candidate remain included.

A query-to-catalog-ear similarity `s` becomes `2*s - background`. The held-out query never enters the catalog neighborhood. Cached pair similarities do not change which evidence a call is allowed to use. There is no query-side CSLS constant.

One candidate's corrected sighting scores are combined by a similarity-weighted mean. Weights are proportional to `exp((score - maximum)/temperature)`, where temperature is the standard deviation of all corrected catalog-ear scores for this query side. Zero spread uses the maximum. This is a weighted mean, not a probability or log-sum-exp score.

The candidate's left and right scores are averaged equally. Candidate scores need not lie in `[0,1]` and are not confidence probabilities.

## Composition and evaluation

`build_standard_alphaphant` builds the fixed composition. `compose_alphaphant` applies its fixed matching settings to supplied scale analyzers; `build_versioned_analyzers` shares preparation across those scales. `build_standard_analyzer` retains the original single-scale analysis baseline.

There is no public `AlphaPhantConfig` product. The exact ablation uses isolated scientific compositions to remove the catalog correction or replace the similarity-weighted mean with the original maximum.

The constructor names state each component's role: `scale_analyzers`, `channel_matchers`, and `channel_weights`. Custom weights are scaled before normalization so large finite values cannot overflow their sum. Ablation caches identify the script, active package sources, and NumPy/SciPy versions before reusing comparison matrices.

See [results.md](results.md) for measured contributions and [evaluation.md](evaluation.md) for the protocol and reproduction commands.
