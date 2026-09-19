# Keep selected scoring inside AlphaPhant

AlphaPhant accepts shared preparation and one ear-profile extractor.
Its comparison channels, directional alignment, catalog correction,
and evidence aggregation belong to the algorithm. Standard composition builds
one raw or cached extractor. Evaluation depends only on `CatalogMatcher`.

The 2026-09-16 design review compared this fixed pipeline with typed scoring
strategies and a separate public array scorer. Strategies would preserve
variation used only by rejected experiments. A public array scorer would expose
ordering, deduplication, and evidence multiplicity to callers. Small private
numeric helpers keep those details inside the matcher with fewer interfaces.

`TearMatcher.match_many` returns combined similarities. It does not expose
channel configuration or alignment records. `AlphaPhant.profiles` remains the
source-labelled inspection interface and reuses matching's extraction.

Research keeps experimental implementations and source snapshots in its study
directory. This requires explicit research code when a new mechanism is tested,
but prevents abandoned alternatives from becoming permanent production paths.
The architecture refactor must preserve complete candidate score tables before
any separately documented numerical change is considered.

The follow-up design review removed the caller's ordered extractor sequence.
`AlphaTearExtractor` now owns the complete representation, shares contour
resampling, opening, and polar rays across its reference shapes, and returns ordered
immutable profiles. Its existing cache decorator stores the complete result
under one producer slug. This replaces per-scale orchestration; it does not
add a wrapper around the old sequence. Source-labelled inspection remains
available through `AlphaPhant.profiles`.

For that follow-up, the alternatives were a wrapper around separate extractors,
a new two-dimensional profile value, and direct extraction of the complete
representation. The wrapper would hide the sequence without removing repeated
geometry. A new array value would change inspection and persistence interfaces
without simplifying the current computation. Direct extraction reuses immutable
`TearProfile` values, performs shared work once, and keeps the caller interface
independent of the number of reference shapes.
