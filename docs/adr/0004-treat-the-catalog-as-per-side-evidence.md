# Treat the catalog as accumulated per-side evidence

Once evidence enters the known-elephant catalog, its original sighting grouping does not constrain matching. AlphaPhant combines left-ear evidence and right-ear evidence separately, then averages the two candidate scores. The sides may draw support from different historical sightings.

The original implementation selected the maximum on each side. The measured method uses a similarity-weighted mean of corrected sighting scores. Replacing that mean with the maximum lost nine percentage points of tuning top-1 accuracy in the 2026-09-14 study. See [results](../research.md) and [pipeline](../pipeline.md) for evidence and the current rule.
