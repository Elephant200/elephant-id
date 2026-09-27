# Allow limited elephant overlap between tuning and benchmark

The human-curated tuning and benchmark sets must both be large and of high image quality, but few elephants have enough excellent sightings to fill two elephant-disjoint sets at a high bar. We therefore allow a small number of well-photographed elephants to appear in both sets, with at most five sightings in each and ten in total. The sets still share no sighting or photo, and elephants in the ear-model training data remain disjoint from both. We accept this because AlphaPhant has no training phase and tunes only a few global matching parameters, and each set builds its own catalog, so a shared elephant never supplies reference evidence across sets.

## Consequences

The benchmark estimates performance on unseen sightings; it is no longer strictly on unseen elephants. The rule for splitting a shared elephant's sightings between the sets, and any ceiling on the fraction of shared elephants, is decided at cohort allocation.
