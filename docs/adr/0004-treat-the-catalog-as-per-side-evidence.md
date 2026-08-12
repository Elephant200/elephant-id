# Treat the catalog as accumulated per-side evidence

## Decision

Once evidence enters the known-elephant catalog, its original sighting grouping does not constrain matching. For each known elephant, AlphaPhant independently selects the strongest left-ear and right-ear similarity scores across all catalog evidence, then averages those two scores; the winning sides may come from different historical sightings and retain their individual provenance.
