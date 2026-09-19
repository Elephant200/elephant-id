# Sighting pair review

From the repository root, start the private local workspace with:

```bash
uv run --group visualization python scripts/curation/review_pairs.py
```

Open <http://127.0.0.1:5055>. The workspace reads the frozen archive assessment
and crop files. It does not run inference or change the dataset and caches.

Review starts with the left ear. Click a loaded crop to move to the right ear;
clicking the right crop immediately saves the complete pair. Use **Reject
sighting** on either side and **Back** or Left Arrow to revisit an earlier
screen. Right Arrow confirms the existing selection on a revisited side and
advances; it does nothing when there is no selection. **Backup** downloads the full portable state and provenance; **Restore**
accepts a matching backup that is at least as recent as the saved local state.

The authoritative state and its recoverable projections are written under
`outputs/curation/sighting-pair-review/`:

- `state.json` contains drafts, accepted and rejected sightings, navigation,
  scroll positions, and the inventory fingerprint.
- `approved-manifest.csv` is the evaluation-compatible complete-pair manifest.
- `approved-manifest.provenance.json` contains crop provenance, protected
  elephant identities, and every photo ID excluded from later model training.

Keep this output separate from historical review and benchmark manifests. Cohort
assignment happens after pair review.

Within each elephant and pair tier, sightings rank by the weighted number of
distinct available photos on each side: 10 = 1, 9 = 0.5, 8 = 0.25, 7 = 0.125.
Each photo counts once per side at its highest crop score. Pair priority is
0.9 times the weaker side strength plus 0.1 times the stronger side strength.
The current review screen is preserved when this ordering changes.
