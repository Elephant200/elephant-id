# Current Status

Alphaphant is mid-rebuild. Sloppy experimental changes on `optimized` are being transferred to `main` ticket by ticket. Consult the user when documentation and code disagree.

## Rebuild

- Tickets live in `.scratch/alphaphant-rebuild/`; the table in its `README.md` is the source of truth.
- Done: tickets 01–14. Next: 15 (performance for deployment).
- Tuning data is `dataset/elephants-alive/benchmark/tuning_final.csv`. It shares no elephant with `benchmark_final.csv`.
- The method is frozen. Its one benchmark result is `13-frozen-benchmark`: top-1 84.3% (macro 81.8%), MRR 0.881.
- Ticket 14 ablated the frozen method on the benchmark. The paper tables are in `.scratch/alphaphant-paper-figures/ablations.md`.
- The current tuning-set checkpoint is `13-retuned` (top-1 87.9%). The comparison point is `optimized-96f6514-tuning-final` (top-1 88.6%). Compare new checkpoints on `tuning_final.csv` only.
- The tear-profile extraction producer slug is `alpha-tear-v1`.

## Pipeline

Shared preparation retrieves photos, runs ear segmentation and landmark detection, and resolves both ears. Alphaphant extracts one signed tear profile per alpha scale for each ear, computes ear similarities for corresponding sides, takes the log-mean-exp per side as the left score and right score, and averages them into the candidate score.

SAM3 supplies segmentation. YOLO supplies landmarks. Alpha-shape extraction supplies tear profiles.

## Documentation

- [Architecture](architecture.md): ownership, dependencies, and caches.
- [Pipeline](pipeline.md): extraction and scoring.
- [Evaluation](evaluation.md): identity-retrieval evaluation.
- [Context](context.md): terms and definitions.
