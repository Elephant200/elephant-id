# Product Workflow

Elephant ID turns one grouped sighting folder into a reviewed identity decision. The reviewer must be able to inspect and correct the evidence before matching runs.

## Core User Journey

1. Import one grouped sighting folder into the App Library.
2. Analyze the source photos and build an analysis package.
3. Review the analysis package.
4. Approve one left-ear and one right-ear evidence set.
5. Generate tear profiles from the approved evidence.
6. Match the sighting against the known-elephant catalog.
7. Compare the top aligned candidates.
8. Log an identity decision.

The product is not a SEEK coding tool. It is an evidence-review and identity-decision tool.

## App Library

The App Library is an app-controlled workspace folder. It may live on an external drive or on the user's regular hard drive.

The app controls import into the App Library. For v1, the reviewer imports one folder that already represents one sighting of one elephant. The app copies source photos into the App Library by default, stores metadata in SQLite, and assigns generated product identifiers. Filenames must not encode elephant identity, because the identity may be unknown at import time.

V2 may add batch import of multiple sighting folders or raw camera-dump grouping. V1 does not require those workflows.

## Analysis Package

The analysis package is the intermediate review artifact for one sighting. It contains the source-photo references, segmentation results, candidate ear crops, selected ear evidence, generated tear profiles, and any reviewer corrections.

The reviewer first asks: did the system extract the right evidence from this sighting?

For v1, the analysis package must support:

- viewing source photos,
- viewing segmentation overlays,
- viewing ranked left-ear and right-ear crop candidates,
- selecting one best left ear and one best right ear,
- editing the crop and ear polygon or segmentation region,
- viewing the resulting tear profile next to the crop and overlay,
- saving whether the evidence was manually corrected.

The app should show a ranked candidate grid for each side, not an unfiltered wall of images. Show the top three ear candidates per side by default.

## Evidence Review Gate

Matching must not run until evidence review is complete.

For v1, the reviewer must approve both:

- one usable left-ear crop and segmentation,
- one usable right-ear crop and segmentation.

If either side cannot be approved, the sighting is saved as unresolved. One-sided matching is a v2 capability and should arrive with stronger learned embeddings or other compensating signals.

After the reviewer changes a selected ear, crop, or segmentation, the app regenerates the tear profile from the corrected evidence. Candidate matching starts only after the approved profiles exist.

## Candidate Comparison

After evidence review, the app matches the sighting against the known-elephant catalog.

The default comparison view should show the top five known-elephant candidates, with an option to show more. Each candidate should show the query left and right ears beside the matched catalog ears, with aligned tear profiles so the reviewer can judge whether the similarity is meaningful.

The reviewer asks: do these aligned signals support this identity?

The current matching signal is the tear-profile matcher. Future matching may combine tear profiles with learned embeddings, visual descriptors, scars, body features, reviewer history, or other evidence.

## Identity Decision

The reviewer owns the final decision. The system ranks and explains candidates; it does not silently identify the elephant.

The decision states are:

- existing known elephant,
- new known elephant,
- unresolved.

Unresolved means the intermediate analysis has been saved, but the sighting is not filed into the known-elephant catalog.

The decision log should include:

- decision state,
- selected known elephant when applicable,
- reviewer and timestamp,
- selected left and right evidence references,
- whether each evidence set was corrected,
- candidate scores shown at decision time,
- optional reviewer note.

Operational telemetry should exist so analysis quality, correction frequency, and workflow problems can be understood later. The product documentation does not require a detailed telemetry schema yet.

## V1 Boundaries

V1 is intentionally narrow:

- input is one already-grouped one-elephant sighting folder,
- normal use works offline after setup,
- original source locations are not mutated,
- user-facing review must be obvious and responsive,
- analysis can run in the background and may be slower than review,
- both ears are required for matching,
- the reviewer makes every identity decision.

The design should not prevent later batch import, raw camera-dump grouping, one-sided matching, learned embedding models, remote collaboration, or field-time sufficiency feedback.
