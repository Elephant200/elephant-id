# Pin evaluation reports by git commit

## Decision

Identity-retrieval evaluation reports are pinned by the git commit that produced them, marked dirty when the working tree has uncommitted changes. The evaluation suite is the picker-authored sighting-pair directory held in the private, gitignored dataset, organized by known elephant and sighting; it is never committed, so no private identity data enters version control and there is no separate suite manifest or key-resolver file. There are no separate suite, plan, or system fingerprints. The reproduction inputs are code, models, and images; caches only accelerate and never change a result, so a fresh reproduction with an empty cache computes everything from scratch. Hosted-model outputs are recorded rather than re-called, standing in for a model that cannot be shipped, and exact-number reproduction is understood to be internal because the images are private.
