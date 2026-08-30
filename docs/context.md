# AlphaPhant Context

AlphaPhant is a fully automated catalog-matching algorithm for elephant re-identification. Given a high-quality image of each ear, it returns one similarity score per known elephant without further human input. A candidate ranking is the descending view of those scores.

This glossary is the canonical source for domain and technical language. Implementation plans and specifications live in the other top-level docs.

## Sightings and Catalog

**Sighting**:
An immutable snapshot of one observed event, grouping its distinct photo assets under a permanent opaque sighting ID. Its date is required observation metadata used for time-gap research; the elephant may be unknown until a later identity decision.
_Avoid_: Identity, elephant, match

**Photo**:
One immutable original photo asset belonging to a sighting, represented throughout the system only by its permanent opaque photo ID and parent sighting ID. It carries no known-elephant identity, storage location, date, filename, or encoded bytes.
_Avoid_: Image when referring to the domain object

**Photo ID**:
The permanent opaque identifier of one immutable original photo asset. Replacing or re-encoding the original bytes creates a new photo ID.
_Avoid_: Filename, path, content hash

**Sighting ID**:
The permanent opaque identifier of one observed event, independent of elephant identity, date, and its photo IDs.
_Avoid_: Elephant-and-date identifier

**Photo store**:
The storage capability that resolves a Photo to its original encoded bytes without exposing known-elephant identity. A research dataset or application library may own a photo store.
_Avoid_: Dataset, identity resolver

**Dataset**:
The private research dataset object that owns metadata, known-elephant resolution, and a PhotoStore.
_Avoid_: Historical data access, data layer

**Sighting ear pair**:
The selected evidence for one sighting: its opaque sighting ID, one high-quality left-ear Photo, and one high-quality right-ear Photo. Both Photos belong to that sighting; one source Photo may supply both sides.
_Avoid_: Curated sighting, synthetic sighting, cross-sighting pair

**Known elephant**:
An individual elephant already represented in the reference catalog.
_Avoid_: Sighting, folder

**Known-elephant catalog**:
The reference evidence grouped by known elephant and ear side for matching.
_Avoid_: Gallery when naming the domain concept

**Candidate catalog**:
The query-specific view of reference evidence supplied to a catalog matcher, grouped under opaque candidate keys. Every listed candidate has one or more sighting ear pairs.
_Avoid_: Dataset, gallery, ranked candidates

**Candidate key**:
An ephemeral opaque UUID assigned to one matching candidate for an evaluation run. It is stable within that run and carries no known-elephant identity.
_Avoid_: Known-elephant name, photo ID, sighting ID

**Matching candidate**:
A known elephant for which a catalog matcher returns a candidate score.
_Avoid_: Identity decision, automatic identification

**Catalog matcher**:
A complete retrieval algorithm that compares one sighting ear pair with a candidate catalog and returns one comparable similarity score for every matching candidate. AlphaPhant, CurvRank, and MiewID are catalog matchers.
_Avoid_: Candidate scorer, ranker when referring to the catalog-matching role

**Candidate scores**:
The complete association of matching candidates with finite similarity floats for one query. Larger scores indicate stronger matches, and every catalog candidate appears exactly once.
_Avoid_: Ranking, probabilities, confidence labels

## AlphaPhant Algorithm

**Sighting analysis**:
The processing of a sighting ear pair into left- and right-ear tear profiles for catalog matching.
_Avoid_: Per-photo identification, SEEK coding

**Ear localization**:
The determination of where a left or right ear appears in a photo.
_Avoid_: Ear segmentation when referring only to location

**Ear segmentation**:
The extraction of an ear mask and its ear contour from a localized ear.
_Avoid_: Ear localization, body segmentation

**Ear landmark detection**:
The location of the two anatomical endpoints that define the relevant ear contour. Code may call these endpoints the upper anchor and lower anchor.
_Avoid_: Anchor detection in technical writing

**Ear contour**:
The ordered boundary of an ear mask used for tear-profile extraction.
_Avoid_: Ear margin when referring to the computational representation

**Alpha shape**:
The geometric reference constructed from an ear contour during tear-profile extraction.
_Avoid_: Alpha hull

**AlphaTear**:
The current tear-profile extraction algorithm. It uses an alpha shape as an internal geometric reference to locate and measure tears along a prepared ear contour. AlphaTear names the extractor; AlphaPhant names the complete catalog-matching algorithm.
_Avoid_: AlphaPhant when referring only to extraction

**AlphaPhant**:
The project's complete catalog-matching algorithm, which analyzes a sighting ear pair, compares its tear profiles with catalog evidence, and returns candidate scores.
_Avoid_: AlphaTear, AlphaPhant matcher

**Tear profile**:
A one-dimensional, alpha-shape-derived representation of an ear contour used for matching.
_Avoid_: Tear signature, SEEK code, embedding when referring to the current algorithm

**Tear-profile extraction**:
The transformation of a prepared ear contour into a tear profile. AlphaTear is the current extraction algorithm.
_Avoid_: Tear coding, tear-signature extraction

**Tear-profile matching**:
The computation of a similarity score between ears using their alpha-shape-derived tear profiles.
_Avoid_: SEEK matching, tear-signature matching

**Similarity score**:
A numeric measure of how strongly two ears, or a sighting and known elephant, match under the current algorithm.
_Avoid_: Probability, confidence

**Candidate ranking**:
The descending view of candidate scores. Ranking adds no scientific result beyond the scores and is not a separate catalog-matching module.
_Avoid_: Identity decision, prediction

## Evaluation

**Retrieval benchmark set**:
The fixed private collection of real sighting ear pairs, one per sighting, that the identity-retrieval benchmark runs leave-one-sighting-out over. Shorten to benchmark set in running writing. Held under the gitignored research dataset at `dataset/elephants-alive/benchmark/`.
_Avoid_: Evaluation suite, test set

**Parameter-tuning set**:
The image set on which AlphaPhant's non-qualitative matching parameters, such as the profile stretch exponent and penalty weights, are tuned. Disjoint from the retrieval benchmark set. Extraction parameters are not tuned here; they are set qualitatively from the alpha shapes.
_Avoid_: Validation set, training set, dev set

**AI-model datasets**:
The ear-segmentation and landmark models use their own train, validation, and test splits, drawn from singly-sighted elephants so they stay disjoint from all matching data. Those three terms name model data only; the AlphaPhant algorithm has no training phase and tunes its parameters on the parameter-tuning set.

**Identity-retrieval evaluation**:
End-to-end measurement of how a complete retrieval system ranks the correct known elephant for real sighting ear pairs.
_Avoid_: Model benchmark, stage-level evaluation

**Protocol exclusion**:
An evaluation example omitted before any system runs because the dataset cannot support the required comparison, such as a query elephant with no other catalog sighting.
_Avoid_: Extraction failure

## Future Application

**Ear selection**:
The process of choosing one usable left-ear reference photo and one usable right-ear reference photo from all photos in a sighting.
_Avoid_: Matching, per-photo identification

**Ear candidate**:
One suggested ear evidence item before ear selection.
_Avoid_: Selected ear, matching candidate

**Review**:
The human step in which automated suggestions are accepted, corrected, or rejected.
_Avoid_: Verification when it implies a passive check

**Reviewer**:
The person who inspects evidence, compares candidates, and makes an identity decision.
_Avoid_: User when the review role matters

**Analysis package**:
A future application artifact containing sighting photos, automated evidence, selected ears, tear profiles, and correction state.
_Avoid_: SEEK record, raw model output

**Identity decision**:
A reviewer's decision to link a sighting to a known elephant, create a new known elephant, or leave the sighting unresolved.
_Avoid_: Candidate ranking, automatic match

**Unresolved sighting**:
A sighting whose intermediate analysis is saved without linking it to a known elephant.
_Avoid_: Failed sighting

**App Library**:
A future app-controlled workspace containing imported photos, derived assets, and metadata.
_Avoid_: Source camera folder

**Import**:
The future application action that copies a grouped sighting into the App Library and assigns application identifiers.
_Avoid_: Upload when no network transfer occurs

## Legacy Language

**Coding**:
Wording from the SEEK-centered era. Current work uses analysis, tear-profile extraction, matching, candidate ranking, and identity decision.
_Avoid_: Use as a current workflow term
