# AlphaPhant Context

This glossary is the primary source for domain and technical language.

## Sightings and Catalog

**Sighting**:
A set of photos of one observed event on a given date, identified by a unique sighting ID. The elephant captured in the sighting may be unknown.
_Avoid_: Identity, elephant, match

**Photo**:
One photo asset belonging to a sighting, represented throughout the system by its photo ID and parent sighting ID. It does not carry elephant identity, storage details, or date.
_Avoid_: Image (when referring to the domain object)

**Photo ID**:
The permanent opaque identifier of one immutable original photo asset, as a UUID.
_Avoid_: Filename, path, hash

**Sighting ID**:
The permanent opaque identifier of one observed event, as a UUID.
_Avoid_: Sighting hash

**Photo store**:
The storage object that resolves a Photo to its original encoded bytes without revealing the elephant identity. A research dataset or application library may own a photo store.
_Avoid_: Dataset, identity resolver

**Dataset**:
The private research dataset object that owns metadata, elephant identity resolution, and a PhotoStore.
_Avoid_: Historical data access, data layer

**Sighting ear pair**:
The selected evidence for one sighting: its opaque sighting ID, one high-quality left-ear Photo, and one high-quality right-ear Photo. Both Photos belong to that sighting; one source Photo may supply both sides.
_Avoid_: Curated sighting, synthetic sighting, cross-sighting pair

**Known elephant**:
An individual elephant already represented in the reference catalog.
_Avoid_: Sighting, folder

**Known Elephant catalog**:
The reference evidence grouped by known elephant and ear side for matching.
_Avoid_: Gallery when naming the domain concept

**Candidate key**:
An ephemeral opaque UUID assigned to one matching candidate for one call to `evaluate`. It is stable across that evaluation's query folds and carries no known-elephant identity.
_Avoid_: Known-elephant name, photo ID, sighting ID

**Matching candidate**:
A known elephant for which a catalog matcher returns a candidate score.
_Avoid_: Identity decision, automatic identification

**Catalog matcher**:
A complete retrieval algorithm that compares one sighting ear pair with a candidate catalog and returns one comparable similarity score for every matching candidate. AlphaPhant, CurvRank, and MiewID are catalog matchers.
_Avoid_: Candidate scorer

## AlphaPhant Algorithm

**Sighting analysis**:
The processing of a sighting ear pair into left- and right-ear tear profiles for catalog matching.
_Avoid_: Per-photo identification

**Analyzed ear**:
One ear's extracted tear profile set (per alpha scale) together with its anatomical side, source photo, and source box.
_Avoid_: Ear representation, ear profile

**Analyzed sighting ear pair**:
The analyzed counterpart of a sighting ear pair, retaining its sighting ID and its left and right analyzed ears.
_Avoid_: Sighting representations, identity decision

**Ear Preparation**
Combines ear localization, ear segmentation, and ear landmark detection to extract the ear contour from an ear image.
_Avoid_: Ear analysis

**Ear contour**:
The ordered boundary of an ear mask used for tear-profile extraction.
_Avoid_: Ear margin when referring to the computational representation

**Alpha shape**:
The geometric reference constructed from an ear contour during tear-profile extraction using a given alpha scale as the scale parameter.
_Avoid_: Alpha hull

**AlphaTear**:
The current tear-profile extraction algorithm. It uses an alpha shape as a geometric reference to locate and measure tears along a prepared ear contour.
_Avoid_: AlphaPhant (when referring only to extraction)

**AlphaPhant**:
The project's complete catalog-matching algorithm, which extracts analyzed ears from a sighting ear pair, compares its tear profiles with catalog evidence, and returns candidate scores.
_Avoid_: AlphaTear (when referring to matching), AlphaPhant matcher

**Tear profile**:
A one-dimensional, alpha-shape-derived representation of an ear contour used for matching. Generated with a single alpha-scale.
_Avoid_: Tear signature, embedding when referring to the current algorithm

**Tear-profile extraction**:
The transformation of a prepared ear contour into a tear profile using a single alpha scale. AlphaTear is the current extraction algorithm.
_Avoid_: Tear-signature extraction

**Similarity score**:
A numeric measure of how similar two ears, or a sighting and known elephant, are.
_Avoid_: Probability, confidence

**Candidate ranking**:
The descending _view_ of candidate scores.
_Avoid_: Identity decision, prediction

## Evaluation

**Retrieval benchmark set**:
The fixed sample of real sighting ear pairs used to evaluate algorithm performance. It contains one pair per sampled sighting, and the data is stored in a benchmark manifest, which contains one row per selected sighting ear pair and lists the elephant identity, sighting ID, left photo ID, and right photo ID. Shorten to benchmark set in running writing.
_Avoid_: Evaluation suite, test set

**Parameter-tuning set**:
The image set on which AlphaPhant's matching parameters, such as the profile stretch exponent, are tuned. Disjoint from the retrieval benchmark set. Shorten to tuning set in running writing.
_Avoid_: Validation set, training set

**AI-model training datasets**:
The ear-segmentation and landmark models use their own train, validation, and test splits. Elephants included here are disjoint from those used in the tuning and benchmark sets. Note that the AlphaPhant algorithm itself has no training phase and tunes its parameters on the parameter-tuning set.

**Retrieval evaluation**:
End-to-end estimation of how a complete retrieval system ranks the correct known elephant for previously unseen elephants and sightings from the target deployment population. Measures statistics such as top-k accuracies and MRR.
_Avoid_: Model benchmark, stage-level evaluation

**Evaluation result**:
The scientific outcome of evaluating one catalog matcher against one retrieval benchmark. Its canonical evidence is the score for every known elephant for each eligible query; ranks, metrics, intervals, and eligibility summaries are derived views rather than separately recorded facts.
_Avoid_: Evaluation run, evaluation report

**Eligible query**:
A benchmark-set or tuning-set sighting ear pair used as a query because its elephant still has catalog evidence after that sighting is held out.
_Avoid_: Protocol-eligible query

**Ineligible query**:
A benchmark or tuning sighting ear pair whose elephant has only one sighting ear pair in total, and thus cannot be used as a query. They are retained as distractors.
_Avoid_: Protocol exclusion, Extraction failure

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

**Identity decision**:
A reviewer's decision to link a sighting to a known elephant, create a new known elephant, or leave the sighting unresolved.
_Avoid_: Candidate ranking, automatic match

**App Library**:
A future app-controlled workspace containing imported photos, derived assets, and metadata.
_Avoid_: Source camera folder

**Import**:
The future application action that copies a grouped sighting into the App Library and assigns application identifiers.
_Avoid_: Upload when no network transfer occurs
