# Alphaphant Context

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

**Reference sighting**:
A sighting ear pair of a known elephant in the catalog. Matching compares the query with the reference sightings of each candidate.
_Avoid_: Catalog evidence

**Reference ear**:
The left ear or the right ear of a reference sighting.

**Candidate key**:
An opaque identifier that stands in for one known elephant in the catalog that a catalog matcher receives. It does not carry known-elephant identity.
_Avoid_: Known-elephant name, photo ID, sighting ID

**Matching candidate**:
A known elephant for which a catalog matcher returns a candidate score.
_Avoid_: Identity decision, automatic identification

**Catalog matcher**:
A complete retrieval algorithm that compares one sighting ear pair with a candidate catalog and returns one comparable candidate score for every matching candidate. Alphaphant and the CurvRank v2 baseline are catalog matchers.
_Avoid_: Candidate scorer

## Alphaphant Algorithm

**Sighting analysis**:
The processing of a sighting ear pair into left- and right-ear tear profiles for catalog matching.
_Avoid_: Per-photo identification

**Analyzed ear**:
One ear's extracted tear profiles, one for each alpha scale, together with its anatomical side, source photo, and source box.
_Avoid_: Ear representation, ear profile

**Analyzed sighting ear pair**:
The analyzed counterpart of a sighting ear pair. It contains the sighting ID and the left and right analyzed ears.
_Avoid_: Sighting representations, identity decision

**Ear preparation**:
Combines ear localization, ear segmentation, and ear landmark detection to extract the ear contour from an ear image.
_Avoid_: Ear analysis

**Landmark**:
One of the two anatomical points that ear landmark detection finds on an ear: the upper landmark and the lower landmark. The landmarks set the polar frame of the tear profiles.
_Avoid_: Endpoint, keypoint

**Contour anchor**:
A landmark moved to the nearest vertex of the ear-mask boundary. The two contour anchors are the ends of the ear contour.
_Avoid_: Endpoint, snapped landmark

**Ear-flap path**:
Of the two boundary paths between the contour anchors, the path that encloses the larger area with the chord between the contour anchors. Preparation keeps the ear-flap path as the ear contour.

**Ear contour**:
The ordered boundary of an ear mask used for tear-profile extraction: the ear-flap path, from the upper contour anchor to the lower contour anchor.
_Avoid_: Ear margin when referring to the computational representation

**Declared side**:
The side, left or right, that a sighting ear pair gives to an ear photo. The pipeline treats the declared side as correct.

**Inferred side**:
The side of a prepared ear that preparation calculates from the position of its contour anchors in the bounding box of its ear contour. For each declared side, preparation selects the largest prepared ear with the same inferred side.

**Alpha scale**:
The radius of the disc that builds an alpha shape, as a fraction of the ear radius. The disc bridges each indentation that is narrower than its diameter, so the indentation shows as depth. A small alpha scale shows the tears only. A large alpha scale also shows the broad, shallow indentations of the ear margin.

**Alpha shape**:
The geometric reference constructed from an ear contour during tear-profile extraction using a given alpha scale as the scale parameter.
_Avoid_: Alpha hull

**Alphaphant**:
The project's complete catalog-matching algorithm, which extracts analyzed ears from a sighting ear pair, compares its tear profiles with those of the reference sightings, and returns candidate scores.

**Producer slug**:
The name of one producer of cached output, such as a model or a tear-profile extraction configuration, for example `alpha-tear-v1`. The slug identifies all settings that affect the output, so an output change requires a new slug.

**Tear profile**:
The depths of one ear at one alpha scale, at uniform angles from 0° to 180°. Each tear profile also contains its measured interval.
_Avoid_: Tear signature, embedding when referring to the current algorithm

**Measured interval**:
The angles at which tear-profile extraction measures depths: 5° to 175°, both ends excluded. The measured interval excludes the angles near the landmarks to reduce noise. The depths outside the measured interval are zero.

**Depth**:
The distance from the alpha-shape boundary to the ear contour, divided by the ear radius. Tear-profile extraction measures the distance inward. A positive depth shows a tear: the contour is inside the alpha shape. A negative depth shows that the contour is outside the alpha shape.

**Ear radius**:
The radius of a half-disc that has the same area as the ear. Depths are fractions of the ear radius. Thus, ears of different image sizes have comparable depths.

**Tear-profile extraction**:
The transformation of a prepared ear contour into one tear profile for each alpha scale. It uses an alpha shape as a geometric reference to locate and measure tears along the contour.
_Avoid_: Tear-signature extraction

**Comparison grid**:
The uniform angles from 0° to 180° that the comparator uses. Each angle is one bin. Each channel has one value in each bin.

**Channel**:
One vector of values on the comparison grid, for one alpha scale. Each ear has six channels: one depth channel and one gradient channel for each alpha scale.
_Avoid_: Row

**Depth channel**:
A tear profile that is ready for comparison after steps such as resampling and negative-depth clipping.
_Avoid_: Depth row, comparison row

**Gradient channel**:
The rate of change of a tear profile's clipped depth along the angles, ready for comparison. Each value is the square root of the magnitude, with the sign kept. It is positive where a tear begins and negative where a tear ends. It is zero outside the measured interval.
_Avoid_: Gradient row

**Channel weight**:
The part of the alignment score that the depth channels or the gradient channels supply. The two channel weights add to 1.

**Alignment**:
One stretch and one shift. The comparator applies the same alignment to all channels of the query ear. The comparator moves only the query ear.
_Avoid_: Variant, transform

**Stretch**:
A scale factor for a query channel. The comparator stretches the channel about the center of the comparison grid. A stretch above 1 makes the channel wider. A stretch below 1 makes the channel narrower.

**Shift**:
A movement of a query channel along the comparison grid, in whole bins. A positive shift moves the channel to larger angles. A negative shift moves the channel to smaller angles.
_Avoid_: Offset, rotation

**Shift penalty**:
A factor from 0 to 1 that decreases as the shift increases. It makes the comparator prefer small shifts.

**Stretch penalty**:
A factor from 0 to 1 that decreases as `|ln t|` increases, where `t` is the stretch. It is 1 at stretch 1, and it is the same for `t` and `1 / t`. It makes the comparator prefer stretches near 1.

**Angle weight**:
The importance of each bin of the comparison grid: a Gaussian of the angle with tuned center and width. It is largest on the lower part of the ear contour, because the upper half of an elephant ear is almost always folded back. The angle weight stays at the same angles when the comparator aligns the query.

**Weighted real-valued Jaccard**:
A similarity of two channels from 0 to 1: the magnitude that the two channels share divided by their combined magnitude, with each bin multiplied by its angle weight. In a bin where both values are positive, or both are negative, the channels share the smaller magnitude. In a bin with opposite signs, the channels share nothing, and both magnitudes count in the combined magnitude. Equal nonzero channels give 1. Channels with no common nonzero bin of the same sign give 0.
_Avoid_: Overlap, overlap score

**Alignment score**:
The match quality of a query ear at one alignment against one reference ear: the sum of the channel means, each multiplied by its channel weight, multiplied by the shift penalty and the stretch penalty. There are two channel means: the mean weighted real-valued Jaccard similarity of the depth channels, and the same mean for the gradient channels.

**Ear similarity**:
The highest alignment score of a query ear against one reference ear of the same side, from 0 to 1. The comparator moves only the query, so the similarity is asymmetric. 
_Avoid_: Similarity score, ear comparison, profile similarity

**Log-mean-exp**:
A smooth maximum of `n` values `s` with temperature `τ`: `τ * (logsumexp(s / τ) - log n)`. It lies between the mean and the maximum. Equal values give that value. A rise in any one value raises the result.

**Left score**, **right score**:
The score of one candidate on one side: the log-mean-exp of the ear similarities of the candidate's reference ears on that side.
_Avoid_: Side winner

**Candidate score**:
The mean of a candidate's left score and right score. A larger candidate score is stronger evidence.
_Avoid_: Probability, confidence, similarity score

**Candidate ranking**:
The descending _view_ of candidate scores.
_Avoid_: Identity decision, prediction

## Evaluation

**Retrieval benchmark set**:
The fixed sample of real sighting ear pairs used to evaluate algorithm performance. It contains one pair per sampled sighting. Shorten to benchmark set in running writing.
_Avoid_: Evaluation suite, test set

**Parameter-tuning set**:
The image set on which Alphaphant's settings, such as the alpha scales and the temperature, are tuned. It shares no elephant, sighting, or photo with the retrieval benchmark set. Shorten to tuning set in running writing.
_Avoid_: Validation set, training set

**AI-model training datasets**:
The ear-segmentation and landmark models use their own train, validation, and test splits. Elephants included here are disjoint from those used in the tuning and benchmark sets. Note that the Alphaphant algorithm itself has no training phase and tunes its parameters on the parameter-tuning set.

**Retrieval evaluation**:
End-to-end estimation of how a complete retrieval system ranks the correct known elephant for previously unseen elephants and sightings from the target deployment population. Measures statistics such as top-k accuracies and MRR.
_Avoid_: Model benchmark, stage-level evaluation

**Evaluation result**:
The scientific outcome of evaluating one catalog matcher against one retrieval benchmark. Its canonical evidence is the score for every known elephant for each eligible query; ranks, metrics, intervals, and eligibility summaries are derived views rather than separately recorded facts.
_Avoid_: Evaluation run, evaluation report

**Eligible query**:
A benchmark-set or tuning-set sighting ear pair used as a query because its elephant still has reference sightings after that sighting is held out.
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
