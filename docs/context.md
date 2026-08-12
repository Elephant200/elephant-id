# AlphaPhant Context

AlphaPhant is a fully automated candidate-ranking algorithm for elephant re-identification. Given a high-quality image of each ear, it produces similarity scores and ranks the known-elephant catalog without further human input.

This glossary is the canonical source for domain and technical language. Implementation plans and specifications live in the other top-level docs.

## Sightings and Catalog

**Sighting**:
One observed event involving one elephant and its associated photos. The elephant may be unknown until a later identity decision.
_Avoid_: Identity, elephant, match

**Photo**:
One image belonging to a sighting. In historical data it may already carry a known identity label.
_Avoid_: Image when referring to the domain object

**Dataset photo**:
A labeled historical or evaluation photo whose elephant identity is already known.
_Avoid_: Product photo, imported photo

**Sighting ear pair**:
One high-quality left-ear reference photo and one high-quality right-ear reference photo from the same observed sighting. After the full term is established, use ear pair.
_Avoid_: Curated sighting, synthetic sighting, cross-sighting pair

**Known elephant**:
An individual elephant already represented in the reference catalog.
_Avoid_: Sighting, folder

**Known-elephant catalog**:
The reference evidence grouped by known elephant and ear side for matching.
_Avoid_: Gallery when naming the domain concept

**Matching candidate**:
A known elephant returned as a possible identity in the candidate ranking.
_Avoid_: Identity decision, automatic identification

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

**Tear profile**:
A one-dimensional, alpha-shape-derived representation of an ear contour used for matching.
_Avoid_: Tear signature, SEEK code, embedding when referring to the current algorithm

**Tear-profile extraction**:
The transformation of an ear contour into an alpha-shape-derived tear profile.
_Avoid_: Tear coding, tear-signature extraction

**Tear-profile matching**:
The computation of a similarity score between ears using their alpha-shape-derived tear profiles.
_Avoid_: SEEK matching, tear-signature matching

**Similarity score**:
A numeric measure of how strongly two ears, or a sighting and known elephant, match under the current algorithm.
_Avoid_: Probability, confidence

**Candidate ranking**:
The ordered known-elephant candidates produced by combining left- and right-ear similarity scores.
_Avoid_: Identity decision, prediction

## Evaluation

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
