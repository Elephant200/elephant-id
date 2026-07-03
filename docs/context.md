# Elephant ID Context

Elephant ID is a human-in-the-loop system for turning grouped elephant photo folders into reviewed identity decisions. This glossary defines the project's canonical domain language; implementation plans and product specs live in other `docs/` files.

## Language

**Sighting**:
One observed one-elephant event represented in the product by a grouped photo folder. A sighting may be unidentified until review, matching, and an identity decision are complete.
_Avoid_: Identity, elephant, match

**Known elephant**:
An individual elephant already present in the long-term database or reference catalog.
_Avoid_: Sighting, folder

**Known-elephant catalog**:
The reference set of known elephants and their stored evidence used for matching.
_Avoid_: Gallery, database when naming the product concept

**App Library**:
The app-controlled workspace folder that contains imported images, derived assets, and SQLite metadata. It may live on an external drive or on the user's regular hard drive.
_Avoid_: App-owned directory, random source folder

**Import**:
The app-controlled action that copies a grouped sighting folder into the App Library and creates generated product identifiers.
_Avoid_: Upload when no network transfer is involved

**Identity decision**:
The reviewer's final decision to link a reviewed sighting to a known elephant or create a new known elephant.
_Avoid_: Automatic match, prediction

**Unresolved sighting**:
A sighting whose intermediate analysis is saved but whose identity has not been linked to the known-elephant catalog.
_Avoid_: Failed sighting

**Photo**:
One image belonging to a sighting. In labeled historical data, a photo may already carry a known elephant identity; in the product workflow, it is evidence for a sighting.
_Avoid_: Image when referring to the domain object

**Dataset photo**:
A labeled historical or evaluation photo whose elephant identity is already known.
_Avoid_: Product photo, imported photo

**Review**:
The human step where automated suggestions become accepted, corrected, or rejected.
_Avoid_: Verification when it implies a passive check

**Reviewer**:
The person using the app to inspect evidence, compare candidates, and make the identity decision.
_Avoid_: User when the identity-review role matters

**Analysis package**:
The intermediate review artifact for one sighting, including source-photo references, segmentation results, selected ear crops, approved tear profiles, and correction state.
_Avoid_: SEEK record, raw model output

**Ear candidate**:
One suggested ear evidence item extracted from a sighting photo before reviewer approval.
_Avoid_: Approved evidence, matching candidate

**Evidence review**:
The step where the reviewer selects the best left and right ear evidence and corrects crops or segmentation before matching.
_Avoid_: Coding

**Canonical ear image**:
The single chosen left-ear photo and single chosen right-ear photo for one sighting; evidence review and contour correction operate only on these two images. The reviewer picks each canonical image from a small subset of ear candidates (an automatic top-ranked default may stand in until the reviewer chooses).
_Avoid_: Ear candidate (a pre-selection suggestion), approved evidence when referring to the image itself

**Tear profile**:
A one-dimensional representation of an approved ear margin used as the current matching signal.
_Avoid_: SEEK code, embedding when referring to the current tear-profile algorithm

**Matching candidate**:
A known elephant returned by the matching workflow as a possible identity for a reviewed sighting.
_Avoid_: Identity decision, automatic identification

**Coding**:
Legacy wording from the SEEK-centered era of the project. Current product docs should use analysis, evidence review, matching, and identity decision instead.
_Avoid_: Use as a product workflow term
