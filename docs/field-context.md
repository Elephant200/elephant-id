# Field Context

## Purpose

This document captures real-world workflow and product constraints from field conversations. It is intended as durable context for agents and contributors when making product, pipeline, architecture, or AI decisions.

## Current Identification Workflow

The current process is organized around field sightings and office review:

- A field outing contains multiple elephant sightings.
- For each sighting, the team keeps a data sheet that records which photos show which elephant.
- In the field, they may photograph one elephant, then photograph the sky as a separator, then photograph another elephant, and so on.
- They often wait at the sighting until the elephant turns enough to show the other side.
- Back at the office, the data sheet is used to group photos into folders, one folder per elephant.

For the software, the practical starting point remains a folder that has already been grouped to one elephant. Future field tools may help decide whether the team has enough coverage before leaving a sighting, but v1 should not require that.

## SEEK In Practice

SEEK is useful as a structured identification language, but it is not the team's preferred day-to-day manual workflow today.

- SEEK coding is unpleasant enough that the team does not rely on it heavily.
- They never built a SEEK-code matching algorithm.
- Ronny, the ranger, can recognize nearly all elephants from memory and does not like making SEEK codes.

This means Elephant ID should use SEEK to make coding faster, more consistent, and easier to review. It should not assume that users are already comfortable coding SEEK manually or that a legacy SEEK matcher exists to plug into.

Classic SEEK should be the primary v1 coding target. The first version should try to produce the existing SEEK fields faithfully and make them easier to review.

At the same time, the system should not make the fixed character string a permanent ceiling. Storage and matching should leave room for future evidence such as Curvrank curvature signatures, contour plots, learned vector embeddings, special markings, scars, body features, or future structured features that do not fit the original SEEK grammar. Those richer features are expansion paths, not required v1 scope.

## Initial Identification Target

Older bulls are the best first target:

- They are easier to match than young bulls and cows.
- They often have more distinctive ear markings.
- Ronny's memory provides strong human oversight for this group.

Young bulls and cows remain important for future studies, but they are harder and should not drive the first pipeline design unless explicitly requested.

Useful field cues:

- Female elephants tend to have more squared-off foreheads.
- Bulls tend to have more hourglass-shaped heads as the tusk sockets spread apart.

## Ear Geometry Notes

Curvrank-style contour work needs a deterministic coordinate system.

- Ear contours must use a consistent side-aware orientation.
- Contour traversal must start and end at deterministic anchor points.
- Internal contour point indices should be stable enough that repeated runs on the same mask produce comparable feature positions.
- Many tears were observed between internal contour points 416 and 800, corresponding roughly to sectors 3, 4, and the boundary around 4.5 on the SEEK figure. The 4.5 reference is an internal geometric hint, not a valid SEEK code value.

## Deployment Constraints

Internet bandwidth is the major product constraint, but it differs by team and site.

- Current internet is not usable for bulk image upload.
- A cloud-backed mode requires better connectivity, likely satellite data or another reliable high-bandwidth path.
- The preferred user experience is a desktop app, but the exact app shell and service packaging are not decided.
- The design should not lock the backend to local-only operation; teams with reliable connectivity may use remote storage, inference, sync, or collaboration services where useful.
- Local operation may require attached storage and hardware capable of running the needed models.
- Models may need to be distilled or otherwise optimized for local inference.

The desktop workflow should remain familiar whether analysis runs locally or uses some remote services.

## Future Field Direction

A valuable future capability is field-time sufficiency feedback: run enough analysis while the team is still at the sighting to say whether they have enough evidence and can leave.

This is a future direction, not a v1 requirement. The v1 workflow should focus on office review of grouped photo folders and reliable older-bull identification.
