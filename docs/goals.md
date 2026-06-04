# Elephant ID Project

## Purpose
Elephant ID is a human-in-the-loop platform for identifying individual African elephants from photographs. It combines machine learning, rules-based logic, structured SEEK coding, and expert review to reduce manual effort while preserving human oversight and compatibility with existing field workflows.

## Problem
Elephant identification is valuable for conservation, population monitoring, conflict mitigation, and longitudinal research, but the current process is labor-intensive, subjective, and difficult to scale. Identification often depends on multiple images from a single sighting, because no one photo is guaranteed to show all relevant features.

## Goal
Build a practical production web app that takes a folder of images of a single elephant sighting and turns it into:
1. a draft structured identification record,
2. a human-in-the-loop analysis workflow,
3. a review-ready evidence package with a preview SEEK code,
4. a final reviewed SEEK code,
5. a ranked matching workflow against the existing elephant database,
6. a final filed identity decision.

## Core Principle
The system’s atomic unit is:

**one folder = one sighting of one elephant**

Every image in the folder is treated as evidence about the same individual. The system should combine information across the folder rather than trying to identify the elephant from isolated images.

## Scope

### In Scope
- One-elephant sightings only for the first version
- Dropbox-based ingestion
- Folder-level AI analysis
- Human questions during analysis when useful evidence is ambiguous
- Review-ready evidence package generation
- Preview SEEK generation before review
- Final SEEK generation after human review
- Human review before matching
- Candidate matching against a database
- Filing reviewed results into a long-term database

### Out of Scope for v1
- Multi-elephant sightings in one folder
- Fully automatic filing without human review
- Purely image-based matching without structured coding
- Mobile-first field collection software
- Offline-first operation

## Identification Framework
The project uses SEEK as the structured identification language. SEEK encodes:
- sex
- age
- tusks
- right ear features
- left ear features
- extreme features
- special features

The final reviewed result should be stored both as a structured record and as a final SEEK code string. Before review, the system may show a clearly labeled preview SEEK code, but the canonical final SEEK code is computed only after the reviewer has accepted or corrected the structured fields.

## Human-in-the-Loop Philosophy
The platform is not meant to replace expert judgment. It is meant to:
- prefill likely values,
- reduce repetitive manual work,
- preserve uncertainty where evidence is incomplete,
- ask targeted questions when human input can unlock useful evidence,
- help reviewers reach the correct identification faster,
- provide a shortlist of likely matches rather than a silent automatic answer.

Human input appears in two places:
- During analysis, the system may ask focused questions when an image contains useful but ambiguous evidence. For example, if multiple elephants are detected in one image, the user can choose the correct elephant or discard the image.
- During review, the user inspects the complete evidence package, corrects any field or representative image, approves the final structured record, and only then generates the final SEEK code.

Questions should not be treated as failures. A job should continue running every independent analysis task it can. If all remaining work depends on unanswered questions, the sighting simply waits until answers arrive, then resumes.

## End-to-End Workflow

### 1. Field collection
Conservationists photograph a single elephant during a sighting. The photos are uploaded to Dropbox in one folder, where each image belongs to the same elephant.

### 2. Intake
A signed-in user opens the platform and sees newly available Dropbox folders.

### 3. Import
The user selects a folder and starts the workflow. The platform copies the folder into its own cloud storage, creates a sighting record, and starts analysis.

### 4. AI analysis
The system processes the folder as a whole, using parallel per-image analysis and folder-level aggregation.

For each photo, the system first runs shared image preparation:
- body and feature detection or segmentation,
- body selection and feature filtering,
- image quality scoring,
- view and visibility evidence,
- any targeted questions needed to resolve useful ambiguity.

After shared preparation, the photo analysis splits into parallel field analyzers:
- age evidence,
- gender evidence,
- tusk evidence,
- ear evidence.

Age and gender may be simple model-output analyzers. Tusks and ears may combine model output, geometry, heuristics, and eventually user answers. All field analyzers should produce structured evidence and telemetry rather than only final values.

When one photo or field is blocked on a question, other photos and other field analyzers continue. The job only waits when no runnable automated work remains.

### 5. Sighting aggregation
After per-photo analysis, the system combines evidence across the folder into one sighting-level analysis package. This package includes:
- all raw and filtered model outputs,
- per-photo evidence,
- answered and unanswered question records,
- suggested sighting-level fields,
- representative images and crops,
- a preview SEEK code.

Representative images should support the system's current sighting-level interpretation. The review package should include the best cropped right ear image, best cropped left ear image, useful tusk/front/side views, and photo-level drill-down for the full sighting.

### 6. Review
Once all required analysis and question-dependent work is complete, the folder becomes available for human review. The reviewer checks the representative images, predicted fields, supporting evidence, and preview SEEK code. The reviewer can correct any field, crop, representative image, or individual photo-level output.

Corrections made to representative images should act as field-level overrides when appropriate. For example, if the representative tusk image is corrected from left tusk to right tusk, that correction should become authoritative for the sighting's tusk field unless the reviewer chooses a narrower edit.

The final SEEK code is computed from the reviewed structured fields, not directly from raw model output.

### 7. Matching
After review, the sighting moves to a separate matching workflow. The system compares the reviewed sighting against the database and presents the most likely candidates.

### 8. Filing
The reviewer chooses an existing elephant or creates a new one. The final reviewed record is then filed into the database.

## Success Criteria
The project is successful if it:
- makes SEEK-based coding faster and more consistent,
- works with real field photo folders rather than idealized lab inputs,
- uses human input during analysis without blocking unrelated work,
- preserves a strong final human review step,
- supports scalable matching against a growing database,
- remains usable under low-bandwidth conditions.

## Real-World Constraints
- Field and office internet speeds may be slow
- Reviewers should not need full-resolution images by default
- Upload and review bandwidth matter more than cloud-internal transfer speed
- Different images in a sighting may reveal different critical features
- The system should tolerate incomplete views and uncertain fields
- User questions may be answered asynchronously and at the user's leisure
- The system should not ask the same question twice for the same feature

## Product Design Principles
- Simple user-facing workflow
- Clear separation between coding and matching
- Full traceability from input images to final decision
- Clear separation between raw model output, system suggestions, human answers, review corrections, preview SEEK code, and final SEEK code
- Strong compatibility with conservation workflows
- Backend complexity is acceptable; user-facing complexity is not

## Long-Term Direction
After the first version is working well, the project can expand to:
- sightings with multiple elephants,
- stronger visual matching models,
- more automated candidate ranking,
- broader NGO deployment,
- better support for collaborative review.
