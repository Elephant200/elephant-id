# Elephant ID Project

## Purpose
Elephant ID is a human-in-the-loop platform for identifying individual African elephants from photographs. It combines machine learning, rules-based logic, structured SEEK coding, and expert review to reduce manual effort while preserving human oversight and compatibility with existing field workflows.

## Problem
Elephant identification is valuable for conservation, population monitoring, conflict mitigation, and longitudinal research, but the current process is labor-intensive, subjective, and difficult to scale. Identification often depends on multiple images from a single sighting, because no one photo is guaranteed to show all relevant features.

## Goal
Build a practical production system that takes a folder of images of a single elephant sighting and turns it into:
1. a draft structured identification record,
2. a draft SEEK code,
3. a human review workflow,
4. a ranked matching workflow against the existing elephant database,
5. a final filed identity decision.

## Core Principle
The system’s atomic unit is:

**one folder = one sighting of one elephant**

Every image in the folder is treated as evidence about the same individual. The system should combine information across the folder rather than trying to identify the elephant from isolated images.

## Scope

### In Scope
- One-elephant sightings only for the first version
- Dropbox-based ingestion
- Folder-level AI analysis
- Draft SEEK generation
- Human review after AI completion
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

The final reviewed result should be stored both as a structured record and as a final SEEK code string.

## Human-in-the-Loop Philosophy
The platform is not meant to replace expert judgment. It is meant to:
- prefill likely values,
- reduce repetitive manual work,
- preserve uncertainty where evidence is incomplete,
- help reviewers reach the correct identification faster,
- provide a shortlist of likely matches rather than a silent automatic answer.

## End-to-End Workflow

### 1. Field collection
Conservationists photograph a single elephant during a sighting. The photos are uploaded to Dropbox in one folder, where each image belongs to the same elephant.

### 2. Intake
A signed-in user opens the platform and sees newly available Dropbox folders.

### 3. Import
The user selects a folder and starts the workflow. The platform copies the folder into its own cloud storage, creates a sighting record, and starts analysis.

### 4. AI analysis
The system processes the folder as a whole. Internally, it may analyze images individually, but the output is one folder-level draft identification package.

### 5. Review
Once the full AI analysis is complete, the folder becomes available for human review. The reviewer checks the predicted fields and the draft SEEK code, then accepts or edits the results.

### 6. Matching
After review, the sighting moves to a separate matching workflow. The system compares the reviewed sighting against the database and presents the most likely candidates.

### 7. Filing
The reviewer chooses an existing elephant or creates a new one. The final reviewed record is then filed into the database.

## Success Criteria
The project is successful if it:
- makes SEEK-based coding faster and more consistent,
- works with real field photo folders rather than idealized lab inputs,
- preserves a strong human review step,
- supports scalable matching against a growing database,
- remains usable under low-bandwidth conditions.

## Real-World Constraints
- Field and office internet speeds may be slow
- Reviewers should not need full-resolution images by default
- Upload and review bandwidth matter more than cloud-internal transfer speed
- Different images in a sighting may reveal different critical features
- The system should tolerate incomplete views and uncertain fields

## Product Design Principles
- Simple user-facing workflow
- Clear separation between coding and matching
- Full traceability from input images to final decision
- Strong compatibility with conservation workflows
- Backend complexity is acceptable; user-facing complexity is not

## Long-Term Direction
After the first version is working well, the project can expand to:
- sightings with multiple elephants,
- stronger visual matching models,
- more automated candidate ranking,
- broader NGO deployment,
- better support for collaborative review.