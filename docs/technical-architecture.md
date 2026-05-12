# Elephant ID Technical Architecture

## Overview
Elephant ID is a web platform that ingests a folder of images of a single elephant sighting, runs folder-level AI analysis, produces a draft SEEK-based identification record, supports human review, and then helps match the reviewed sighting against an elephant identity database.

The architecture is split into:
- a frontend web application,
- an authentication layer,
- a backend API and orchestration layer,
- a storage layer,
- a database layer,
- an inference layer,
- a matching layer.

## Tech Stack

### Domain and DNS
- Domain managed through Cloudflare
- Cloudflare used for DNS only
- No Cloudflare proxy in front of the frontend by default
- No Cloudflare proxy required in front of the backend by default

### Frontend
- Next.js
- Hosted on Vercel
- Main site: `elephant-id.org`

### Authentication
- Firebase Auth
- Used for user identity and session management

### Backend API
- FastAPI
- Hosted on Google Cloud Run
- Region: Johannesburg

### API Exposure
- Public API hostname: `api.elephant-id.org`
- Cloud Run should sit behind a GCP HTTPS load balancer
- Use a serverless NEG
- Do not rely on Cloud Run domain mapping

### Database
- PostgreSQL on Cloud SQL
- Region: Johannesburg

### Storage
- Google Cloud Storage
- Region: Johannesburg
- Stores:
  - raw imported images,
  - derived crops and masks,
  - review-ready derivatives,
  - supporting artifacts

### Inference
- Vertex AI
- Region: Johannesburg
- Used for heavier custom-model inference

### Job Orchestration
- Async orchestration service
- Current design assumes a queue-based job model
- One top-level analysis job per folder

### Ingest Source
- Dropbox
- Used as the intake source only
- Not the system of record after import

## Regional Placement

### Frontend
- Vercel should be configured with an appropriate region for server-side compute
- Static asset delivery is handled globally by Vercel’s network

### Backend
The following should all remain colocated in Johannesburg:
- Cloud Run
- Cloud SQL
- Cloud Storage
- Vertex AI

This minimizes latency for dynamic backend traffic and reduces unnecessary cross-region movement of hot data.

## Folder-Centric Design
The system is intentionally folder-centric.

### Rule
**One folder represents one sighting of one elephant.**

### Reason
Each image may provide different evidence:
- left ear
- right ear
- frontal view
- caudal view
- body shape
- tusk visibility

The AI should analyze the folder as a unit and produce one sighting-level draft.

## Request and Processing Flow

### 1. Dropbox discovery
The platform lists available Dropbox folders that have not yet been imported.

### 2. Import request
The user chooses a folder and clicks import.

### 3. Folder analysis
The orchestration layer runs the full analysis workflow on the folder.

Internally this may include:
- per-image analysis
- derived asset generation
- folder-level aggregation
- draft SEEK generation

### 4. Folder completion
Only after the full AI analysis finishes is the sighting marked:
- `Ready for Review`

### 5. Review
The reviewer opens the sighting and sees:
- the images,
- the predicted fields,
- the draft SEEK code,
- supporting crops/overlays where useful.

The reviewer edits or confirms the fields and finalizes the reviewed record.

### 6. Matching
After review, the sighting enters the matching workflow.

The system generates likely candidates from the database and ranks them for human inspection.

### 7. Filing
The reviewer either:
- links the sighting to an existing elephant,
- creates a new elephant identity,
- or leaves the match unresolved.

## AI Pipeline

### Per-image stages
Possible stages include:
- elephant localization
- detection or segmentation
- crop generation
- view classification
- ear localization
- tusk inference
- sex inference
- age inference
- ear-feature inference
- special-feature inference

### Folder-level aggregation
After all images are analyzed, the system combines evidence across the folder to produce one draft sighting record.

This includes:
- sex
- age
- right/left tusk presence
- right ear fields
- left ear fields
- extreme flags
- special feature flags
- draft SEEK code

## Review Architecture

### User-facing rule
Folders are **not reviewable until full AI analysis is finished**.

### Review UI goals
- show all images from the sighting
- show the draft SEEK code
- show structured fields behind the code
- allow edits at the field level
- make review clear without exposing unnecessary backend complexity

### Stored outputs
The platform should keep these layers distinct:
- raw model outputs
- system draft
- human-approved final

## Matching Architecture

### Matching is separate from coding
The coding UI and matching UI should remain distinct.

### Matching inputs
- reviewed structured fields
- final reviewed SEEK code
- optional learned visual features later

### Matching outputs
- ranked candidates
- comparison metadata
- final human match decision

## Performance and Bandwidth Strategy

### Main bottlenecks
1. Upload from the organization into Dropbox
2. Download from cloud storage to the reviewer during review

### Usually not the bottleneck
Cloud-internal transfer between colocated backend services

### Design implications
- avoid loading original full-resolution images by default
- generate compressed review images
- generate targeted crops for ears and other features
- serve originals only on demand

## Image Delivery Strategy
Because the review UI runs in the browser, client-side image delivery is required.

### Recommended pattern
- keep GCS as the canonical storage layer
- serve review assets using short-lived signed URLs
- serve compressed derivatives first
- reserve originals for drill-down only

## Security Model

### Frontend auth
- Firebase Auth signs users in
- frontend sends authenticated requests to the API

### Backend auth
- FastAPI verifies Firebase identity information
- backend enforces organization and workflow permissions

### Storage access
- backend controls which assets are exposed to the browser
- browser receives signed URLs only for allowed assets

## Operational Simplicity Principles
- keep the user-facing workflow simple
- use one top-level job per folder
- hide internal fan-out/fan-in complexity from users
- keep the backend modular but not overengineered
- optimize for maintainability and traceability

## Main Risks and Constraints

### Technical risks
- poor quality or incomplete field imagery
- ambiguous ear views
- bandwidth constraints during review
- queueing/orchestration complexity
- model performance variability across populations and views

### Product risks
- overcomplicating the review experience
- overtrusting model output
- trying to automate final identity decisions too early

## Long-Term Extensions
- multi-elephant folders
- stronger visual re-identification models
- richer candidate ranking
- broader deployment to additional organizations
- active learning from reviewer corrections
- more automation for candidate generation without removing review