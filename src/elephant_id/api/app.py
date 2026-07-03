"""FastAPI application for the Alphaphant desktop sidecar."""

import hashlib
import mimetypes
import threading
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel

from elephant_id.api import ingest, paths
from elephant_id.api.engine import MatchingEngine
from elephant_id.api.gallery import GalleryData, load_gallery
from elephant_id.api.store import SightingStore

DECISION_ACTIONS = ("confirm", "enroll", "unresolved")


def _sighting_date(photo_ids: tuple[str, ...], fallback: str) -> str:
    """Derive the sighting date from photo identifiers, not the filing time."""
    for photo_id in photo_ids:
        match = ingest.PHOTO_STEM_PATTERN.match(photo_id)
        if match:
            return match["date"]
    return fallback


class IngestRequest(BaseModel):
    """Request body for starting a sighting ingest."""

    folder: str


class MatchRequest(BaseModel):
    """Request body for ranking a sighting against the catalog."""

    top_n: int = 12


class DecisionRequest(BaseModel):
    """Request body for filing a review decision."""

    action: str
    elephant_name: str | None = None


class AppState:
    """Mutable sidecar state shared across requests."""

    def __init__(self, data_dir: Path) -> None:
        """Load the gallery and store; the engine is built in the background."""
        self.data_dir = data_dir
        self.store = SightingStore(data_dir)
        self.gallery: GalleryData = load_gallery(
            paths.gallery_profiles_path(), paths.GALLERY_MANIFEST_CSV
        )
        self.engine: MatchingEngine | None = None
        self.engine_error: str | None = None
        threading.Thread(target=self._build_engine, daemon=True).start()

    def _build_engine(self) -> None:
        """Build the matching engine, then refile previously decided sightings."""
        try:
            engine = MatchingEngine(
                self.gallery, self.data_dir / "gallery_pairwise.npy"
            )
            self._refile_decisions(engine)
            self.engine = engine
        except Exception as error:
            logger.exception(f"Matching engine failed to initialize: {error}")
            self.engine_error = str(error)

    def _refile_decisions(self, engine: MatchingEngine) -> None:
        """Re-apply confirmed and enrolled sightings from previous sessions."""
        for record in reversed(self.store.list()):
            decision = record.get("decision")
            if not decision or decision["action"] not in ("confirm", "enroll"):
                continue
            try:
                profiles, sides, photo_ids, crop_paths = self.store.load_profiles(
                    record["sighting_id"]
                )
                engine.extend(
                    profiles,
                    sides,
                    decision["elephant_name"],
                    _sighting_date(photo_ids, decision["decided_at"][:10]),
                    photo_ids,
                    crop_paths,
                )
            except Exception as error:
                logger.warning(
                    f"Could not refile sighting {record['sighting_id']}: {error}"
                )

    def allowed_image_roots(self) -> list[Path]:
        """Directories the image endpoint may serve from."""
        roots = [
            paths.GALLERY_MANIFEST_CSV.parent,
            self.data_dir,
            paths.DATASET_ROOT,
        ]
        roots.extend(Path(record["folder"]) for record in self.store.list())
        return roots


def create_app(data_dir: Path | None = None) -> FastAPI:
    """Build the sidecar application.

    Args:
        data_dir: Writable state directory; defaults to the repo-local
            Alphaphant outputs directory.
    """
    resolved_data_dir = data_dir or paths.default_data_dir()
    resolved_data_dir.mkdir(parents=True, exist_ok=True)
    state = AppState(resolved_data_dir)

    app = FastAPI(title="Alphaphant Sidecar")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        """Report readiness of the sidecar and matching engine."""
        engine = state.engine
        return {
            "status": "ok",
            "engine_ready": engine is not None,
            "engine_error": state.engine_error,
            "elephants": engine.elephant_count if engine else None,
            "profiles": engine.profile_count if engine else None,
        }

    @app.get("/catalog")
    def catalog() -> list[dict]:
        """List every known elephant."""
        return _require_engine().catalog()

    @app.get("/catalog/{name}")
    def catalog_elephant(name: str) -> dict:
        """Return one elephant's gallery photos."""
        try:
            return _require_engine().elephant_detail(name)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/sightings")
    def create_sighting(request: IngestRequest) -> dict:
        """Start analyzing a sighting folder."""
        folder = Path(request.folder).expanduser()
        try:
            files = ingest.list_photo_files(folder)
        except NotADirectoryError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if not files:
            raise HTTPException(
                status_code=400, detail=f"No photos (.jpg/.jpeg/.png) in {folder}"
            )
        record = state.store.create(folder)
        state.store.update(
            record["sighting_id"], progress={"processed": 0, "total": len(files)}
        )
        threading.Thread(
            target=_run_analysis,
            args=(record["sighting_id"], folder),
            daemon=True,
        ).start()
        return state.store.get(record["sighting_id"])

    @app.get("/sightings")
    def list_sightings() -> list[dict]:
        """List all sightings, newest first."""
        return state.store.list()

    @app.get("/sightings/{sighting_id}")
    def get_sighting(sighting_id: str) -> dict:
        """Return one sighting record."""
        return _get_record(sighting_id)

    @app.post("/sightings/{sighting_id}/match")
    def match_sighting(sighting_id: str, request: MatchRequest) -> dict:
        """Rank catalog elephants against an analyzed sighting."""
        record = _get_record(sighting_id)
        if record["status"] != "ready":
            raise HTTPException(
                status_code=409, detail=f"Sighting is {record['status']}, not ready"
            )
        if record["profile_count"] == 0:
            raise HTTPException(
                status_code=409, detail="Sighting has no usable ear profiles"
            )
        engine = _require_engine()
        profiles, sides, photo_ids, _ = state.store.load_profiles(sighting_id)
        ranked = engine.rank(profiles, sides, photo_ids, top_n=request.top_n)
        match = {
            "matched_at": datetime.now(UTC).isoformat(),
            "candidates": [candidate.to_dict() for candidate in ranked],
        }
        return state.store.update(sighting_id, match=match)

    @app.post("/sightings/{sighting_id}/decision")
    def decide_sighting(sighting_id: str, request: DecisionRequest) -> dict:
        """File the reviewer's identity decision for a sighting."""
        record = _get_record(sighting_id)
        if record.get("decision"):
            raise HTTPException(status_code=409, detail="Sighting already decided")
        if request.action not in DECISION_ACTIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Action must be one of {DECISION_ACTIONS}",
            )

        decision = {
            "action": request.action,
            "elephant_name": None,
            "decided_at": datetime.now(UTC).isoformat(),
        }
        if request.action in ("confirm", "enroll"):
            name = (request.elephant_name or "").strip()
            engine = _require_engine()
            if not name:
                raise HTTPException(status_code=400, detail="elephant_name is required")
            if request.action == "confirm" and not engine.has_identity(name):
                raise HTTPException(
                    status_code=400, detail=f"Unknown elephant: {name}"
                )
            if request.action == "enroll" and engine.has_identity(name):
                raise HTTPException(
                    status_code=400, detail=f"Elephant already exists: {name}"
                )
            profiles, sides, photo_ids, crop_paths = state.store.load_profiles(
                sighting_id
            )
            if len(profiles) == 0:
                raise HTTPException(
                    status_code=409, detail="Sighting has no usable ear profiles"
                )
            engine.extend(
                profiles,
                sides,
                name,
                _sighting_date(photo_ids, decision["decided_at"][:10]),
                photo_ids,
                crop_paths,
            )
            decision["elephant_name"] = name
        return state.store.update(sighting_id, decision=decision)

    @app.post("/dev/analyze")
    def dev_analyze(file: UploadFile) -> dict:
        """Development Lab: run the full photo analyzer on one uploaded image.

        Uploads keep their original stem when it already follows the dataset
        naming convention (warm cache); otherwise a content-hashed Lab stem is
        generated, which requires the live model services.
        """
        suffix = Path(file.filename or "upload.jpg").suffix.lower() or ".jpg"
        if suffix not in ingest.IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400, detail=f"Unsupported image type: {suffix}"
            )
        data = file.file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload")

        original_stem = Path(file.filename or "").stem
        if ingest.PHOTO_STEM_PATTERN.match(original_stem):
            stem = original_stem
        else:
            digest = hashlib.sha1(data).hexdigest()[:8]
            stem = f"Lab{digest}_{datetime.now(UTC).date().isoformat()}_01"
        work_dir = state.data_dir / "dev" / stem
        work_dir.mkdir(parents=True, exist_ok=True)
        photo_path = work_dir / f"{stem}{suffix}"
        photo_path.write_bytes(data)
        try:
            result = ingest.analyze_single_photo(
                photo_path, work_dir, paths.MODEL_CACHE_ROOT
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return result.to_dict()

    @app.get("/image")
    def image(path: str) -> FileResponse:
        """Serve an image from the gallery, dataset, or sighting storage."""
        resolved = Path(path).expanduser().resolve()
        if not any(
            resolved.is_relative_to(root.resolve())
            for root in state.allowed_image_roots()
        ):
            raise HTTPException(status_code=403, detail="Path outside allowed roots")
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail=f"Not found: {resolved}")
        media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        return FileResponse(resolved, media_type=media_type)

    def _require_engine() -> MatchingEngine:
        """Return the engine or fail with 503 while it warms up."""
        if state.engine is None:
            detail = state.engine_error or "Matching engine is still starting"
            raise HTTPException(status_code=503, detail=detail)
        return state.engine

    def _get_record(sighting_id: str) -> dict:
        """Return a sighting record or fail with 404."""
        try:
            return state.store.get(sighting_id)
        except KeyError as error:
            raise HTTPException(
                status_code=404, detail=f"Unknown sighting: {sighting_id}"
            ) from error

    def _run_analysis(sighting_id: str, folder: Path) -> None:
        """Background ingest: extract profiles and mark the sighting ready."""
        try:
            result = ingest.ingest_sighting(
                folder,
                state.store.sighting_dir(sighting_id),
                state.gallery,
                paths.MODEL_CACHE_ROOT,
                progress=lambda processed, total: state.store.update(
                    sighting_id, progress={"processed": processed, "total": total}
                ),
            )
            state.store.save_profiles(
                sighting_id,
                result.profiles,
                result.sides,
                result.photo_ids,
                result.crop_paths,
            )
            state.store.update(
                sighting_id,
                status="ready",
                photos=[photo.to_dict() for photo in result.photos],
                profile_count=len(result.profiles),
                sides=sorted(set(result.sides)),
            )
        except Exception as error:
            logger.exception(f"Ingest failed for sighting {sighting_id}: {error}")
            state.store.update(sighting_id, status="failed", error=str(error))

    return app
