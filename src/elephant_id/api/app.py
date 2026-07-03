"""FastAPI application for the Alphaphant desktop sidecar.

Routes parse requests, delegate to :class:`SightingWorkflow`, and translate
workflow errors into HTTP status codes; business logic lives in
``elephant_id.api.workflow``.
"""

import hashlib
import mimetypes
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from elephant_id.api import ingest, paths
from elephant_id.api.analysis import decorate_record
from elephant_id.api.engine import MatchingEngine
from elephant_id.api.gallery import GalleryData, load_gallery
from elephant_id.api.store import SightingStore
from elephant_id.api.workflow import (
    SightingWorkflow,
    WorkflowConflict,
    WorkflowInvalid,
)

EngineFactory = Callable[[GalleryData, Path], MatchingEngine]


class IngestRequest(BaseModel):
    """Request body for starting a sighting ingest."""

    folder: str


class MatchRequest(BaseModel):
    """Request body for ranking a sighting against the catalog."""

    top_n: int = 12


class EvidenceApprovalRequest(BaseModel):
    """Request body for approving one left and one right ear candidate."""

    left_candidate_id: str
    right_candidate_id: str


class DecisionRequest(BaseModel):
    """Request body for filing a review decision."""

    action: str
    elephant_name: str | None = None


def _default_engine_factory(gallery: GalleryData, cache_path: Path) -> MatchingEngine:
    """Build the production matching engine."""
    return MatchingEngine(gallery, cache_path)


class AppState:
    """Mutable sidecar state shared across requests.

    Args:
        data_dir: Writable state directory.
        store: Sighting persistence; a real store is created when omitted.
        gallery: Known-elephant catalog data; loaded from disk when omitted.
        engine_factory: Builds the matching engine; production factory when
            omitted. The engine is always built on a background thread.
    """

    def __init__(
        self,
        data_dir: Path,
        store: SightingStore | None = None,
        gallery: GalleryData | None = None,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        """Wire dependencies and start the background engine build."""
        self.data_dir = data_dir
        self.store = store or SightingStore(data_dir)
        self.gallery = gallery or load_gallery(
            paths.gallery_profiles_path(), paths.GALLERY_MANIFEST_CSV
        )
        self.workflow = SightingWorkflow(
            self.store, self.gallery, paths.MODEL_CACHE_ROOT
        )
        self.engine: MatchingEngine | None = None
        self.engine_error: str | None = None
        self._engine_factory = engine_factory or _default_engine_factory
        threading.Thread(target=self._build_engine, daemon=True).start()

    def _build_engine(self) -> None:
        """Build the matching engine, then refile previously decided sightings."""
        try:
            engine = self._engine_factory(
                self.gallery, self.data_dir / "gallery_pairwise.npy"
            )
            self.workflow.refile_decisions(engine)
            self.engine = engine
        except Exception as error:
            logger.exception(f"Matching engine failed to initialize: {error}")
            self.engine_error = str(error)

    def allowed_image_roots(self) -> list[Path]:
        """Directories the image endpoint may serve from."""
        roots = [
            paths.GALLERY_MANIFEST_CSV.parent,
            self.data_dir,
            paths.DATASET_ROOT,
        ]
        roots.extend(Path(record["folder"]) for record in self.store.list())
        return roots


def create_app(data_dir: Path | None = None, state: AppState | None = None) -> FastAPI:
    """Build the sidecar application.

    Args:
        data_dir: Writable state directory; defaults to the repo-local
            Alphaphant outputs directory.
        state: Prebuilt state (with injected store/gallery/engine) for tests.
    """
    if state is None:
        resolved_data_dir = data_dir or paths.default_data_dir()
        resolved_data_dir.mkdir(parents=True, exist_ok=True)
        state = AppState(resolved_data_dir)
    workflow = state.workflow

    app = FastAPI(title="Alphaphant Sidecar")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(WorkflowInvalid)
    def _invalid(request: object, error: WorkflowInvalid) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.exception_handler(WorkflowConflict)
    def _conflict(request: object, error: WorkflowConflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

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
            "data_dir": str(state.data_dir),
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
            target=workflow.run_analysis,
            args=(record["sighting_id"], folder),
            daemon=True,
        ).start()
        return decorate_record(state.store.get(record["sighting_id"]))

    @app.get("/sightings")
    def list_sightings() -> list[dict]:
        """List all sightings, newest first."""
        return [decorate_record(record) for record in state.store.list()]

    @app.get("/sightings/{sighting_id}")
    def get_sighting(sighting_id: str) -> dict:
        """Return one sighting record."""
        return decorate_record(_get_record(sighting_id))

    @app.get("/sightings/{sighting_id}/analysis")
    def get_analysis(sighting_id: str) -> dict:
        """Return the V1-preview analysis package for evidence review."""
        _get_record(sighting_id)
        return workflow.analysis_package(sighting_id)

    @app.post("/sightings/{sighting_id}/approve-evidence")
    def approve_evidence(sighting_id: str, request: EvidenceApprovalRequest) -> dict:
        """Approve exactly one left and one right ear candidate for matching."""
        _get_record(sighting_id)
        return decorate_record(
            workflow.approve_evidence(
                sighting_id, request.left_candidate_id, request.right_candidate_id
            )
        )

    @app.post("/sightings/{sighting_id}/match")
    def match_sighting(sighting_id: str, request: MatchRequest) -> dict:
        """Rank catalog elephants against an analyzed sighting."""
        _get_record(sighting_id)
        return decorate_record(
            workflow.match(sighting_id, _require_engine(), request.top_n)
        )

    @app.post("/sightings/{sighting_id}/decision")
    def decide_sighting(sighting_id: str, request: DecisionRequest) -> dict:
        """File the reviewer's identity decision for a sighting."""
        _get_record(sighting_id)
        return decorate_record(
            workflow.decide(
                sighting_id, _require_engine, request.action, request.elephant_name
            )
        )

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

    return app
