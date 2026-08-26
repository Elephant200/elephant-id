"""Session-only orchestration and export helpers for photo analysis."""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any

import numpy as np

from elephant_id.dataset import Dataset
from elephant_id.domain import Photo

if TYPE_CHECKING:
    from elephant_id.coding.ears.tear_profile import TearProfile

logger = logging.getLogger(__name__)

ANALYSIS_SCHEMA_VERSION = "photo-analysis-v1"


class AnalyzerUnavailableError(RuntimeError):
    """Raised when the optional local analyzer cannot be initialized."""


class AnalyzerResultNotFoundError(KeyError):
    """Raised when an in-memory analyzer result no longer exists."""


class AnalyzerPhotoNotFoundError(KeyError):
    """Raised when an analyzer request names a photo outside the dataset."""


class NoUsableEvidenceError(RuntimeError):
    """Raised when a photo produces no usable analyzer evidence."""


@dataclass(frozen=True)
class AnalyzerRun:
    """A completed analyzer result retained for the lifetime of the app."""

    run_id: str
    photo: Photo
    analysis: dict[str, Any]
    normalized: dict[str, Any]
    summary: dict[str, Any]


def _json_value(value: Any) -> Any:
    """Convert analyzer values into JSON-safe primitives without NaN values."""
    from elephant_id.ai.detection import Detection
    from elephant_id.coding.ears.anchored_ear import AnchoredEar
    from elephant_id.coding.ears.tear_profile import TearProfile

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Detection):
        return _json_value(value.to_dict())
    if isinstance(value, AnchoredEar):
        return {
            "side": value.side,
            "xyxy": _json_value(value.xyxy),
            "area": _json_value(value.area),
            "anchorPoints": _json_value(value.anchor_points),
            "originalAnchorPoints": _json_value(value.original_anchor_points),
            "contour": _json_value(value.resampled_contour(1024)),
            "rleMask": _json_value(value.rle_mask),
        }
    if isinstance(value, TearProfile):
        return {
            "profile": _json_value(value.profile),
            "scale": _json_value(value.scale),
            "reference": _json_value(value.reference),
            "origins": _json_value(value.origins),
            "normals": _json_value(value.normals),
        }
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if value is Ellipsis:
        return None
    return {"unsupportedType": type(value).__name__, "repr": repr(value)}


def _profile_summary(profile: TearProfile) -> dict[str, Any]:
    """Return compact tear-profile values suitable for the modal."""
    angles = np.linspace(0.0, 180.0, len(profile.profile))
    finite = np.isfinite(profile.profile)
    if not finite.any():
        return {"bins": len(profile.profile), "maxDepth": None, "maxAngle": None}
    max_index = int(np.nanargmax(profile.profile))
    return {
        "bins": len(profile.profile),
        "maxDepth": _json_value(float(profile.profile[max_index])),
        "maxAngle": _json_value(float(angles[max_index])),
        "scale": _json_value(profile.scale),
    }


def summarize_analysis(photo: Photo, analysis: dict[str, Any]) -> dict[str, Any]:
    """Create compact, user-facing evidence from a full analyzer result."""
    shared = analysis["shared_data"]
    raw_ears = shared.get("raw_ears", shared.get("ears", []))
    ears = []
    for ear_data in analysis["ears"]:
        ear = ear_data["ear"]
        ears.append(
            {
                "side": ear.side,
                "xyxy": _json_value(ear.xyxy),
                "area": _json_value(ear.area),
                "anchorPoints": _json_value(ear.anchor_points),
                "tearProfile": _profile_summary(ear_data["tear_profile"]),
            }
        )
    return {
        "schemaVersion": ANALYSIS_SCHEMA_VERSION,
        "identifier": photo.identifier,
        "view": analysis["view"],
        "age": _json_value(analysis["age"]),
        "gender": _json_value(analysis["gender"]),
        "featureCounts": {
            "trunks": len(shared["trunks"]),
            "rawEars": len(raw_ears),
            "anchoredEars": len(analysis["ears"]),
            "tusks": len(analysis["tusks"]),
        },
        "tusks": _json_value(analysis["tusks"]),
        "ears": ears,
    }


class AnalyzerWorkbench:
    """Run one optional local photo analyzer at a time and retain its results."""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._analyzer: Any | None = None
        self._init_error: str | None = None
        self._runs: dict[str, AnalyzerRun] = {}
        self._init_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._render_lock = threading.Lock()
        self._runs_lock = threading.Lock()

    def run(self, identifier: str) -> AnalyzerRun:
        """Analyze one canonical dataset photo and keep the result in memory."""
        try:
            photo = self._dataset.get_photo(identifier)
        except KeyError:
            logger.error("No photo with identifier: %s", identifier)
            raise AnalyzerPhotoNotFoundError(
                f"No photo with identifier: {identifier}"
            ) from None

        analyzer = self._get_analyzer()
        with self._run_lock:
            try:
                analysis = analyzer.analyze(photo)
            except Exception as error:
                logger.exception("Full analyzer failed for %s", identifier)
                raise RuntimeError(f"Full analysis failed: {error}") from error
        if analysis is None:
            raise NoUsableEvidenceError("No usable body and feature evidence was found")

        normalized = {
            "schemaVersion": ANALYSIS_SCHEMA_VERSION,
            "identifier": photo.identifier,
            "analysis": _json_value(analysis),
        }
        run = AnalyzerRun(
            run_id=uuid.uuid4().hex,
            photo=photo,
            analysis=analysis,
            normalized=normalized,
            summary=summarize_analysis(photo, analysis),
        )
        with self._runs_lock:
            self._runs[run.run_id] = run
        return run

    def get(self, run_id: str) -> AnalyzerRun:
        """Return one in-memory run or raise when it is unavailable."""
        with self._runs_lock:
            run = self._runs.get(run_id)
        if run is None:
            raise AnalyzerResultNotFoundError(run_id)
        return run

    def dashboard_png(self, run_id: str) -> bytes:
        """Render a completed run as the full diagnostic dashboard PNG."""
        run = self.get(run_id)
        with self._render_lock:
            try:
                import matplotlib

                matplotlib.use("Agg", force=True)
                from .analyzer_render import dashboard_png

                return dashboard_png(
                    run.analysis,
                    run.photo.identifier,
                    self._dataset.read_image(run.photo),
                )
            except Exception as error:
                logger.exception("Could not render analyzer dashboard for %s", run.photo.identifier)
                raise RuntimeError(f"Dashboard rendering failed: {error}") from error

    def json_bytes(self, run_id: str) -> bytes:
        """Return the normalized JSON export for one completed run."""
        return json.dumps(self.get(run_id).normalized, indent=2, allow_nan=False).encode()

    def profile_npy(self, run_id: str, side: str) -> bytes:
        """Return one selected ear's raw tear profile in NumPy `.npy` format."""
        if side not in {"left", "right"}:
            raise ValueError("Ear side must be left or right")
        run = self.get(run_id)
        for ear_data in run.analysis["ears"]:
            if ear_data["ear_side"] == side:
                output = BytesIO()
                np.save(output, ear_data["tear_profile"].profile)
                return output.getvalue()
        raise KeyError(f"No {side} ear profile is available")

    def _get_analyzer(self) -> Any:
        """Construct the optional local analyzer once, preserving a clear error."""
        with self._init_lock:
            if self._analyzer is not None:
                return self._analyzer
            if self._init_error is not None:
                raise AnalyzerUnavailableError(self._init_error)

            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            if not os.environ.get("ROBOFLOW_API_KEY", "").strip():
                self._init_error = (
                    "Full analysis is unavailable: set ROBOFLOW_API_KEY and run "
                    "`uv sync --group visualization --group local`."
                )
                raise AnalyzerUnavailableError(self._init_error)

            try:
                from elephant_id.coding import PhotoAnalyzer

                self._analyzer = PhotoAnalyzer(dataset=self._dataset)
            except Exception as error:
                logger.exception("Could not initialize the full photo analyzer")
                self._init_error = (
                    "Full analysis is unavailable. Install the local dependencies, "
                    f"check model weights, and retry after restart: {error}"
                )
                raise AnalyzerUnavailableError(self._init_error) from error
            return self._analyzer
