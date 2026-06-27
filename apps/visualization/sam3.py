"""Session-only SAM3 overlay runs for the visualization app."""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from typing import Any

import cv2

from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.visualize import visualize_predictions

from .analyzer import _json_value

SAM3_SCHEMA_VERSION = "sam3-overlay-v1"
SAM3_PRESETS = ("body", "features")


class Sam3UnavailableError(RuntimeError):
    """Raised when SAM3 cannot be initialized in this environment."""


class Sam3PhotoNotFoundError(KeyError):
    """Raised when a SAM3 request names a photo outside the dataset."""


class Sam3ResultNotFoundError(KeyError):
    """Raised when an in-memory SAM3 result no longer exists."""


@dataclass(frozen=True)
class Sam3Run:
    """A completed SAM3 overlay result retained for the app session."""

    run_id: str
    photo: Photo
    detections_by_preset: dict[str, list[Any]]
    summary: dict[str, Any]


def _summarize_sam3(photo: Photo, detections_by_preset: dict[str, list[Any]]) -> dict[str, Any]:
    """Create compact, JSON-safe evidence for a SAM3 overlay run."""
    class_counts: dict[str, int] = {}
    preset_counts = {}
    total = 0
    for preset, detections in detections_by_preset.items():
        preset_counts[preset] = len(detections)
        total += len(detections)
        for detection in detections:
            class_counts[detection.class_name] = class_counts.get(detection.class_name, 0) + 1

    return {
        "schemaVersion": SAM3_SCHEMA_VERSION,
        "identifier": photo.identifier,
        "presets": list(SAM3_PRESETS),
        "totalDetections": total,
        "presetCounts": preset_counts,
        "classCounts": class_counts,
        "detections": {
            preset: _json_value(detections)
            for preset, detections in detections_by_preset.items()
        },
    }


class Sam3Workbench:
    """Run SAM3 overlays one at a time and retain results in memory."""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._sam3: Any | None = None
        self._init_error: str | None = None
        self._runs: dict[str, Sam3Run] = {}
        self._init_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._runs_lock = threading.Lock()

    def run(self, identifier: str) -> Sam3Run:
        """Run the standard SAM3 overlay presets for one dataset photo."""
        try:
            photo = self._dataset.get_photo(identifier)
        except KeyError:
            raise Sam3PhotoNotFoundError(f"No photo with identifier: {identifier}") from None

        sam3 = self._get_sam3()
        detections_by_preset = {}
        with self._run_lock:
            try:
                for preset in SAM3_PRESETS:
                    detections_by_preset[preset] = sam3.run(photo, preset)
            except Exception as error:
                raise RuntimeError(f"SAM3 failed: {error}") from error

        run = Sam3Run(
            run_id=uuid.uuid4().hex,
            photo=photo,
            detections_by_preset=detections_by_preset,
            summary=_summarize_sam3(photo, detections_by_preset),
        )
        with self._runs_lock:
            self._runs[run.run_id] = run
        return run

    def get(self, run_id: str) -> Sam3Run:
        """Return one in-memory SAM3 run or raise when it is unavailable."""
        with self._runs_lock:
            run = self._runs.get(run_id)
        if run is None:
            raise Sam3ResultNotFoundError(run_id)
        return run

    def overlay_png(self, run_id: str) -> bytes:
        """Render a completed SAM3 run as a PNG overlay."""
        run = self.get(run_id)
        detections = [
            detection
            for preset in SAM3_PRESETS
            for detection in run.detections_by_preset.get(preset, [])
        ]
        image = self._dataset.read_image(run.photo)
        overlay = visualize_predictions(image, detections)
        ok, encoded = cv2.imencode(".png", overlay)
        if not ok:
            raise RuntimeError("Could not encode SAM3 overlay PNG")
        return encoded.tobytes()

    def _get_sam3(self) -> Any:
        """Construct the optional SAM3 service once, preserving clear errors."""
        with self._init_lock:
            if self._sam3 is not None:
                return self._sam3
            if self._init_error is not None:
                raise Sam3UnavailableError(self._init_error)

            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
            if not os.environ.get("ROBOFLOW_API_KEY", "").strip():
                self._init_error = (
                    "SAM3 is unavailable: set ROBOFLOW_API_KEY and run "
                    "`uv sync --group visualization --group local`."
                )
                raise Sam3UnavailableError(self._init_error)

            try:
                from elephant_id.ai.sam3 import Sam3Service

                self._sam3 = Sam3Service(dataset=self._dataset)
            except Exception as error:
                self._init_error = (
                    "SAM3 is unavailable. Install the local dependencies, "
                    f"check model configuration, and retry after restart: {error}"
                )
                raise Sam3UnavailableError(self._init_error) from error
            return self._sam3
