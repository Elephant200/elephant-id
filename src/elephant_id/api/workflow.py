"""Sighting workflow: analysis, evidence review, matching, and identity decisions.

This module hosts the business logic behind the sidecar HTTP routes so the
whole workflow can be exercised through one small interface with fake stores
and engines. Route handlers only parse requests, call these methods, and map
:class:`WorkflowInvalid` / :class:`WorkflowConflict` to HTTP status codes.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger

from elephant_id.api import figures, ingest
from elephant_id.api.analysis import (
    EAR_SIDES,
    analysis_payload,
    approved_profile_rows,
)
from elephant_id.api.engine import MatchingEngine
from elephant_id.api.gallery import GalleryData
from elephant_id.api.store import SightingStore

DECISION_ACTIONS = (
    "existing_known_elephant",
    "new_known_elephant",
    "unresolved",
    "confirm",
    "enroll",
)
DECISION_ACTION_ALIASES = {
    "confirm": "existing_known_elephant",
    "enroll": "new_known_elephant",
}


class WorkflowInvalid(ValueError):
    """The request contents are invalid for the workflow step (HTTP 400)."""


class WorkflowConflict(RuntimeError):
    """The sighting is not in a state that allows the operation (HTTP 409)."""


def canonical_decision_action(action: str) -> str:
    """Return the V1 decision action, accepting old preview aliases."""
    return DECISION_ACTION_ALIASES.get(action, action)


def _sighting_date(record: dict, fallback: str) -> str:
    """Derive the sighting date from the record's photos, not the filing time."""
    for photo in record.get("photos", []):
        date = photo.get("date")
        if date:
            return str(date)
    return fallback


class SightingWorkflow:
    """Evidence-first sighting workflow over one store and gallery.

    The evidence-review gate lives here: matching and identity decisions only
    ever see profiles that passed :func:`approved_profile_rows`.
    """

    def __init__(
        self,
        store: SightingStore,
        gallery: GalleryData,
        model_cache_root: Path,
    ) -> None:
        """Wire the workflow to its persistence and catalog dependencies."""
        self.store = store
        self.gallery = gallery
        self.model_cache_root = model_cache_root

    def working_profiles(
        self, sighting_id: str, record: dict
    ) -> tuple:
        """Return the approved profile rows for matching or enrollment.

        Raises:
            WorkflowConflict: If profiles are missing or approval is invalid.
        """
        try:
            profiles, sides, photo_ids, crop_paths = self.store.load_profiles(
                sighting_id
            )
        except FileNotFoundError as error:
            raise WorkflowConflict(
                "Sighting has no stored analysis package"
            ) from error
        try:
            return approved_profile_rows(record, profiles, sides, photo_ids, crop_paths)
        except ValueError as error:
            raise WorkflowConflict(str(error)) from error

    def run_analysis(self, sighting_id: str, folder: Path) -> None:
        """Background ingest: extract profiles and mark the sighting ready."""
        try:
            result = ingest.ingest_sighting(
                folder,
                self.store.sighting_dir(sighting_id),
                self.gallery,
                self.model_cache_root,
                progress=lambda processed, total: self.store.update(
                    sighting_id, progress={"processed": processed, "total": total}
                ),
            )
            self.store.save_profiles(
                sighting_id,
                result.profiles,
                result.sides,
                result.photo_ids,
                result.crop_paths,
            )
            self.store.save_row_geometry(sighting_id, result.row_geometry)
            self.store.update(
                sighting_id,
                status="ready",
                photos=[photo.to_dict() for photo in result.photos],
                profile_count=len(result.profiles),
                sides=sorted(set(result.sides)),
                approved_evidence=None,
                match=None,
                decision=None,
            )
        except Exception as error:
            logger.exception(f"Ingest failed for sighting {sighting_id}: {error}")
            self.store.update(sighting_id, status="failed", error=str(error))

    def analysis_package(self, sighting_id: str) -> dict:
        """Return the analysis package payload for evidence review."""
        record = self.store.get(sighting_id)
        try:
            profiles, sides, photo_ids, crop_paths = self.store.load_profiles(
                sighting_id
            )
        except FileNotFoundError:
            profiles, sides, photo_ids, crop_paths = None, (), (), ()
        if profiles is not None and len(profiles) > 0:
            updated_record = _ensure_analysis_profile_plots(
                record,
                profiles,
                sides,
                photo_ids,
                self.store.sighting_dir(sighting_id) / "profile_plots",
            )
            if updated_record["photos"] != record.get("photos", []):
                self.store.update(sighting_id, photos=updated_record["photos"])
            record = updated_record
        return analysis_payload(
            record,
            profiles,
            sides,
            photo_ids,
            crop_paths,
            row_geometry=self.store.load_row_geometry(sighting_id),
        )

    def approve_evidence(
        self, sighting_id: str, left_candidate_id: str, right_candidate_id: str
    ) -> dict:
        """Approve exactly one left and one right ear candidate.

        Raises:
            WorkflowConflict: If the sighting is not ready or has no package.
            WorkflowInvalid: If the selection is not one valid ear per side.
        """
        record = self.store.get(sighting_id)
        if record.get("status") != "ready":
            raise WorkflowConflict(
                f"Sighting is {record.get('status')}, not ready for evidence review"
            )
        try:
            profiles, sides, photo_ids, crop_paths = self.store.load_profiles(
                sighting_id
            )
        except FileNotFoundError as error:
            raise WorkflowConflict(
                "Sighting has no stored analysis package"
            ) from error
        package = analysis_payload(record, profiles, sides, photo_ids, crop_paths)
        candidate_lookup = {
            candidate["candidate_id"]: candidate
            for side_candidates in package["ear_candidates"].values()
            for candidate in side_candidates
        }
        selected = {
            "left": candidate_lookup.get(left_candidate_id),
            "right": candidate_lookup.get(right_candidate_id),
        }
        for side in EAR_SIDES:
            candidate = selected[side]
            if candidate is None or candidate["side"] != side:
                raise WorkflowInvalid(
                    "Evidence review requires one valid left and one valid right ear candidate"
                )
        approved = {
            "approved_at": datetime.now(UTC).isoformat(),
            "left": _approved_candidate(selected["left"]),
            "right": _approved_candidate(selected["right"]),
        }
        return self.store.update(
            sighting_id, approved_evidence=approved, match=None, decision=None
        )

    def match(self, sighting_id: str, engine: MatchingEngine, top_n: int) -> dict:
        """Rank catalog elephants against a sighting's approved evidence.

        Raises:
            WorkflowConflict: If the sighting is not ready or approval is invalid.
        """
        record = self.store.get(sighting_id)
        if record["status"] != "ready":
            raise WorkflowConflict(f"Sighting is {record['status']}, not ready")
        profiles, sides, photo_ids, _ = self.working_profiles(sighting_id, record)
        ranked = engine.rank(profiles, sides, photo_ids, top_n=top_n)
        candidates = [candidate.to_dict() for candidate in ranked]
        _render_match_profile_plots(
            candidates,
            self.store.sighting_dir(sighting_id) / "match_plots",
        )
        match = {
            "matched_at": datetime.now(UTC).isoformat(),
            "candidates": candidates,
        }
        return self.store.update(sighting_id, match=match)

    def decide(
        self,
        sighting_id: str,
        engine_supplier: Callable[[], MatchingEngine],
        action: str,
        elephant_name: str | None,
    ) -> dict:
        """File the reviewer's identity decision, enrolling when required.

        The supplier is only called for actions that need the engine, so
        unresolved decisions can be filed while the engine is still warming up.

        Raises:
            WorkflowConflict: If the sighting is already decided.
            WorkflowInvalid: If the action or elephant name is invalid.
        """
        record = self.store.get(sighting_id)
        if record.get("decision"):
            raise WorkflowConflict("Sighting already decided")
        action = canonical_decision_action(action)
        if action not in DECISION_ACTIONS:
            raise WorkflowInvalid(
                "Action must be existing_known_elephant, new_known_elephant, or unresolved"
            )
        decision = {
            "action": action,
            "elephant_name": None,
            "decided_at": datetime.now(UTC).isoformat(),
        }
        if action in ("existing_known_elephant", "new_known_elephant"):
            name = (elephant_name or "").strip()
            engine = engine_supplier()
            if not name:
                raise WorkflowInvalid("elephant_name is required")
            if action == "existing_known_elephant" and not engine.has_identity(name):
                raise WorkflowInvalid(f"Unknown elephant: {name}")
            if action == "new_known_elephant" and engine.has_identity(name):
                raise WorkflowInvalid(f"Elephant already exists: {name}")
            profiles, sides, photo_ids, crop_paths = self.working_profiles(
                sighting_id, record
            )
            engine.extend(
                profiles,
                sides,
                name,
                _sighting_date(record, decision["decided_at"][:10]),
                photo_ids,
                crop_paths,
            )
            decision["elephant_name"] = name
        return self.store.update(sighting_id, decision=decision)

    def refile_decisions(self, engine: MatchingEngine) -> None:
        """Re-apply confirmed and enrolled sightings from previous sessions."""
        for record in reversed(self.store.list()):
            decision = record.get("decision")
            action = (
                canonical_decision_action(decision["action"]) if decision else None
            )
            if action not in ("existing_known_elephant", "new_known_elephant"):
                continue
            try:
                profiles, sides, photo_ids, crop_paths = self.store.load_profiles(
                    record["sighting_id"]
                )
                if record.get("approved_evidence"):
                    profiles, sides, photo_ids, crop_paths = approved_profile_rows(
                        record, profiles, sides, photo_ids, crop_paths
                    )
                engine.extend(
                    profiles,
                    sides,
                    decision["elephant_name"],
                    _sighting_date(record, decision["decided_at"][:10]),
                    photo_ids,
                    crop_paths,
                )
            except Exception as error:
                logger.warning(
                    f"Could not refile sighting {record['sighting_id']}: {error}"
                )


def _approved_candidate(candidate: dict) -> dict:
    """Return the persisted reference for an approved ear candidate."""
    return {
        "candidate_id": candidate["candidate_id"],
        "profile_row_index": candidate["profile_row_index"],
        "side": candidate["side"],
        "photo_id": candidate["photo_id"],
        "file_name": candidate["file_name"],
        "crop_path": candidate["crop_path"],
        "display_crop_path": candidate.get("display_crop_path") or candidate["crop_path"],
        "photo_path": candidate.get("photo_path"),
        "corrected": False,
    }


def _render_match_profile_plots(candidates: list[dict], output_dir: Path) -> None:
    """Attach one server-rendered aligned-profile PNG to each match evidence item."""
    renderable: list[tuple[dict, np.ndarray, np.ndarray]] = []
    for candidate in candidates:
        for evidence in candidate.get("evidence", []):
            query = np.asarray(evidence.get("query_profile", ()), dtype=np.float64)
            catalog = np.asarray(evidence.get("gallery_profile", ()), dtype=np.float64)
            evidence["profile_plot_path"] = None
            if (
                query.ndim == 1
                and catalog.ndim == 1
                and len(query) > 1
                and query.shape == catalog.shape
            ):
                renderable.append((evidence, query, catalog))
    if not renderable:
        return

    y_max = figures.shared_profile_ymax(
        [(query, catalog) for _, query, catalog in renderable]
    )
    for candidate_index, candidate in enumerate(candidates, start=1):
        for evidence in candidate.get("evidence", []):
            if evidence.get("profile_plot_path") is not None:
                continue
            query = np.asarray(evidence.get("query_profile", ()), dtype=np.float64)
            catalog = np.asarray(evidence.get("gallery_profile", ()), dtype=np.float64)
            if (
                query.ndim != 1
                or catalog.ndim != 1
                or len(query) <= 1
                or query.shape != catalog.shape
            ):
                continue
            side = str(evidence["side"])
            output_path = output_dir / f"candidate-{candidate_index:02d}-{side}.png"
            figures.render_aligned_profiles_png(
                query,
                catalog,
                output_path,
                side=side,
                shift_degrees=float(evidence.get("alignment_shift_degrees", 0.0)),
                stretch=float(evidence.get("alignment_stretch", 1.0)),
                score=float(evidence["score"]),
                y_max=y_max,
            )
            evidence["profile_plot_path"] = str(output_path)


def _ensure_analysis_profile_plots(
    record: dict,
    profiles: np.ndarray,
    sides: tuple[str, ...],
    photo_ids: tuple[str, ...],
    output_dir: Path,
) -> dict:
    """Backfill missing server-rendered profile PNGs on stored photo evidence."""
    resolved_output_dir = output_dir.resolve()
    rows_by_photo_side: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
    for row_index, (profile, side, photo_id) in enumerate(
        zip(profiles, sides, photo_ids, strict=True)
    ):
        rows_by_photo_side.setdefault((photo_id, side), []).append(
            (row_index, np.asarray(profile, dtype=np.float64))
        )

    photos = []
    for photo in record.get("photos", []):
        photo_copy = dict(photo)
        ears = []
        for ear in photo.get("ears", []):
            ear_copy = dict(ear)
            side = str(ear_copy.get("side", ""))
            rows = rows_by_photo_side.get((str(photo.get("photo_id", "")), side), [])
            existing_path = ear_copy.get("profile_plot_path")
            resolved_existing_path = (
                Path(existing_path).expanduser().resolve() if existing_path else None
            )
            if (
                resolved_existing_path is not None
                and resolved_existing_path.is_file()
                and resolved_existing_path.is_relative_to(resolved_output_dir)
            ):
                if rows:
                    rows.pop(0)
                ears.append(ear_copy)
                continue
            if rows:
                row_index, profile = rows.pop(0)
                output_path = output_dir / f"row-{row_index:03d}-{side}.png"
                figures.render_tear_profile_png(profile, output_path, side=side)
                ear_copy["profile_plot_path"] = str(output_path)
            ears.append(ear_copy)
        photo_copy["ears"] = ears
        photos.append(photo_copy)

    updated = dict(record)
    updated["photos"] = photos
    return updated
