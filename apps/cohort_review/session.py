"""Durable review sessions for selecting sighting ear pairs."""

import copy
import csv
import hashlib
import json
import os
from collections import Counter
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from uuid import UUID

from elephant_id.dataset import PhotoStore
from elephant_id.domain import Photo

Side = Literal["left", "right"]
CandidateSide = Literal["left", "right", "unresolved"]


@dataclass(frozen=True, slots=True)
class Candidate:
    """One assessed ear crop available to a human reviewer."""

    crop_key: str
    photo_id: UUID
    sighting_id: UUID
    elephant: str
    side: CandidateSide
    score: int
    crop_box: tuple[int, int, int, int]
    crop_path: str
    original_path: str | None = None
    assessment_provenance: str | None = None
    available: bool = True

    def __post_init__(self) -> None:
        """Validate identity, score, side, and crop geometry."""
        if not self.crop_key:
            raise ValueError("crop_key must not be empty")
        if self.photo_id.version != 4 or self.sighting_id.version != 4:
            raise ValueError("Candidate IDs must be UUIDv4 values")
        if self.side not in ("left", "right", "unresolved"):
            raise ValueError("Candidate side must be left, right, or unresolved")
        if not 1 <= self.score <= 10:
            raise ValueError("Candidate score must be between 1 and 10")
        x1, y1, x2, y2 = self.crop_box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Candidate crop_box must have positive area")


@dataclass(frozen=True, slots=True)
class ReviewProgress:
    """Processed-sighting counts for the current elephant and the whole queue.

    An elephant's target is its usable sightings, limited to the sighting cap plus its
    rejections. It is complete when its processed count reaches that target.
    """

    elephant_processed: int
    elephant_accepted: int
    elephant_usable: int
    elephants_complete: int
    elephants_total: int
    sightings_processed: int
    sightings_target: int


@dataclass(frozen=True, slots=True)
class ReviewScreen:
    """The current side and its complete visible candidate inventory."""

    status: Literal["review", "complete"]
    sighting_id: UUID | None
    elephant: str | None
    side: Side | None
    tier: int | None
    progress: ReviewProgress
    candidates: tuple[Candidate, ...]
    selected_crop_key: str | None
    scroll_y: int
    can_forward: bool


class ReviewSession:
    """Validate inventory, schedule sightings, and persist review decisions."""

    _SCHEMA_VERSION = 1
    _SIGHTING_CAP = 10
    _MIN_USABLE_SIGHTINGS = 3

    def __init__(
        self,
        candidates: tuple[Candidate, ...],
        state_path: Path,
        manifest_path: Path,
        photos_by_elephant: dict[str, tuple[UUID, ...]],
        photo_store: PhotoStore | None = None,
    ) -> None:
        """Open or create a persistent review session.

        Args:
            candidates: Frozen assessed crop inventory, including scores below the
                display floor so validation cannot silently lose evidence.
            state_path: Authoritative local review-state JSON path.
            manifest_path: Complete approved-pair CSV projection path.
            photos_by_elephant: Every dataset photo grouped by known elephant.
            photo_store: Optional image-only access to original photo bytes.
        """
        self._state_path = state_path
        self._manifest_path = manifest_path
        self._photos_by_elephant = photos_by_elephant
        self._photo_store = photo_store
        self._candidates = self._validate_inventory(candidates)
        fingerprint_rows = [
            (c.crop_key, str(c.photo_id), str(c.sighting_id), c.elephant, c.side, c.score, c.crop_box)
            for c in sorted(candidates, key=lambda item: item.crop_key)
        ]
        self.inventory_fingerprint = hashlib.sha256(
            json.dumps(fingerprint_rows, separators=(",", ":")).encode()
        ).hexdigest()
        self._by_key = {candidate.crop_key: candidate for candidate in candidates}
        self._by_sighting: dict[UUID, dict[Side, tuple[Candidate, ...]]] = {}
        for candidate in candidates:
            if candidate.score < 5 or candidate.side not in ("left", "right"):
                continue
            sides = self._by_sighting.setdefault(candidate.sighting_id, {})
            existing = list(sides.get(candidate.side, ()))
            existing.append(candidate)
            sides[candidate.side] = tuple(
                sorted(existing, key=lambda item: (-item.score, item.crop_key))
            )
        tiers = {
            sighting_id: min(max(c.score for c in sides["left"]), max(c.score for c in sides["right"]))
            for sighting_id, sides in self._by_sighting.items()
            if "left" in sides
            and "right" in sides
            and max(c.score for c in sides["left"]) >= 8
            and max(c.score for c in sides["right"]) >= 8
        }
        usable = Counter(self._elephant(sighting_id) for sighting_id in tiers)
        self._usable_counts = {
            elephant: count for elephant, count in usable.items() if count >= self._MIN_USABLE_SIGHTINGS
        }
        self._tiers = {
            sighting_id: tier for sighting_id, tier in tiers.items() if self._elephant(sighting_id) in self._usable_counts
        }
        self._state = self._load_state()
        if not self._state["history"] and self._next_sighting() is not None:
            first = self._next_sighting()
            self._state["history"] = [self._location(first, "left")]
            self._state["history_index"] = 0
            self._persist(project=False)
        else:
            self._write_projections()

    @staticmethod
    def _validate_inventory(candidates: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
        """Require unique crop keys and consistent sighting ownership."""
        keys: set[str] = set()
        owners: dict[UUID, str] = {}
        photos: dict[UUID, UUID] = {}
        for candidate in candidates:
            if candidate.crop_key in keys:
                raise ValueError(f"Duplicate crop_key: {candidate.crop_key}")
            keys.add(candidate.crop_key)
            if owners.setdefault(candidate.sighting_id, candidate.elephant) != candidate.elephant:
                raise ValueError(f"Sighting has multiple elephants: {candidate.sighting_id}")
            if photos.setdefault(candidate.photo_id, candidate.sighting_id) != candidate.sighting_id:
                raise ValueError(f"Photo belongs to multiple sightings: {candidate.photo_id}")
        return candidates

    def _blank_state(self) -> dict[str, object]:
        """Return an empty serializable session state."""
        return {
            "schema_version": self._SCHEMA_VERSION,
            "inventory_fingerprint": self.inventory_fingerprint,
            "revision": 0,
            "drafts": {},
            "accepted": {},
            "rejected": {},
            "history": [],
            "history_index": -1,
            "scroll": {},
            "current_elephant_by_tier": {},
        }

    def _load_state(self) -> dict[str, object]:
        """Load and minimally validate an existing state file."""
        if not self._state_path.exists():
            return self._blank_state()
        state = json.loads(self._state_path.read_text())
        if state.get("schema_version") != self._SCHEMA_VERSION:
            raise ValueError("Unsupported review-state schema_version")
        if state.get("inventory_fingerprint") != self.inventory_fingerprint:
            raise ValueError("Review state does not match this candidate inventory")
        required = set(self._blank_state())
        if not required.issubset(state):
            raise ValueError("Review state is missing required fields")
        self._validate_state(state)
        return state

    def _validate_state(self, state: dict[str, object]) -> None:
        """Validate persisted decisions and navigation against the inventory."""
        for key, record in state["accepted"].items():
            self._validate_pair_record(key, record)
        for sighting_key, draft in state["drafts"].items():
            if not isinstance(draft, dict):
                raise ValueError("Draft must be an object")
            for side, crop_key in draft.items():
                candidate = self._by_key.get(crop_key)
                if (
                    side not in ("left", "right")
                    or candidate is None
                    or str(candidate.sighting_id) != sighting_key
                    or candidate.side != side
                ):
                    raise ValueError("Draft selection is inconsistent with inventory")
        for location in state["history"]:
            if not isinstance(location, dict) or location.get("side") not in ("left", "right"):
                raise ValueError("Review history contains an invalid location")
            try:
                sighting_id = UUID(location.get("sighting_id", ""))
            except (TypeError, ValueError):
                raise ValueError("Review history contains an invalid sighting ID") from None
            if sighting_id not in self._tiers:
                raise ValueError("Review history references an unknown sighting")
        index = state["history_index"]
        if not isinstance(index, int) or not -1 <= index <= len(state["history"]):
            raise ValueError("Review history index is invalid")

    def _validate_pair_record(self, key: str, record: object) -> None:
        """Validate one persisted pair against immutable inventory."""
        if not isinstance(record, dict):
            raise ValueError("Accepted pair record must be an object")
        left = self._by_key.get(record.get("left_crop_key"))
        right = self._by_key.get(record.get("right_crop_key"))
        if left is None or right is None:
            raise ValueError("Accepted pair references an unknown crop")
        if left.side != "left" or right.side != "right":
            raise ValueError("Accepted pair references the wrong ear side")
        if left.sighting_id != right.sighting_id:
            raise ValueError("Accepted pair crosses sightings")
        if key != record.get("sighting_id") or key != str(left.sighting_id):
            raise ValueError("Accepted pair sighting identity is inconsistent")
        if record.get("elephant") != left.elephant or left.elephant != right.elephant:
            raise ValueError("Accepted pair elephant identity is inconsistent")

    def _approved_counts(self) -> dict[str, int]:
        """Count distinct approved sightings for each elephant."""
        counts: dict[str, int] = {}
        for record in self._state["accepted"].values():
            elephant = record["elephant"]
            counts[elephant] = counts.get(elephant, 0) + 1
        return counts

    def _rejected_counts(self) -> Counter[str]:
        """Count rejected sightings for each elephant."""
        return Counter(record["elephant"] for record in self._state["rejected"].values())

    def _can_reach_minimum(self, elephant: str, rejected: Counter[str]) -> bool:
        """Return whether accepted plus undecided sightings can still reach the minimum."""
        return self._usable_counts[elephant] - rejected[elephant] >= self._MIN_USABLE_SIGHTINGS

    def _progress(self, elephant: str | None) -> ReviewProgress:
        """Count processed sightings against each elephant's completion target."""
        decided = Counter(
            record["elephant"] for record in (*self._state["accepted"].values(), *self._state["rejected"].values())
        )
        rejected = self._rejected_counts()

        def target(name: str) -> int:
            if not self._can_reach_minimum(name, rejected):
                return decided[name]
            return min(self._usable_counts[name], self._SIGHTING_CAP + rejected[name])

        return ReviewProgress(
            elephant_processed=decided[elephant] if elephant else 0,
            elephant_accepted=self._approved_counts().get(elephant, 0) if elephant else 0,
            elephant_usable=self._usable_counts[elephant] if elephant else 0,
            elephants_complete=sum(decided[name] >= target(name) for name in self._usable_counts),
            elephants_total=len(self._usable_counts),
            sightings_processed=sum(decided.values()),
            sightings_target=sum(target(name) for name in self._usable_counts),
        )

    def _elephant_priority(self, elephant: str, tier: int) -> tuple[object, ...]:
        """Return the specified rich-evidence scheduling priority."""
        proposed = [
            (value, self._stronger_score(sighting_id))
            for sighting_id, value in self._tiers.items()
            if self._elephant(sighting_id) == elephant
        ]
        proposed.sort(reverse=True)
        fifth = proposed[4] if len(proposed) >= 5 else (-1, -1)
        third = proposed[2] if len(proposed) >= 3 else (-1, -1)
        current_tier = sum(value == tier for value, _ in proposed)
        progress = min(self._approved_counts().get(elephant, 0), 3)
        return (-fifth[0], -fifth[1], -third[0], -third[1], -current_tier, -progress, elephant)

    def _stronger_score(self, sighting_id: UUID) -> int:
        """Return the better side's best score."""
        sides = self._by_sighting[sighting_id]
        return max(max(c.score for c in sides["left"]), max(c.score for c in sides["right"]))

    def _sighting_strength(self, sighting_id: UUID) -> float:
        """Rank distinct-photo alternatives with emphasis on the weaker side."""
        strengths = []
        weights = {10: 1.0, 9: 0.5, 8: 0.25, 7: 0.125}
        for side in ("left", "right"):
            photo_scores: dict[UUID, int] = {}
            for candidate in self._by_sighting[sighting_id][side]:
                if candidate.available:
                    photo_scores[candidate.photo_id] = max(
                        photo_scores.get(candidate.photo_id, 0), candidate.score
                    )
            strengths.append(sum(weights.get(score, 0.0) for score in photo_scores.values()))
        return 0.9 * min(strengths) + 0.1 * max(strengths)

    def _elephant(self, sighting_id: UUID) -> str:
        """Return the known elephant for a sighting."""
        return next(iter(self._by_sighting[sighting_id].values()))[0].elephant

    def _next_sighting(self) -> UUID | None:
        """Return the next unreviewed sighting in the global tier queue."""
        accepted = self._state["accepted"]
        rejected = self._state["rejected"]
        counts = self._approved_counts()
        rejected_counts = self._rejected_counts()
        current = self._state["current_elephant_by_tier"]
        for tier in (10, 9, 8):
            eligible = [
                sighting_id
                for sighting_id, proposed_tier in self._tiers.items()
                if proposed_tier == tier
                and str(sighting_id) not in accepted
                and str(sighting_id) not in rejected
                and counts.get(self._elephant(sighting_id), 0) < self._SIGHTING_CAP
                and self._can_reach_minimum(self._elephant(sighting_id), rejected_counts)
            ]
            if not eligible:
                continue
            elephants = {self._elephant(sighting_id) for sighting_id in eligible}
            active = current.get(str(tier))
            if active not in elephants:
                active = min(elephants, key=lambda name: self._elephant_priority(name, tier))
                current[str(tier)] = active
            choices = [sid for sid in eligible if self._elephant(sid) == active]
            return min(choices, key=lambda sid: (-self._sighting_strength(sid), str(sid)))
        return None

    @staticmethod
    def _location(sighting_id: UUID, side: Side) -> dict[str, str]:
        """Return a serializable review-screen location."""
        return {"sighting_id": str(sighting_id), "side": side}

    def screen(self) -> ReviewScreen:
        """Return the current screen, including history-only reviewed records."""
        history = self._state["history"]
        index = self._state["history_index"]
        if index < 0 or index >= len(history):
            return ReviewScreen("complete", None, None, None, None, self._progress(None), (), None, 0, False)
        location = history[index]
        sighting_id = UUID(location["sighting_id"])
        side = location["side"]
        selected = self._selected_for_side(sighting_id, side)
        scroll = self._state["scroll"].get(f"{sighting_id}:{side}", 0)
        return ReviewScreen(
            status="review",
            sighting_id=sighting_id,
            elephant=self._elephant(sighting_id),
            side=side,
            tier=self._tiers[sighting_id],
            progress=self._progress(self._elephant(sighting_id)),
            candidates=self._by_sighting[sighting_id][side],
            selected_crop_key=selected,
            scroll_y=scroll,
            can_forward=index + 1 < len(history) or self._is_decided(sighting_id),
        )

    def _is_decided(self, sighting_id: UUID) -> bool:
        """Return whether a sighting has an accepted pair or a rejection."""
        key = str(sighting_id)
        return key in self._state["accepted"] or key in self._state["rejected"]

    def _selected_for_side(self, sighting_id: UUID, side: Side) -> str | None:
        """Return a draft selection, falling back to the committed pair."""
        drafts = self._state["drafts"].get(str(sighting_id), {})
        if side in drafts:
            return drafts[side]
        record = self._state["accepted"].get(str(sighting_id))
        return record.get(f"{side}_crop_key") if record else None

    def select(self, crop_key: str) -> ReviewScreen:
        """Persist a side selection and advance only after the write succeeds."""
        snapshot = copy.deepcopy(self._state)
        screen = self.screen()
        if screen.status != "review":
            raise ValueError("Review is complete")
        candidate = self._by_key.get(crop_key)
        if candidate is None or candidate.sighting_id != screen.sighting_id or candidate.side != screen.side:
            raise ValueError("Candidate is not on the current review screen")
        if candidate.score < 5:
            raise ValueError("Candidate is below the display floor")
        if not candidate.available:
            raise ValueError("Candidate image is unavailable")
        sighting_key = str(screen.sighting_id)
        drafts = self._state["drafts"].setdefault(sighting_key, {})
        drafts[screen.side] = crop_key
        if screen.side == "left":
            self._open_right_side(screen.sighting_id)
            project = False
        else:
            left_key = drafts.get("left") or self._selected_for_side(screen.sighting_id, "left")
            if left_key is None:
                raise ValueError("Select a left ear before committing the pair")
            left = self._by_key[left_key]
            if left.sighting_id != candidate.sighting_id or left.side != "left":
                raise ValueError("Selected ears do not form a same-sighting pair")
            existing = self._state["accepted"].get(sighting_key)
            approved_count = self._approved_counts().get(candidate.elephant, 0)
            if existing is None and approved_count >= self._SIGHTING_CAP:
                raise ValueError(f"This elephant already has {self._SIGHTING_CAP} approved pairs; reopen an approval first")
            self._state["accepted"][sighting_key] = {
                "sighting_id": sighting_key,
                "elephant": candidate.elephant,
                "left_crop_key": left_key,
                "right_crop_key": crop_key,
            }
            self._state["rejected"].pop(sighting_key, None)
            self._state["drafts"].pop(sighting_key, None)
            self._advance_after_decision()
            project = True
        try:
            self._persist(project=project)
        except Exception:
            self._rollback(snapshot, project=project)
            raise
        return self.screen()

    def reject(self) -> ReviewScreen:
        """Reject the current sighting while retaining recoverable evidence."""
        snapshot = copy.deepcopy(self._state)
        screen = self.screen()
        if screen.status != "review":
            raise ValueError("Review is complete")
        self._record_rejection(screen.sighting_id)
        self._advance_after_decision()
        try:
            self._persist(project=True)
        except Exception:
            self._rollback(snapshot, project=True)
            raise
        return self.screen()

    def _record_rejection(self, sighting_id: UUID) -> None:
        """Reject a sighting, retaining any approved pair as superseded evidence."""
        key = str(sighting_id)
        if key in self._state["accepted"]:
            self._state.setdefault("superseded", {}).setdefault(key, []).append(self._state["accepted"].pop(key))
        self._state["rejected"][key] = {"sighting_id": key, "elephant": self._elephant(sighting_id)}

    def candidates(self, sighting_id: UUID, side: Side) -> tuple[Candidate, ...]:
        """Return a queued sighting's visible candidates for one side, best first."""
        if sighting_id not in self._tiers:
            raise KeyError(f"Sighting is not in the review queue: {sighting_id}")
        return self._by_sighting[sighting_id][side]

    def approved_pair(self, sighting_id: UUID) -> dict[str, str] | None:
        """Return a sighting's approved pair record, or `None` when it has none."""
        record = self._state["accepted"].get(str(sighting_id))
        return dict(record) if record else None

    def replace_pair(self, sighting_id: UUID, left_crop_key: str, right_crop_key: str) -> None:
        """Approve a new pair for a queued sighting outside the picker history.

        A different earlier pair is kept as superseded evidence, and an earlier
        rejection is withdrawn.
        """
        snapshot = copy.deepcopy(self._state)
        left = self._by_key.get(left_crop_key)
        right = self._by_key.get(right_crop_key)
        for candidate, side in ((left, "left"), (right, "right")):
            if candidate is None or candidate.sighting_id != sighting_id or candidate.side != side:
                raise ValueError(f"The {side} crop is not a {side} ear from this sighting")
            if candidate.score < 5 or not candidate.available:
                raise ValueError(f"The {side} crop is not selectable")
        key = str(sighting_id)
        existing = self._state["accepted"].get(key)
        if existing is None and self._approved_counts().get(left.elephant, 0) >= self._SIGHTING_CAP:
            raise ValueError(f"This elephant already has {self._SIGHTING_CAP} approved pairs; reopen an approval first")
        record = {
            "sighting_id": key,
            "elephant": left.elephant,
            "left_crop_key": left_crop_key,
            "right_crop_key": right_crop_key,
        }
        if existing is not None and existing != record:
            self._state.setdefault("superseded", {}).setdefault(key, []).append(existing)
        self._state["accepted"][key] = record
        self._state["rejected"].pop(key, None)
        self._state["drafts"].pop(key, None)
        try:
            self._persist(project=True)
        except Exception:
            self._rollback(snapshot, project=True)
            raise

    def reject_sighting(self, sighting_id: UUID) -> None:
        """Reject a queued sighting outside the picker history."""
        if sighting_id not in self._tiers:
            raise KeyError(f"Sighting is not in the review queue: {sighting_id}")
        snapshot = copy.deepcopy(self._state)
        self._record_rejection(sighting_id)
        try:
            self._persist(project=True)
        except Exception:
            self._rollback(snapshot, project=True)
            raise

    def _open_right_side(self, sighting_id: UUID) -> None:
        """Move to the sighting's right screen, inserting it when it is not next."""
        history = self._state["history"]
        index = self._state["history_index"] + 1
        location = self._location(sighting_id, "right")
        if index >= len(history) or history[index] != location:
            history.insert(index, location)
        self._state["history_index"] = index

    def _advance_after_decision(self) -> None:
        """Move to the next other sighting in history, or else the next queued sighting."""
        history = self._state["history"]
        index = self._state["history_index"]
        current = history[index]["sighting_id"]
        for later in range(index + 1, len(history)):
            if history[later]["sighting_id"] != current:
                self._state["history_index"] = later
                return
        next_sighting = self._next_sighting()
        if next_sighting is None:
            self._state["history_index"] = len(history)
        else:
            history.append(self._location(next_sighting, "left"))
            self._state["history_index"] = len(history) - 1

    def _navigate(self, index: int) -> ReviewScreen:
        """Persist a new history position without changing decisions."""
        snapshot = copy.deepcopy(self._state)
        self._state["history_index"] = index
        try:
            self._persist(project=False)
        except Exception:
            self._rollback(snapshot, project=False)
            raise
        return self.screen()

    def back(self) -> ReviewScreen:
        """Return to the previous screen without changing decisions."""
        return self._navigate(max(self._state["history_index"] - 1, 0))

    def forward(self) -> ReviewScreen:
        """Move to the next screen, or to the next queued sighting after the last decided one."""
        index = self._state["history_index"]
        if index + 1 < len(self._state["history"]):
            return self._navigate(index + 1)
        if not self.screen().can_forward:
            return self.screen()
        snapshot = copy.deepcopy(self._state)
        self._advance_after_decision()
        try:
            self._persist(project=False)
        except Exception:
            self._rollback(snapshot, project=False)
            raise
        return self.screen()

    def save_scroll(self, sighting_id: UUID, side: Side, scroll_y: int) -> None:
        """Persist the current screen's nonnegative vertical scroll offset."""
        screen = self.screen()
        if screen.status == "review" and screen.sighting_id == sighting_id and screen.side == side:
            snapshot = copy.deepcopy(self._state)
            self._state["scroll"][f"{screen.sighting_id}:{screen.side}"] = max(0, int(scroll_y))
            try:
                self._persist(project=False)
            except Exception:
                self._rollback(snapshot, project=False)
                raise

    def candidate(self, crop_key: str) -> Candidate | None:
        """Return the inventory candidate for a crop key, if any."""
        return self._by_key.get(crop_key)

    def read_original(self, candidate: Candidate) -> bytes:
        """Read a candidate's original photo bytes."""
        if self._photo_store is None:
            raise FileNotFoundError("Original photo access is unavailable")
        return self._photo_store.read(
            Photo(photo_id=candidate.photo_id, sighting_id=candidate.sighting_id)
        )

    def approved_candidate(self, crop_key: str) -> Candidate | None:
        """Return the candidate for a crop selected in an approved pair, if any."""
        for record in self._state["accepted"].values():
            if crop_key in (record["left_crop_key"], record["right_crop_key"]):
                return self._by_key[crop_key]
        return None

    def approved_pairs(self) -> list[dict[str, object]]:
        """Return every approved pair with crop provenance, in manifest order."""
        pairs = [self._pair_export(record) for record in self._state["accepted"].values()]
        return sorted(pairs, key=lambda pair: (pair["elephant"], pair["sighting_id"]))

    def export(self) -> dict[str, object]:
        """Return portable decisions, provenance, and protected elephant photos."""
        pairs = self.approved_pairs()
        selected = sorted({pair["elephant"] for pair in pairs})
        return {
            "schema_version": self._SCHEMA_VERSION,
            "inventory_fingerprint": self.inventory_fingerprint,
            "revision": self._state["revision"],
            "approved_pairs": pairs,
            "retrieval_selected_elephants": selected,
            "training_excluded_photo_ids": sorted(
                str(photo_id) for elephant in selected for photo_id in self._photos_by_elephant.get(elephant, ())
            ),
            "review_state": copy.deepcopy(self._state),
        }

    def restore(self, backup: object) -> ReviewScreen:
        """Validate and restore a portable backup without overwriting newer work."""
        if not isinstance(backup, dict) or backup.get("inventory_fingerprint") != self.inventory_fingerprint:
            raise ValueError("Backup does not match this candidate inventory")
        incoming = backup.get("review_state")
        if (
            not isinstance(incoming, dict)
            or incoming.get("schema_version") != self._SCHEMA_VERSION
            or incoming.get("inventory_fingerprint") != self.inventory_fingerprint
        ):
            raise ValueError("Backup review state is invalid")
        if not isinstance(incoming.get("revision"), int) or incoming["revision"] < self._state["revision"]:
            raise ValueError("Backup is older than the current review state")
        required = set(self._blank_state())
        if not required.issubset(incoming):
            raise ValueError("Backup review state is missing required fields")
        self._validate_state(incoming)
        snapshot = self._state
        self._state = copy.deepcopy(incoming)
        try:
            self._persist(project=True)
        except Exception:
            self._rollback(snapshot, project=True)
            raise
        return self.screen()

    def _rollback(self, snapshot: dict[str, object], *, project: bool) -> None:
        """Restore in-memory state and any projection written before a failed commit."""
        self._state = snapshot
        if project:
            # On the next open, authoritative state also rewrites both projections.
            with suppress(OSError):
                self._write_projections()

    def _pair_export(self, record: dict[str, str]) -> dict[str, object]:
        """Expand one accepted record with selected crop provenance."""
        left = self._by_key[record["left_crop_key"]]
        right = self._by_key[record["right_crop_key"]]
        return {
            "elephant": record["elephant"],
            "sighting_id": record["sighting_id"],
            "left_photo_id": str(left.photo_id),
            "right_photo_id": str(right.photo_id),
            "left_crop_key": left.crop_key,
            "right_crop_key": right.crop_key,
            "left_crop_box": list(left.crop_box),
            "right_crop_box": list(right.crop_box),
            "left_score": left.score,
            "right_score": right.score,
            "left_assessment_provenance": left.assessment_provenance,
            "right_assessment_provenance": right.assessment_provenance,
        }

    def _persist(self, *, project: bool) -> None:
        """Stage recoverable projections, then atomically commit authoritative state."""
        self._state["revision"] += 1
        if project:
            self._write_projections()
        self._atomic_json(self._state_path, self._state)

    def _write_projections(self) -> None:
        """Atomically rewrite the approved manifest and provenance exports."""
        export = self.export()
        rows = export["approved_pairs"]
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ("known_elephant_name", "sighting_id", "left_photo_id", "right_photo_id")
        with NamedTemporaryFile("w", dir=self._manifest_path.parent, delete=False, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                {
                    "known_elephant_name": row["elephant"],
                    "sighting_id": row["sighting_id"],
                    "left_photo_id": row["left_photo_id"],
                    "right_photo_id": row["right_photo_id"],
                }
                for row in rows
            )
            temp_path = Path(handle.name)
        os.replace(temp_path, self._manifest_path)
        self._atomic_json(self._manifest_path.with_suffix(".provenance.json"), export)

    @staticmethod
    def _atomic_json(path: Path, value: object) -> None:
        """Write JSON through an adjacent temporary file and atomic replacement."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)
        os.replace(temp_path, path)

    def as_dict(self) -> dict[str, object]:
        """Serialize the current screen for a thin local UI."""
        screen = self.screen()
        return {
            **asdict(screen),
            "sighting_id": str(screen.sighting_id) if screen.sighting_id else None,
            "candidates": [
                {**asdict(candidate), "photo_id": str(candidate.photo_id), "sighting_id": str(candidate.sighting_id)}
                for candidate in screen.candidates
            ],
        }
