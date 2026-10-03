"""Audit approved sighting ear pairs, then re-pick or reject the flagged ones."""

import copy
import json
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal
from uuid import UUID

from .session import ReviewSession, Side

Resolution = Literal["repicked", "rejected"]


class PairAudit:
    """Step through approved pairs, flag bad ones, and queue flagged sightings for replacement.

    A flag records the pair it was raised against. It stays pending until the
    replace queue re-picks or rejects its sighting. Flags, the audit position,
    and the replace position persist in their own JSON file; pair decisions are
    committed through the `ReviewSession`, which rewrites the approved manifest.
    """

    _SCHEMA_VERSION = 1

    def __init__(self, session: ReviewSession, path: Path) -> None:
        """Open or create the audit file for a review session.

        Args:
            session: Review session that owns approved pairs and their manifest.
            path: Audit JSON path, conventionally `audit.json` beside `state.json`.
        """
        self._session = session
        self._path = path
        self._state = self._load()

    def _load(self) -> dict[str, object]:
        """Load an existing audit file or return an empty one."""
        if not self._path.exists():
            return {
                "schema_version": self._SCHEMA_VERSION,
                "position": None,
                "flags": {},
                "replace_position": None,
                "replace_drafts": {},
            }
        state = json.loads(self._path.read_text())
        if state.get("schema_version") != self._SCHEMA_VERSION:
            raise ValueError("Unsupported audit schema_version")
        return state

    # Audit mode

    def _audit_index(self, pairs: list[dict[str, object]]) -> int:
        """Return the saved position, or the next pair after it when it has left the list."""
        position = self._state["position"]
        if position is None:
            return 0
        saved = (position["elephant"], position["sighting_id"])
        for index, pair in enumerate(pairs):
            if (pair["elephant"], pair["sighting_id"]) >= saved:
                return index
        return len(pairs) - 1

    def audit_screen(self) -> dict[str, object]:
        """Return the current approved pair with its flag status and neighbors to prefetch."""
        pairs = self._session.approved_pairs()
        flags = self._state["flags"]
        pending = sum(flag["resolution"] is None for flag in flags.values())
        counts = {"total": len(pairs), "flagged": pending, "resolved": len(flags) - pending}
        if not pairs:
            return {"pair": None, "index": 0, "prefetch": [], **counts}
        index = self._audit_index(pairs)
        pair = pairs[index]
        flag = flags.get(pair["sighting_id"])
        status = None
        if flag is not None:
            current = (flag["left_crop_key"], flag["right_crop_key"]) == (pair["left_crop_key"], pair["right_crop_key"])
            status = "flagged" if flag["resolution"] is None and current else flag["resolution"]
        neighbors = [pairs[i] for i in (index + 1, index - 1) if 0 <= i < len(pairs)]
        return {
            "pair": pair,
            "index": index,
            "status": status,
            "reason": flag.get("reason") if flag else None,
            "prefetch": [p[f"{side}_crop_key"] for p in neighbors for side in ("left", "right")],
            **counts,
        }

    def move(self, step: int) -> dict[str, object]:
        """Move the audit position by `step` pairs, clamped to the list, and persist it."""
        pairs = self._session.approved_pairs()
        if pairs:
            pair = pairs[min(max(self._audit_index(pairs) + step, 0), len(pairs) - 1)]
            self._commit(lambda state: state.update(position={"elephant": pair["elephant"], "sighting_id": pair["sighting_id"]}))
        return self.audit_screen()

    def toggle_flag(self) -> dict[str, object]:
        """Flag the current pair, or unflag it when it is flagged and not yet replaced."""
        pair = self.audit_screen()["pair"]
        if pair is None:
            raise ValueError("There are no approved pairs to flag")
        self._commit(lambda state: self._toggle(state, pair))
        return self.audit_screen()

    @staticmethod
    def _toggle(state: dict[str, object], pair: dict[str, object]) -> None:
        """Toggle a pending flag on a pair inside a mutable audit state."""
        key = pair["sighting_id"]
        flag = state["flags"].get(key)
        current = flag is not None and (flag["left_crop_key"], flag["right_crop_key"]) == (
            pair["left_crop_key"],
            pair["right_crop_key"],
        )
        if current and flag["resolution"] is None:
            del state["flags"][key]
            state["replace_drafts"].pop(key, None)
        else:
            state["flags"][key] = {
                "elephant": pair["elephant"],
                "left_crop_key": pair["left_crop_key"],
                "right_crop_key": pair["right_crop_key"],
                "resolution": None,
            }

    def flag(self, sighting_id: UUID, reason: str | None = None) -> None:
        """Flag an approved sighting's current pair, with an optional reason for the replace queue."""
        pair = self._session.approved_pair(sighting_id)
        if pair is None:
            raise KeyError(f"Sighting has no approved pair: {sighting_id}")
        record = {
            "elephant": pair["elephant"],
            "left_crop_key": pair["left_crop_key"],
            "right_crop_key": pair["right_crop_key"],
            "resolution": None,
            "reason": reason,
        }

        def apply(state: dict[str, object]) -> None:
            state["flags"][str(sighting_id)] = record

        self._commit(apply)

    # Replace mode

    def _locations(self) -> list[tuple[str, Side]]:
        """Return every flagged sighting's left then right screen, in manifest order."""
        flags = self._state["flags"]
        ordered = sorted(flags, key=lambda key: (flags[key]["elephant"], key))
        return [(key, side) for key in ordered for side in ("left", "right")]

    def _replace_index(self) -> int | None:
        """Return the current replace screen index, or `None` when every flag is resolved."""
        locations = self._locations()
        position = self._state["replace_position"]
        if position is not None and (position["sighting_id"], position["side"]) in locations:
            return locations.index((position["sighting_id"], position["side"]))
        flags = self._state["flags"]
        return next(
            (index for index, (key, side) in enumerate(locations) if side == "left" and flags[key]["resolution"] is None),
            None,
        )

    def replace_screen(self) -> dict[str, object]:
        """Return the current replace screen in the picker's screen shape."""
        flags = self._state["flags"]
        progress = {
            "resolved": sum(flag["resolution"] is not None for flag in flags.values()),
            "flagged": len(flags),
        }
        index = self._replace_index()
        if index is None:
            return {"status": "complete", "mode": "replace", "progress": progress, "candidates": [], "can_forward": False}
        key, side = self._locations()[index]
        sighting_id = UUID(key)
        flag = flags[key]
        approved = self._session.approved_pair(sighting_id)
        selected = self._state["replace_drafts"].get(key) if side == "left" else None
        if selected is None and approved is not None:
            selected = approved[f"{side}_crop_key"]
        return {
            "status": "review",
            "mode": "replace",
            "sighting_id": key,
            "elephant": flag["elephant"],
            "side": side,
            "resolution": flag["resolution"],
            "reason": flag.get("reason"),
            "progress": progress,
            "candidates": list(self._session.candidates(sighting_id, side)),
            "selected_crop_key": selected,
            "scroll_y": 0,
            "can_forward": index + 1 < len(self._locations()),
        }

    def replace_select(self, crop_key: str) -> dict[str, object]:
        """Choose a crop on the current replace screen; the right choice commits the new pair."""
        screen = self.replace_screen()
        if screen["status"] != "review":
            raise ValueError("Every flagged sighting is resolved")
        key, side = screen["sighting_id"], screen["side"]
        if crop_key not in {candidate.crop_key for candidate in screen["candidates"]}:
            raise ValueError("Candidate is not on the current replace screen")
        if side == "left":

            def choose_left(state: dict[str, object]) -> None:
                state["replace_drafts"][key] = crop_key
                state["replace_position"] = {"sighting_id": key, "side": "right"}

            self._commit(choose_left)
            return self.replace_screen()
        approved = self._session.approved_pair(UUID(key)) or {}
        left_key = self._state["replace_drafts"].get(key) or approved.get("left_crop_key")
        if left_key is None:
            raise ValueError("Select a left ear before committing the pair")
        self._session.replace_pair(UUID(key), left_key, crop_key)
        self._resolve(key, "repicked")
        return self.replace_screen()

    def replace_reject(self) -> dict[str, object]:
        """Reject the current flagged sighting and move to the next pending one."""
        screen = self.replace_screen()
        if screen["status"] != "review":
            raise ValueError("Every flagged sighting is resolved")
        self._session.reject_sighting(UUID(screen["sighting_id"]))
        self._resolve(screen["sighting_id"], "rejected")
        return self.replace_screen()

    def _resolve(self, key: str, resolution: Resolution) -> None:
        """Record a committed decision and return to the first pending flag."""

        def apply(state: dict[str, object]) -> None:
            state["flags"][key]["resolution"] = resolution
            state["replace_drafts"].pop(key, None)
            state["replace_position"] = None

        self._commit(apply)

    def replace_step(self, step: int) -> dict[str, object]:
        """Move between replace screens without changing decisions."""
        index = self._replace_index()
        locations = self._locations()
        if index is None:
            index = len(locations)
        target = min(max(index + step, 0), len(locations) - 1)
        if locations and target != index:
            key, side = locations[target]
            self._commit(lambda state: state.update(replace_position={"sighting_id": key, "side": side}))
        return self.replace_screen()

    def visible_crop_keys(self) -> set[str]:
        """Return crop keys on the current replace screen and in flagged pairs."""
        keys = {candidate.crop_key for candidate in self.replace_screen()["candidates"]}
        for flag in self._state["flags"].values():
            keys.update((flag["left_crop_key"], flag["right_crop_key"]))
        return keys

    # Persistence

    def _commit(self, change: Callable[[dict[str, object]], None]) -> None:
        """Apply `change` to a copy of the state and atomically persist it before adopting it."""
        state = copy.deepcopy(self._state)
        change(state)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", dir=self._path.parent, delete=False) as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)
        os.replace(temp_path, self._path)
        self._state = state
