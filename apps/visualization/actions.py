"""Typed undo-stack actions.

Each action records exactly the information needed to undo a single user-
initiated mutation. Action objects are immutable; the dispatcher in
``state.py`` pattern-matches on the concrete subtype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import SightingKey


@dataclass(frozen=True)
class PriorityToggle:
    """A priority star was toggled on a single image within a sighting folder."""

    key: SightingKey
    queue_index: int
    folder_rel: str
    created_folder: bool
    from_basename: str
    to_basename: str

    @property
    def affected_basenames(self) -> tuple[str, ...]:
        # ``from`` and ``to`` differ only by the priority prefix, so any one
        # of them stripped gives the affected plain basename.
        from .samples import plain_basename
        return (plain_basename(self.from_basename),)


@dataclass(frozen=True)
class SavedRemoveSighting:
    """A saved sighting folder was removed from disk."""

    saved_rel: str
    affected_priority_basenames: tuple[str, ...]


Action = PriorityToggle | SavedRemoveSighting
