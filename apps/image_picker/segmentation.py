"""Live overlap check against the separate segmentation-annotation batch.

A sighting is flagged for reviewer awareness when any of its photos already
appears in the segmentation annotation batch folder. The flag is derived from
that folder's `{photo_identifier}_{side}.jpg` naming convention and is never
persisted in the manifest, so it can never drift out of sync with what is
actually in the batch (spec user stories 8 and 17).
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from elephant_id.domain import Sighting

from .config import SEGMENTATION_BATCH_ROOT, SIDES

# Batch crops are named `{photo_identifier}_{side}.jpg`; the photo identifier
# is recovered by stripping the trailing side suffix.
_BATCH_STEM_RE = re.compile(rf"^(?P<photo>.+)_(?:{'|'.join(SIDES)})$")


class SegmentationBatch:
    """In-memory index of dataset photos present in the segmentation batch.

    The batch folder is listed once, on first use, and the resulting set of
    photo identifiers is cached for the process. This is derived state, not
    persisted state: it is rebuilt from the folder every time the app starts.
    """

    def __init__(self, root: Path = SEGMENTATION_BATCH_ROOT) -> None:
        """Point at the batch folder without reading it yet."""
        self.root = root
        self._photo_identifiers: frozenset[str] | None = None

    def photo_identifiers(self) -> frozenset[str]:
        """Return the photo identifiers present in the batch, listing once."""
        if self._photo_identifiers is None:
            self._photo_identifiers = self._scan()
        return self._photo_identifiers

    def overlaps(self, sighting: Sighting) -> bool:
        """Whether any of a sighting's photos appears in the batch."""
        present = self.photo_identifiers()
        return any(photo.identifier in present for photo in sighting.photos)

    def _scan(self) -> frozenset[str]:
        """List the batch folder and recover each crop's photo identifier."""
        if not self.root.is_dir():
            logger.warning(f"Segmentation batch folder not found: {self.root}")
            return frozenset()
        identifiers: set[str] = set()
        for path in self.root.glob("*.jpg"):
            match = _BATCH_STEM_RE.match(path.stem)
            if match is None:
                logger.debug(f"Ignoring non-conforming batch file: {path.name}")
                continue
            identifiers.add(match.group("photo"))
        logger.info(
            f"Segmentation batch: {len(identifiers)} photos indexed from {self.root}"
        )
        return frozenset(identifiers)
