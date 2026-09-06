"""Implementation-independent catalog-matching interface."""

from collections.abc import Mapping
from typing import Literal, NewType, Protocol
from uuid import UUID

from elephant_id.domain import Photo, SightingEarPair

CandidateKey = NewType("CandidateKey", UUID)
CandidateScores = Mapping[CandidateKey, float]


class CatalogMatcher(Protocol):
    """Match neutral sighting evidence against a candidate catalog."""

    def match(
        self,
        query: SightingEarPair,
        catalog: Mapping[CandidateKey, tuple[SightingEarPair, ...]],
    ) -> CandidateScores:
        """Return one similarity score per catalog candidate."""
        ...


class MatchingError(RuntimeError):
    """Report a failed declared side and domain stage.

    A shared source photo is prepared when its first declared side is analyzed,
    so a pre-resolution failure is attributed to that side.
    """

    def __init__(
        self,
        *,
        photo: Photo,
        side: Literal["left", "right"],
        stage: str,
        message: str,
    ) -> None:
        """Initialize one structured matching failure."""
        self.photo_id = photo.photo_id
        self.side = side
        self.stage = stage
        super().__init__(
            f"{stage.capitalize()} failed for {side} ear in photo {photo.photo_id}: {message}"
        )
