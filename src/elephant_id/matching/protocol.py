"""Implementation-independent catalog-matching interface."""

from collections.abc import Mapping
from typing import NewType, Protocol
from uuid import UUID

from elephant_id.domain import SightingEarPair

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
