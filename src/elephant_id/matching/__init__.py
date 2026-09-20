"""Implementation-independent catalog matching contract."""

from elephant_id.matching.protocol import (
    CandidateKey,
    CandidateScores,
    CatalogMatcher,
    MatchingError,
)

__all__ = ["CandidateKey", "CandidateScores", "CatalogMatcher", "MatchingError"]
