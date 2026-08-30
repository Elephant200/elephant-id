"""Public catalog-matching interface for elephant re-identification."""

from elephant_id.matching.alphaphant import AlphaPhant
from elephant_id.matching.protocol import (
    CandidateKey,
    CandidateScores,
    CatalogMatcher,
)

__all__ = [
    "AlphaPhant",
    "CandidateKey",
    "CandidateScores",
    "CatalogMatcher",
]
