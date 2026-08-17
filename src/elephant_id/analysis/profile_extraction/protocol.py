"""Semantic tear-profile extraction protocol."""

from typing import Protocol

from elephant_id.analysis.ear_preparation import PreparedEar
from elephant_id.analysis.tear_profile import TearProfile


class TearProfileExtractor(Protocol):
    """Extract a reusable tear profile from one prepared ear."""

    @property
    def producer_slug(self) -> str | None:
        """Return the settled slug, or none for an experimental extractor."""
        ...

    def extract(self, ear: PreparedEar) -> TearProfile:
        """Extract one reusable tear profile."""
        ...
