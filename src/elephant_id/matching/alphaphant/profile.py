"""Reusable AlphaTear profile value."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from elephant_id.domain import Photo
from elephant_id.image.boxes import BoundingBox
from elephant_id.matching.protocol import MatchingError
from elephant_id.preparation.ear import EarSide, PreparedEar


@dataclass(frozen=True, slots=True, eq=False)
class TearProfile:
    """Immutable normalized tear depths sampled along one ear contour."""

    depths: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Copy and validate the one-dimensional normalized depths."""
        depths = np.array(self.depths, dtype=np.float64, copy=True)
        if depths.ndim != 1 or len(depths) == 0:
            raise ValueError("Tear-profile depths must be a non-empty 1-D array")
        if not np.isfinite(depths).all():
            raise ValueError("Tear-profile depths must be finite")
        depths.setflags(write=False)
        object.__setattr__(self, "depths", depths)


class TearProfileExtractor(Protocol):
    """Extract a reusable tear profile from one prepared ear."""

    @property
    def producer_slug(self) -> str | None:
        """Return the settled slug, or none for an experimental extractor."""
        ...

    def extract(self, ear: PreparedEar) -> TearProfile:
        """Extract one reusable tear profile."""
        ...


@dataclass(frozen=True, slots=True)
class EarProfile:
    """One ear tear profile with source photo and box required for reproduction."""

    source_photo: Photo
    side: EarSide
    source_box: BoundingBox
    tear_profile: TearProfile


@dataclass(frozen=True, slots=True)
class SightingProfiles:
    """Left- and right-ear analysis for one sighting."""

    left: EarProfile
    right: EarProfile


def extract_sighting_profiles(
    ears: tuple[PreparedEar, PreparedEar], extractor: TearProfileExtractor
) -> SightingProfiles:
    """Extract source-labelled profiles from resolved left and right ears.

    Raises:
        MatchingError: If extraction fails, with photo, side, and original cause.
    """
    profiles = []
    for ear, side in zip(ears, ("left", "right"), strict=True):
        try:
            profile = extractor.extract(ear)
        except Exception as error:
            raise MatchingError(
                photo=ear.source_photo,
                side=side,
                stage="tear-profile extraction",
                message=str(error),
            ) from error
        profiles.append(EarProfile(ear.source_photo, side, ear.source_box, profile))
    return SightingProfiles(*profiles)
