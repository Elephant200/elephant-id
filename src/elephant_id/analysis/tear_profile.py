"""Reusable AlphaTear profile value."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


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
