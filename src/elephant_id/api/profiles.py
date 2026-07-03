"""Shared helpers for shipping tear profiles to the review UI."""

import numpy as np

PROFILE_PLOT_BINS = 240


def plot_profile(profile: np.ndarray) -> tuple[float, ...]:
    """Downsample a tear profile to a compact, plottable series."""
    values = np.asarray(profile, dtype=np.float64)
    step = max(1, len(values) // PROFILE_PLOT_BINS)
    return tuple(round(float(value), 4) for value in values[::step])
