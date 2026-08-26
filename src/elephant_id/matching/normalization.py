"""Cohort normalization of pairwise match scores.

Raw overlap scores are biased per profile: a smooth ear with one generic bump
in the common tear zone matches everyone moderately well, while a feature-rich
ear matches everyone poorly. Normalizing each score by both profiles' cohort
statistics removes this bias, acting as an empirical distinctiveness weight
(adaptive score normalization, as used in speaker verification). This step is
fit-free and uses no identity labels.
"""

import numpy as np


def symmetrized_cohort_z(pairwise_scores: np.ndarray) -> np.ndarray:
    """Symmetrize a pairwise score matrix and z-normalize by cohort statistics.

    Matcher scores are directional because the query profile is shifted and
    stretched against the candidate. This first averages the two directions,
    then scores whether the pair is unusually strong for both profiles'
    same-side cohorts.

    Args:
        pairwise_scores: Square matrix of match scores. Entries for unscored
            pairs (for example different ear sides) must be NaN and stay NaN.

    Returns:
        `z[i, j] = (s[i, j] - mean_i) / std_i + (s[i, j] - mean_j) / std_j`
        where `s` is the symmetrized matrix and the statistics are computed
        over each row's and column's finite entries. The sum, rather than the
        average, fixes a score scale that downstream calibration absorbs.

    Raises:
        ValueError: If the input is not a square 2-D matrix.
    """
    scores = np.asarray(pairwise_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("pairwise_scores must be a square 2-D matrix")

    symmetric = (scores + scores.T) / 2.0
    row_mean = np.nanmean(symmetric, axis=1, keepdims=True)
    row_std = np.nanstd(symmetric, axis=1, keepdims=True) + 1e-9
    column_mean = np.nanmean(symmetric, axis=0, keepdims=True)
    column_std = np.nanstd(symmetric, axis=0, keepdims=True) + 1e-9
    return (symmetric - row_mean) / row_std + (symmetric - column_mean) / column_std
