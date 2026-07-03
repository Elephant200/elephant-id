"""Tests for cohort score normalization."""

import numpy as np
import pytest

from elephant_id.matching import symmetrized_cohort_z


class TestSymmetrizedCohortZ:
    def test_rejects_non_square_input(self) -> None:
        with pytest.raises(ValueError, match="square"):
            symmetrized_cohort_z(np.zeros((3, 4)))
        with pytest.raises(ValueError, match="square"):
            symmetrized_cohort_z(np.zeros(5))

    def test_output_is_symmetric(self) -> None:
        rng = np.random.default_rng(0)
        scores = rng.random((6, 6))
        normalized = symmetrized_cohort_z(scores)
        assert np.allclose(normalized, normalized.T)

    def test_preserves_nan_structure(self) -> None:
        scores = np.array(
            [
                [np.nan, 0.5, np.nan],
                [0.4, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        )
        normalized = symmetrized_cohort_z(scores)
        assert np.isfinite(normalized[0, 1])
        assert np.isnan(normalized[0, 2])
        assert np.isnan(normalized[2, 2])

    def test_discounts_promiscuous_rows(self) -> None:
        # Row 0 scores high against everyone; rows 1 and 2 score high only
        # with each other.
        scores = np.array(
            [
                [np.nan, 0.5, 0.5, 0.5],
                [0.5, np.nan, 0.5, 0.1],
                [0.5, 0.5, np.nan, 0.1],
                [0.5, 0.1, 0.1, np.nan],
            ]
        )
        normalized = symmetrized_cohort_z(scores)
        # The same raw 0.5 is stronger evidence between selective rows 1 and 2
        # than between promiscuous row 0 and row 1.
        assert normalized[1, 2] > normalized[0, 1]
