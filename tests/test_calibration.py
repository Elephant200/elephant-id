"""Tests for tear-score calibration."""

import numpy as np
import pytest

from elephant_id.matching import (
    TearScoreCalibrator,
    TearScoreCalibratorConfig,
    tear_mass,
)


class TestTearMass:
    def test_single_profile_returns_one_value(self) -> None:
        profile = np.zeros(720)
        profile[100:110] = 0.04
        mass = tear_mass(profile)
        assert mass.shape == (1,)
        assert mass[0] == pytest.approx(10 * 0.04 * 180.0 / 720)

    def test_negative_depths_are_ignored(self) -> None:
        profile = np.full(720, -0.05)
        assert tear_mass(profile)[0] == 0.0

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError):
            tear_mass(np.zeros((0,)))


def synthetic_pairs(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return separable pairs where high mass makes scores more reliable."""
    rng = np.random.default_rng(seed)
    n = 600
    same = rng.random(n) < 0.5
    masses = rng.uniform(0.2, 8.0, size=(2, n))
    noise = rng.normal(0.0, 0.05, n)
    scores = np.where(same, 0.55, 0.42) + noise
    return scores, masses[0], masses[1], same


class TestTearScoreCalibrator:
    def test_config_validation(self) -> None:
        with pytest.raises(ValueError):
            TearScoreCalibratorConfig(l2_penalty=-1.0)
        with pytest.raises(ValueError):
            TearScoreCalibratorConfig(max_iterations=0)

    def test_requires_fit_before_calibrated_score(self) -> None:
        calibrator = TearScoreCalibrator()
        with pytest.raises(RuntimeError):
            calibrator.calibrated_score(np.array([0.5]), np.array([1.0]), np.array([1.0]))
        with pytest.raises(RuntimeError):
            _ = calibrator.weights

    def test_rejects_single_class_labels(self) -> None:
        calibrator = TearScoreCalibrator()
        with pytest.raises(ValueError, match="both classes"):
            calibrator.fit(
                np.array([0.4, 0.5]),
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
                np.array([True, True]),
            )

    def test_rejects_mismatched_lengths(self) -> None:
        calibrator = TearScoreCalibrator()
        with pytest.raises(ValueError):
            calibrator.fit(
                np.array([0.4, 0.5]),
                np.array([1.0]),
                np.array([1.0, 1.0]),
                np.array([True, False]),
            )

    def test_calibrated_score_orders_same_above_different(self) -> None:
        scores, query_masses, candidate_masses, same = synthetic_pairs()
        calibrator = TearScoreCalibrator()
        calibrator.fit(scores, query_masses, candidate_masses, same)
        calibrated_scores = calibrator.calibrated_score(
            scores,
            query_masses,
            candidate_masses,
        )
        assert calibrator.is_fitted
        assert calibrated_scores[same].mean() > calibrated_scores[~same].mean()

    def test_calibrated_score_increases_with_score(self) -> None:
        scores, query_masses, candidate_masses, same = synthetic_pairs()
        calibrator = TearScoreCalibrator()
        calibrator.fit(scores, query_masses, candidate_masses, same)
        masses = np.full(2, 2.0)
        low, high = calibrator.calibrated_score(np.array([0.3, 0.7]), masses, masses)
        assert high > low

    def test_convergence_warning_is_error(self) -> None:
        scores, query_masses, candidate_masses, same = synthetic_pairs()
        calibrator = TearScoreCalibrator(TearScoreCalibratorConfig(max_iterations=1))
        with pytest.raises(RuntimeError, match="did not converge"):
            calibrator.fit(scores, query_masses, candidate_masses, same)
