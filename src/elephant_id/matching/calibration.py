"""Tear-mass-conditioned calibration of match scores.

A raw overlap score is not comparable across ears: the same score is far
stronger evidence between two feature-rich ears than between two smooth ones.
The calibrator fits a logistic model on ``(score, tear mass)`` pair features.
The default evaluation passes cohort-normalized matcher scores; ablation runs
may pass raw symmetrized matcher scores to test whether normalization helps.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def tear_mass(profiles: np.ndarray) -> np.ndarray:
    """Return total positive tear depth per profile row, in depth-degrees.

    Accepts one profile or a 2-D batch; always returns a 1-D array. Higher
    mass means the ear has more visible tear signal, so the same match score
    tends to be more trustworthy.
    """
    rows = np.asarray(profiles, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] == 0:
        raise ValueError("profiles must be a non-empty 1-D or 2-D array")
    degrees_per_bin = 180.0 / rows.shape[1]
    return np.maximum(rows, 0.0).sum(axis=1) * degrees_per_bin


@dataclass(frozen=True)
class TearScoreCalibratorConfig:
    """Parameters for the sklearn logistic fit."""

    l2_penalty: float = 1e-3
    max_iterations: int = 100
    tolerance: float = 1e-9

    def __post_init__(self) -> None:
        """Validate calibrator parameters."""
        if self.l2_penalty < 0:
            raise ValueError("l2_penalty must be non-negative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


class TearScoreCalibrator:
    """Map selected match scores to calibrated scores conditioned on tear mass."""

    def __init__(self, config: TearScoreCalibratorConfig | None = None) -> None:
        """Create an unfitted calibrator with the default configuration."""
        self.config = config or TearScoreCalibratorConfig()
        self._model: LogisticRegression | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether ``fit`` has been called successfully."""
        return self._model is not None

    @property
    def weights(self) -> np.ndarray:
        """Fitted feature weights.

        Raises:
            RuntimeError: If the calibrator has not been fitted.
        """
        if self._model is None:
            raise RuntimeError("Calibrator has not been fitted")
        return np.concatenate([self._model.coef_[0], self._model.intercept_])

    def fit(
        self,
        scores: np.ndarray,
        query_masses: np.ndarray,
        candidate_masses: np.ndarray,
        same_identity: np.ndarray,
    ) -> None:
        """Fit logistic weights on labeled same/different pairs.

        Args:
            scores: Match scores per pair. The default evaluation passes
                cohort-normalized matcher scores here; ablations may pass raw
                symmetrized matcher scores.
            query_masses: Query tear mass per pair (see ``tear_mass``).
            candidate_masses: Candidate tear mass per pair.
            same_identity: Boolean label per pair.

        Raises:
            ValueError: If inputs are empty, mismatched, or single-class.
        """
        labels = np.asarray(same_identity, dtype=bool)
        features = self._features(scores, query_masses, candidate_masses)
        if len(features) != len(labels):
            raise ValueError("scores and same_identity must have the same length")
        if len(labels) == 0:
            raise ValueError("At least one labeled pair is required")
        if labels.min() == labels.max():
            raise ValueError("same_identity must contain both classes")

        model = LogisticRegression(
            C=self._inverse_regularization_strength(),
            fit_intercept=True,
            max_iter=self.config.max_iterations,
            solver="lbfgs",
            tol=self.config.tolerance,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            try:
                model.fit(features, labels)
            except ConvergenceWarning as error:
                raise RuntimeError("Calibration logistic regression did not converge") from error
        self._model = model

    def calibrated_score(
        self,
        scores: np.ndarray,
        query_masses: np.ndarray,
        candidate_masses: np.ndarray,
    ) -> np.ndarray:
        """Return calibrated evidence logits for score/mass pairs.

        The returned value is the logistic decision function. It is useful for
        ranking and side fusion, but it is not a probability.

        Raises:
            RuntimeError: If the calibrator has not been fitted.
        """
        if self._model is None:
            raise RuntimeError("Calibrator has not been fitted")
        return self._model.decision_function(
            self._features(scores, query_masses, candidate_masses)
        )

    def _features(
        self,
        scores: np.ndarray,
        query_masses: np.ndarray,
        candidate_masses: np.ndarray,
    ) -> np.ndarray:
        """Build the pair feature matrix: score, log mass, and their interaction."""
        score_column = np.atleast_1d(np.asarray(scores, dtype=np.float64))
        query_column = np.atleast_1d(np.asarray(query_masses, dtype=np.float64))
        candidate_column = np.atleast_1d(np.asarray(candidate_masses, dtype=np.float64))
        if not (len(score_column) == len(query_column) == len(candidate_column)):
            raise ValueError("scores and masses must have the same length")
        if (query_column < 0).any() or (candidate_column < 0).any():
            raise ValueError("tear masses must be non-negative")
        log_mass = np.log1p(np.sqrt(query_column * candidate_column))
        return np.column_stack(
            [
                score_column,
                log_mass,
                score_column * log_mass,
            ]
        )

    def _inverse_regularization_strength(self) -> float:
        """Return sklearn's inverse L2 regularization strength."""
        if self.config.l2_penalty == 0:
            return np.inf
        return 1.0 / self.config.l2_penalty
