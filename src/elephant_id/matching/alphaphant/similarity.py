"""Compute ear similarities and side scores between a query ear and reference ears."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial import distance as scipy_distance
from scipy.special import logsumexp

from elephant_id.matching.alphaphant.extraction import AnalyzedEar, TearProfile


@dataclass(frozen=True, slots=True)
class EarComparatorConfig:
    """Settings for the comparison grid, the alignments, the shift and stretch penalties, the weights, and the side score.

    The gradient channels have the weight `gradient_weight`. The depth channels
    have the weight `1 - gradient_weight`. The angle weight is a Gaussian of
    the grid angle, with its center and its width (standard deviation) in
    degrees. The shift penalty is
    `exp(-(|shift_bins| / resampled_bins / shift_penalty_scale) ** shift_penalty_power)`.
    The stretch penalty is
    `exp(-(|ln stretch| / stretch_penalty_scale) ** stretch_penalty_power)`.
    The side score is the log-mean-exp of the ear similarities, with
    temperature `temperature`.
    """

    resampled_bins: int = 240
    gradient_weight: float = 0.5
    angle_weight_center_degrees: float = 120.0
    angle_weight_width_degrees: float = 35.0
    max_shift_fraction: float = 0.10
    shift_penalty_scale: float = 0.16
    shift_penalty_power: float = 4.0
    stretches: tuple[float, ...] = tuple(1.2 ** (index / 8) for index in range(-8, 9))
    stretch_penalty_scale: float = 0.32
    stretch_penalty_power: float = 4.0
    temperature: float = 0.05


@dataclass(frozen=True, slots=True)
class EarSimilarity:
    """An ear similarity and the query alignment that gives it.

    A positive shift moves the query to larger angles. An ear similarity of
    zero has stretch 1 and shift 0.
    """

    score: float
    stretch: float
    shift_bins: int


class EarComparator:
    """Compute the ear similarity of a query ear with each reference ear, and the side score over them."""

    def __init__(self, config: EarComparatorConfig | None = None) -> None:
        """Set the comparison settings; use the defaults if no config is supplied."""
        self._config = config or EarComparatorConfig()
        max_shift_bins = round(self._config.max_shift_fraction * self._config.resampled_bins)
        self._alignments = [
            (stretch, shift_bins)
            for stretch in self._config.stretches
            for shift_bins in range(-max_shift_bins, max_shift_bins + 1)
        ]
        self._alignment_penalties = np.array(
            [
                self._compute_shift_penalty(shift_bins) * self._compute_stretch_penalty(stretch)
                for stretch, shift_bins in self._alignments
            ]
        )
        # A stretch by t divides the gradient by t, and the square root turns 1/t into t ** -0.5.
        self._gradient_stretch_factors = np.array([stretch**-0.5 for stretch, _ in self._alignments])
        grid_angles = np.linspace(0.0, 180.0, self._config.resampled_bins)
        self._angle_weights = np.exp(
            -0.5
            * ((grid_angles - self._config.angle_weight_center_degrees) / self._config.angle_weight_width_degrees)
            ** 2
        )

    def compare_ears(
        self,
        query_ear: AnalyzedEar,
        reference_ears: Sequence[AnalyzedEar],
    ) -> tuple[EarSimilarity, ...]:
        """Return the ear similarity of the query ear with each reference ear.

        For each alignment, the comparator does these steps:

        1. Calculate the mean weighted real-valued Jaccard of the depth channels,
           and the mean weighted real-valued Jaccard of the gradient channels.
        2. Add the two means, each multiplied by its channel weight.
        3. Multiply the sum by the shift penalty and the stretch penalty.

        The ear similarity is the highest result.

        Returns:
            One ear similarity for each reference ear, in the same order.

        Raises:
            ValueError: If an ear does not have one tear profile for each alpha
                scale of the query, or if the tear profiles have different lengths.
        """
        if not _has_same_scale_count(query_ear, reference_ears):
            raise ValueError("Every ear needs one tear profile for each alpha scale")
        if not _has_same_length(query_ear, reference_ears):
            raise ValueError("Compared tear profiles must have the same length")
        if not reference_ears:
            return ()

        query_depth_channels = self._build_depth_channels(query_ear.profiles_by_scale)
        query_gradient_channels = self._build_gradient_channels(query_ear.profiles_by_scale)
        reference_depth_channels = np.array([self._build_depth_channels(ear.profiles_by_scale) for ear in reference_ears])
        reference_gradient_channels = np.array(
            [self._build_gradient_channels(ear.profiles_by_scale) for ear in reference_ears]
        )
        scores = self._score_alignments(
            query_depth_channels, query_gradient_channels, reference_depth_channels, reference_gradient_channels
        )
        return tuple(self._select_best_alignment(reference_scores) for reference_scores in scores)

    def score_side(self, query_ear: AnalyzedEar, reference_ears: Sequence[AnalyzedEar]) -> float:
        """Return the left score or right score of one candidate.

        The side score is the log-mean-exp of the ear similarities of the
        query ear with the candidate's reference ears on the same side.

        Args:
            query_ear: The query ear on one side.
            reference_ears: At least one reference ear of the candidate, on the
                same side as the query ear.

        Raises:
            ValueError: If an ear does not have one tear profile for each alpha
                scale of the query, or if the tear profiles have different lengths.
        """
        ear_similarities = self.compare_ears(query_ear, reference_ears)
        scores = np.array([similarity.score for similarity in ear_similarities])
        return log_mean_exp(scores, self._config.temperature)

    def _build_depth_channels(self, profiles: Sequence[TearProfile]) -> np.ndarray:
        """Return one depth channel for each tear profile.

        For each tear profile, this method sets negative depths to zero,
        calculates the square root of each depth, and resamples the depths to
        `resampled_bins`.

        Returns:
            Depth channels with shape `(scale_count, resampled_bins)`.
        """
        return np.array([self._resample(np.sqrt(np.maximum(profile.depths, 0.0))) for profile in profiles])

    def _build_gradient_channels(self, profiles: Sequence[TearProfile]) -> np.ndarray:
        """Return one gradient channel for each tear profile.

        For each tear profile, this method does these steps:

        1. Set negative depths to zero.
        2. Calculate the gradient per sample, inside the measured interval
           only. The gradient uses one-sided differences at the ends of the
           measured interval. The gradient is zero outside the measured interval.
        3. Calculate the square root of each magnitude, and keep its sign.
        4. Resample the result to `resampled_bins`.

        Returns:
            Gradient channels with shape `(scale_count, resampled_bins)`.
        """
        channels = []
        for profile in profiles:
            angles = np.linspace(0.0, 180.0, len(profile.depths))
            measured = (angles > profile.start_degrees) & (angles < profile.end_degrees)
            gradient = np.zeros(len(profile.depths))
            gradient[measured] = np.gradient(np.maximum(profile.depths[measured], 0.0))
            channels.append(self._resample(np.sign(gradient) * np.sqrt(np.abs(gradient))))
        return np.array(channels)

    def _resample(self, values: np.ndarray) -> np.ndarray:
        """Return the values linearly interpolated to `resampled_bins` uniform angles from 0° to 180°."""
        grid_positions = np.linspace(0.0, len(values) - 1, self._config.resampled_bins)
        return np.interp(grid_positions, np.arange(len(values)), values)

    def _align_channel(self, channel: np.ndarray) -> np.ndarray:
        """Return the channel at every alignment, with shape `(alignment_count, bins)`."""
        stretched = {stretch: _stretch(channel, stretch) for stretch in self._config.stretches}
        return np.array([_shift(stretched[stretch], shift_bins) for stretch, shift_bins in self._alignments])

    def _score_alignments(
        self,
        query_depth_channels: np.ndarray,
        query_gradient_channels: np.ndarray,
        reference_depth_channels: np.ndarray,
        reference_gradient_channels: np.ndarray,
    ) -> np.ndarray:
        """Return the alignment score of every alignment against each reference ear.

        For each alignment, this method does these steps:

        1. Multiply each aligned query gradient channel by `stretch ** -0.5`.
        2. Calculate the weighted real-valued Jaccard of each channel. The angle
           weight stays at the grid angles.
        3. Calculate the mean over the alpha scales, once for the depth channels
           and once for the gradient channels.
        4. Add the two channel means, each multiplied by its channel weight.
        5. Multiply the sum by the shift penalty and the stretch penalty.

        Column `j` contains the score of alignment `self._alignments[j]`.

        Args:
            query_depth_channels: The query's depth channels, with shape `(scale_count, bins)`.
            query_gradient_channels: The query's gradient channels, with shape `(scale_count, bins)`.
            reference_depth_channels: The reference ears' depth channels, with shape
                `(reference_count, scale_count, bins)`.
            reference_gradient_channels: The reference ears' gradient channels, with
                shape `(reference_count, scale_count, bins)`.

        Returns:
            Scores with shape `(reference_count, alignment_count)`.
        """
        depth_score = np.mean(
            [
                weighted_real_valued_jaccard(
                    reference_depth_channels[:, scale_index],
                    self._align_channel(query_channel),
                    self._angle_weights,
                )
                for scale_index, query_channel in enumerate(query_depth_channels)
            ],
            axis=0,
        )
        gradient_score = np.mean(
            [
                weighted_real_valued_jaccard(
                    reference_gradient_channels[:, scale_index],
                    self._align_channel(query_channel) * self._gradient_stretch_factors[:, None],
                    self._angle_weights,
                )
                for scale_index, query_channel in enumerate(query_gradient_channels)
            ],
            axis=0,
        )
        gradient_weight = self._config.gradient_weight
        return ((1.0 - gradient_weight) * depth_score + gradient_weight * gradient_score) * self._alignment_penalties

    def _compute_shift_penalty(self, shift_bins: int) -> float:
        """Return the shift penalty for a shift of `shift_bins`."""
        shift_fraction = abs(shift_bins) / self._config.resampled_bins
        return float(
            np.exp(-((shift_fraction / self._config.shift_penalty_scale) ** self._config.shift_penalty_power))
        )

    def _compute_stretch_penalty(self, stretch: float) -> float:
        """Return the stretch penalty for a stretch of `stretch`. It is the same for `stretch` and `1 / stretch`."""
        log_stretch = abs(np.log(stretch))
        return float(
            np.exp(-((log_stretch / self._config.stretch_penalty_scale) ** self._config.stretch_penalty_power))
        )

    def _select_best_alignment(self, scores: np.ndarray) -> EarSimilarity:
        """Return the highest alignment score as the ear similarity, with its alignment."""
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score <= 1e-12: # approximately zero
            return EarSimilarity(score=0.0, stretch=1.0, shift_bins=0)
        stretch, shift_bins = self._alignments[best]
        return EarSimilarity(score=score, stretch=float(stretch), shift_bins=shift_bins)


def _has_same_scale_count(query_ear: AnalyzedEar, reference_ears: Sequence[AnalyzedEar]) -> bool:
    """Return whether the query and all reference ears have the same number of alpha scales."""
    scale_count = len(query_ear.profiles_by_scale)
    return scale_count > 0 and all(len(ear.profiles_by_scale) == scale_count for ear in reference_ears)


def _has_same_length(query_ear: AnalyzedEar, reference_ears: Sequence[AnalyzedEar]) -> bool:
    """Return whether all tear profiles of all ears have the same length."""
    lengths = {
        len(profile.depths)
        for ear in (query_ear, *reference_ears)
        for profile in ear.profiles_by_scale
    }
    return len(lengths) == 1


def _stretch(channel: np.ndarray, stretch: float) -> np.ndarray:
    """Return the channel stretched about its center. Bins with no source value are zero."""
    output_fractions = np.linspace(0.0, 1.0, len(channel))
    source_positions = ((output_fractions - 0.5) / stretch + 0.5) * (len(channel) - 1)
    return np.interp(source_positions, np.arange(len(channel)), channel, left=0.0, right=0.0)


def _shift(channel: np.ndarray, shift_bins: int) -> np.ndarray:
    """Return the channel moved by `shift_bins`. Bins with no source value are zero.

    A positive shift moves the channel to larger angles. A negative shift moves
    the channel to smaller angles.
    """
    shifted = np.zeros_like(channel)
    if 0 <= shift_bins < len(channel):
        shifted[shift_bins:] = channel[: len(channel) - shift_bins]
    elif -len(channel) < shift_bins < 0:
        shifted[:shift_bins] = channel[-shift_bins:]
    return shifted


def weighted_real_valued_jaccard(
    channels_a: np.ndarray,
    channels_b: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return the weighted real-valued Jaccard of every channel in `channels_a` with every channel in `channels_b`.

    Channels can contain negative values. The value is zero when both channels
    contain only zeros.

    Args:
        channels_a: Channels with shape `(count_a, bins)`.
        channels_b: Channels with shape `(count_b, bins)`.
        weights: One nonnegative weight for each bin.

    Returns:
        Values with shape `(count_a, count_b)`.
    """
    # With weighted L1 norms, (|x| + |y| - |x - y|) / (|x| + |y| + |x - y|)
    # is the weighted shared magnitude of the positive and negative parts divided
    # by their weighted combined magnitude. Values of opposite sign share nothing.
    norms_a = np.abs(channels_a) @ weights
    norms_b = np.abs(channels_b) @ weights
    norm_sum = norms_a[:, None] + norms_b[None, :]
    distance = scipy_distance.cdist(channels_a, channels_b, metric="cityblock", w=weights)
    numerator = norm_sum - distance
    denominator = norm_sum + distance
    return np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator > 0)


def log_mean_exp(values: np.ndarray, temperature: float) -> float:
    """Return the log-mean-exp of at least one value: `temperature * (logsumexp(values / temperature) - log(n))`.

    The log-mean-exp is a smooth maximum. It lies between the mean and the
    maximum of the values.
    """
    return float(temperature * (logsumexp(values / temperature) - np.log(len(values))))
