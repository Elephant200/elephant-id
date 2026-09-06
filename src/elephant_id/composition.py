"""Composition of shared ear preparation and publication AlphaPhant."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import cache
from pathlib import Path

from elephant_id.analysis import PreparedEar, SightingAnalyzer, SightingPreparer
from elephant_id.analysis.profile_extraction import (
    AlphaTearConfig,
    AlphaTearExtractor,
    CachedTearProfileExtractor,
)
from elephant_id.analysis.profile_extraction.alpha_tear import (
    DEFAULT_VERSION,
    MULTISCALE_VERSIONS,
    AlphaTearVersion,
)
from elephant_id.cache import CacheManager
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import PhotoStore
from elephant_id.domain import SightingEarPair
from elephant_id.inference.landmarks import (
    CachedEarLandmarkDetector,
    YoloEarLandmarkDetector,
)
from elephant_id.inference.segmentation.sam3 import (
    CachedSam3FeatureSegmenter,
    Sam3EarSegmenter,
    Sam3FeatureSegmenter,
)
from elephant_id.matching import AlphaPhant
from elephant_id.matching.tear_matcher import (
    TearMatcher,
    TearMatcherConfig,
    angular_weights,
)

SELECTED_PROFILE_SETTINGS = TearMatcherConfig(
    depth_exponent=0.75,
    shift_penalty_scale=0.16,
    bin_weights=angular_weights(240, 120.0, 35.0),
)
"""Fixed profile settings selected on the frozen tuning sets."""

CHANNEL_WEIGHTS = (0.55, 0.45)
"""Depth and signed-depth-change shares of each directional ear similarity."""


def _shared_preparation(
    photo_store: PhotoStore,
    cache_store: CacheManager,
    roboflow_api_key: str | None,
) -> tuple[Callable[[SightingEarPair], tuple[PreparedEar, PreparedEar]], str, str]:
    """Compose reusable preparation over complete cached inference producers."""
    features = CachedSam3FeatureSegmenter(
        Sam3FeatureSegmenter(api_key=roboflow_api_key), cache_store
    )
    segmenter = Sam3EarSegmenter(features)
    landmarks = CachedEarLandmarkDetector(YoloEarLandmarkDetector(), cache_store)
    preparer = SightingPreparer(
        photo_store=photo_store, ear_segmenter=segmenter, landmark_detector=landmarks
    )
    return cache(preparer.prepare), segmenter.producer_slug, landmarks.producer_slug


def build_versioned_analyzers(
    photo_store: PhotoStore,
    versions: Sequence[AlphaTearVersion],
    cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    *,
    roboflow_api_key: str | None = None,
) -> tuple[SightingAnalyzer, ...]:
    """Build extraction versions that share exactly one ear-preparation computation."""
    cache_store = CacheManager(cache_root=cache_root)
    prepare, segmentation_slug, landmark_slug = _shared_preparation(
        photo_store, cache_store, roboflow_api_key
    )
    return tuple(
        SightingAnalyzer(
            prepare_ears=prepare,
            profile_extractor=CachedTearProfileExtractor(
                AlphaTearExtractor(version),
                cache_store,
                segmentation_producer_slug=segmentation_slug,
                landmark_producer_slug=landmark_slug,
            ),
        )
        for version in versions
    )


def build_standard_analyzer(
    photo_store: PhotoStore,
    cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    *,
    roboflow_api_key: str | None = None,
) -> SightingAnalyzer:
    """Build the single-scale analysis baseline over shared standard preparation."""
    return build_versioned_analyzers(
        photo_store, (DEFAULT_VERSION,), cache_root, roboflow_api_key=roboflow_api_key
    )[0]


def compose_alphaphant(scale_analyzers: Sequence[SightingAnalyzer]) -> AlphaPhant:
    """Apply the selected matching rule to the supplied scale analyzers."""
    return AlphaPhant(
        scale_analyzers=scale_analyzers,
        channel_matchers=(
            TearMatcher(SELECTED_PROFILE_SETTINGS),
            TearMatcher(replace(SELECTED_PROFILE_SETTINGS, channel="signed_depth_change")),
        ),
        channel_weights=CHANNEL_WEIGHTS,
    )


def build_standard_alphaphant(
    photo_store: PhotoStore,
    cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    *,
    roboflow_api_key: str | None = None,
) -> AlphaPhant:
    """Build publication AlphaPhant with seven immutable extraction versions."""
    return compose_alphaphant(
        build_versioned_analyzers(
            photo_store,
            MULTISCALE_VERSIONS,
            cache_root,
            roboflow_api_key=roboflow_api_key,
        )
    )


def build_profile_tuning_analyzer(
    photo_store: PhotoStore,
    profile_config: AlphaTearConfig,
    cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    *,
    roboflow_api_key: str | None = None,
) -> SightingAnalyzer:
    """Compose raw experimental extraction over the cached inference producers."""
    prepare, _, _ = _shared_preparation(
        photo_store, CacheManager(cache_root=cache_root), roboflow_api_key
    )
    return SightingAnalyzer(
        prepare_ears=prepare, profile_extractor=AlphaTearExtractor(profile_config)
    )
