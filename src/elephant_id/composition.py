"""Explicit construction of shared preparation and standard AlphaPhant."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import cache
from pathlib import Path

from elephant_id.cache import CacheManager
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import PhotoStore
from elephant_id.domain import SightingEarPair
from elephant_id.matching.alphaphant import AlphaPhant
from elephant_id.matching.alphaphant.cached import CachedTearProfileExtractor
from elephant_id.matching.alphaphant.extraction import (
    MULTISCALE_VERSIONS,
    AlphaTearExtractor,
    AlphaTearVersion,
)
from elephant_id.matching.alphaphant.matcher import CHANNEL_WEIGHTS
from elephant_id.matching.alphaphant.profile import TearProfileExtractor
from elephant_id.matching.alphaphant.similarity import (
    SELECTED_PROFILE_SETTINGS,
    TearMatcher,
)
from elephant_id.preparation import PreparedEar, SightingPreparer


def build_preparer(
    photo_store: PhotoStore,
    cache_store: CacheManager,
    *,
    roboflow_api_key: str | None = None,
) -> SightingPreparer:
    """Construct shared preparation with cached segmentation and landmarks."""
    from elephant_id.inference.landmarks.cached import CachedEarLandmarkDetector
    from elephant_id.inference.landmarks.yolo import YoloEarLandmarkDetector
    from elephant_id.inference.segmentation.sam3 import (
        CachedSam3FeatureSegmenter,
        Sam3EarSegmenter,
        Sam3FeatureSegmenter,
    )

    features = CachedSam3FeatureSegmenter(
        Sam3FeatureSegmenter(api_key=roboflow_api_key), cache_store
    )
    return SightingPreparer(
        photo_store=photo_store,
        ear_segmenter=Sam3EarSegmenter(features),
        landmark_detector=CachedEarLandmarkDetector(
            YoloEarLandmarkDetector(), cache_store
        ),
    )


def build_profile_extractors(
    preparer: SightingPreparer,
    cache_store: CacheManager,
    versions: Sequence[AlphaTearVersion] = MULTISCALE_VERSIONS,
) -> tuple[CachedTearProfileExtractor, ...]:
    """Construct settled profile caches with the actual inference provenance."""
    return tuple(
        CachedTearProfileExtractor(
            AlphaTearExtractor(version),
            cache_store,
            segmentation_producer_slug=preparer.segmentation_producer_slug,
            landmark_producer_slug=preparer.landmark_producer_slug,
        )
        for version in versions
    )


def compose_alphaphant(
    prepare_ears: Callable[[SightingEarPair], tuple[PreparedEar, PreparedEar]],
    profile_extractors: Sequence[TearProfileExtractor],
) -> AlphaPhant:
    """Wire supplied preparation and raw or cached extractors to standard scoring."""
    return AlphaPhant(
        prepare_ears=prepare_ears,
        profile_extractors=profile_extractors,
        channel_matchers=(
            TearMatcher(SELECTED_PROFILE_SETTINGS),
            TearMatcher(
                replace(SELECTED_PROFILE_SETTINGS, channel="signed_depth_change")
            ),
        ),
        channel_weights=CHANNEL_WEIGHTS,
    )


def build_standard_alphaphant(
    photo_store: PhotoStore,
    cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    *,
    roboflow_api_key: str | None = None,
) -> AlphaPhant:
    """Construct standard AlphaPhant with cached inference and seven profile scales."""
    cache_store = CacheManager(cache_root=cache_root)
    preparer = build_preparer(
        photo_store, cache_store, roboflow_api_key=roboflow_api_key
    )
    return compose_alphaphant(
        cache(preparer.prepare), build_profile_extractors(preparer, cache_store)
    )
