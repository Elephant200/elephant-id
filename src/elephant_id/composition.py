"""Construction of standard and profile-tuning sighting analyzers."""

from pathlib import Path

from elephant_id.analysis import SightingAnalyzer
from elephant_id.analysis.profile_extraction import (
    AlphaTearConfig,
    AlphaTearExtractor,
    CachedTearProfileExtractor,
)
from elephant_id.analysis.profile_extraction.alpha_tear import DEFAULT_VERSION
from elephant_id.cache import CacheManager
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import PhotoStore
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
from elephant_id.matching.tear_matcher import TearMatcher


def _cached_inference(
    cache: CacheManager,
    roboflow_api_key: str | None,
) -> tuple[Sam3EarSegmenter, CachedEarLandmarkDetector]:
    """Build the shared cached segmentation and landmark stack."""
    cached_features = CachedSam3FeatureSegmenter(
        Sam3FeatureSegmenter(api_key=roboflow_api_key),
        cache,
    )
    return (
        Sam3EarSegmenter(cached_features),
        CachedEarLandmarkDetector(YoloEarLandmarkDetector(), cache),
    )


def build_standard_analyzer(
    photo_store: PhotoStore,
    cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    *,
    roboflow_api_key: str | None = None,
) -> SightingAnalyzer:
    """Build the standard AlphaPhant sighting analyzer."""
    cache = CacheManager(cache_root=cache_root)
    segmenter, landmarks = _cached_inference(cache, roboflow_api_key)
    profiles = CachedTearProfileExtractor(
        AlphaTearExtractor(DEFAULT_VERSION),
        cache,
        segmentation_producer_slug=segmenter.producer_slug,
        landmark_producer_slug=landmarks.producer_slug,
    )
    return SightingAnalyzer(
        photo_store=photo_store,
        ear_segmenter=segmenter,
        landmark_detector=landmarks,
        profile_extractor=profiles,
    )


def build_standard_alphaphant(
    photo_store: PhotoStore,
    cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    *,
    roboflow_api_key: str | None = None,
) -> AlphaPhant:
    """Build standard cached AlphaPhant catalog matching."""
    return AlphaPhant(
        analyzer=build_standard_analyzer(
            photo_store,
            cache_root,
            roboflow_api_key=roboflow_api_key,
        ),
        tear_matcher=TearMatcher(),
    )


def build_profile_tuning_analyzer(
    photo_store: PhotoStore,
    profile_config: AlphaTearConfig,
    cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    *,
    roboflow_api_key: str | None = None,
) -> SightingAnalyzer:
    """Build an analyzer that bypasses only tear-profile persistence."""
    cache = CacheManager(cache_root=cache_root)
    segmenter, landmarks = _cached_inference(cache, roboflow_api_key)
    return SightingAnalyzer(
        photo_store=photo_store,
        ear_segmenter=segmenter,
        landmark_detector=landmarks,
        profile_extractor=AlphaTearExtractor(profile_config),
    )
