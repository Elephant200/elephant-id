"""Construct shared preparation and standard or experimental AlphaPhant."""

from pathlib import Path

from elephant_id.cache import CacheManager
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import PhotoStore
from elephant_id.matching.alphaphant import AlphaPhant
from elephant_id.matching.alphaphant.cached import CachedTearProfileExtractor
from elephant_id.matching.alphaphant.extraction import (
    DEFAULT_VERSION,
    AlphaTearConfig,
    AlphaTearExtractor,
)
from elephant_id.matching.alphaphant.similarity import EarProfileSimilarity
from elephant_id.preparation import SightingPreparer


def build_preparer(
    photo_store: PhotoStore,
    cache_store: CacheManager,
    *,
    roboflow_api_key: str | None = None,
) -> SightingPreparer:
    """Construct shared preparation with cached segmentation and landmarks."""
    from elephant_id.inference.landmarks import (
        CachedEarLandmarkDetector,
        YoloEarLandmarkDetector,
    )
    from elephant_id.inference.segmentation.sam3 import (
        CachedSam3FeatureSegmenter,
        Sam3EarSegmenter,
        Sam3FeatureSegmenter,
    )

    features = CachedSam3FeatureSegmenter(Sam3FeatureSegmenter(api_key=roboflow_api_key), cache_store)
    return SightingPreparer(
        photo_store=photo_store,
        ear_segmenter=Sam3EarSegmenter(features),
        landmark_detector=CachedEarLandmarkDetector(YoloEarLandmarkDetector(), cache_store),
    )


def build_standard_alphaphant(
    photo_store: PhotoStore,
    cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    *,
    roboflow_api_key: str | None = None,
) -> AlphaPhant:
    """Construct standard AlphaPhant with cached inference and settled profiles."""
    cache_store = CacheManager(cache_root=cache_root)
    preparer = build_preparer(photo_store, cache_store, roboflow_api_key=roboflow_api_key)
    profiles = CachedTearProfileExtractor(
        AlphaTearExtractor(DEFAULT_VERSION),
        cache_store,
        segmentation_producer_slug=preparer.segmentation_producer_slug,
        landmark_producer_slug=preparer.landmark_producer_slug,
    )
    return AlphaPhant(
        prepare_ears=preparer.prepare,
        profile_extractor=profiles,
        ear_similarity=EarProfileSimilarity(),
    )


def build_profile_tuning_alphaphant(
    photo_store: PhotoStore,
    profile_config: AlphaTearConfig,
    cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    *,
    roboflow_api_key: str | None = None,
) -> AlphaPhant:
    """Construct experimental extraction with standard scoring and cached inference."""
    cache_store = CacheManager(cache_root=cache_root)
    preparer = build_preparer(photo_store, cache_store, roboflow_api_key=roboflow_api_key)
    return AlphaPhant(
        prepare_ears=preparer.prepare,
        profile_extractor=AlphaTearExtractor(profile_config),
        ear_similarity=EarProfileSimilarity(),
    )
