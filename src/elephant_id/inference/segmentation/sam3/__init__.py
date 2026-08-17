"""SAM3 full-feature computation, persistence, and ear adaptation."""

from elephant_id.inference.segmentation.sam3.cached import (
    CachedSam3FeatureSegmenter,
)
from elephant_id.inference.segmentation.sam3.ear_segmenter import Sam3EarSegmenter
from elephant_id.inference.segmentation.sam3.features import (
    Sam3FeatureConfig,
    Sam3FeatureSegmenter,
)

__all__ = [
    "CachedSam3FeatureSegmenter",
    "Sam3EarSegmenter",
    "Sam3FeatureConfig",
    "Sam3FeatureSegmenter",
]
