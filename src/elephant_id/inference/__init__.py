"""Public semantic inference values and protocols."""

from elephant_id.inference.detection import Detection
from elephant_id.inference.landmarks import EarLandmarkDetector
from elephant_id.inference.segmentation import EarSegmenter

__all__ = [
    "Detection",
    "EarLandmarkDetector",
    "EarSegmenter",
]
