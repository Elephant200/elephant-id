"""Ear-landmark detection protocol and implementations."""

from elephant_id.inference.landmarks.cached import CachedEarLandmarkDetector
from elephant_id.inference.landmarks.protocol import EarLandmarkDetector
from elephant_id.inference.landmarks.yolo import YoloEarLandmarkDetector

__all__ = [
    "CachedEarLandmarkDetector",
    "EarLandmarkDetector",
    "YoloEarLandmarkDetector",
]
