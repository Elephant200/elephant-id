"""Ear-segmentation protocol and implementations."""

from elephant_id.inference.segmentation.protocol import EarSegmenter
from elephant_id.inference.segmentation.sam3 import Sam3EarSegmenter

__all__ = ["EarSegmenter", "Sam3EarSegmenter"]
