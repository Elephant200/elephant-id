"""
AI model services for the Elephant ID system.

These services provide a unified interface for running AI models and
caching their results.

The services are:
- AgeService: for running the age regression CNN
- AnchorService: for running the anchor keypoint detection YOLO26 model
- GenderService: for running the gender classification CNN
- Sam3Service: for running the Facebook SAM3 segmentation model
"""

from .age import AgeService
from .anchor import AnchorService
from .detection import Detection
from .gender import GenderService
from .sam3 import Sam3Service

__all__ = ["AgeService", "AnchorService", "Detection", "GenderService", "Sam3Service"]
