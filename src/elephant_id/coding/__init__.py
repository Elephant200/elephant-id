"""SEEK-code generation.

Combines per-photo model outputs (segmentation, keypoints, gender, age) into a
single SEEK code for one elephant sighting.
"""

from .coder import SeekCoder

__all__ = ["SeekCoder"]
