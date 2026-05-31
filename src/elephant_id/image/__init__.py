"""Image types and operations for elephant-id.

Only the canonical :data:`BgrImage` type is re-exported here. Import functions
directly from their submodules (:mod:`~elephant_id.image.boxes`,
:mod:`~elephant_id.image.masks`, :mod:`~elephant_id.image.transforms`).
"""

from elephant_id.image.bgr import BgrImage

__all__ = ["BgrImage"]
