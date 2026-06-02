"""Image types and operations for elephant-id.

Only the canonical :data:`BgrImage` type is re-exported here. Import
functions directly from the boxes, masks, and transforms submodules.
"""

from elephant_id.image.bgr import BgrImage

__all__ = ["BgrImage"]
