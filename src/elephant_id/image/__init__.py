"""Image types and operations for elephant-id.

The encoded-byte decode boundary is exported beside the canonical
`BgrImage` type. Import other operations from their submodules.
"""

from elephant_id.image.bgr import BgrImage, decode_image

__all__ = ["BgrImage", "decode_image"]
