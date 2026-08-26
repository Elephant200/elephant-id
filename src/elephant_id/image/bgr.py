"""The canonical in-memory image type and decode boundary for AlphaPhant.

A `BgrImage` is an OpenCV-native image array: HWC layout, 3 channels
in BGR order, and uint8 dtype. The alias is unchecked; arrays enter the
system in this form via `cv2` decode.
"""

import cv2
import numpy as np
from cv2.typing import MatLike

type BgrImage = MatLike
"""An HWC, BGR, uint8 image array."""


def decode_image(encoded: bytes) -> BgrImage:
    """Decode original encoded bytes into an OpenCV-native BGR image.

    Raises:
        ValueError: If the bytes do not contain a valid encoded image.
    """
    buffer = np.frombuffer(encoded, dtype=np.uint8)
    if buffer.size == 0:
        raise ValueError("Encoded bytes do not contain a valid image")
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Encoded bytes do not contain a valid image")
    return image
