"""The canonical in-memory image type for elephant-id.

A ``BgrImage`` is an OpenCV-native image array: HWC layout, 3 channels in BGR
order (the OpenCV / ultralytics / Roboflow convention), uint8 dtype. The alias
is unchecked -- arrays enter the system already in this form via ``cv2`` decode.
"""

from cv2.typing import MatLike

type BgrImage = MatLike
"""An HWC, BGR, uint8 image array."""
