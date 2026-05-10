"""
Generates a SEEK code for one elephant
"""

import numpy as np

from vision.sam3 import segment_image
from seek.view import get_view


def get_seek_code(elephant_image: np.ndarray) -> str:
    """
    Given a cropped image of one elephant, output the corresponding SEEK code in the form:
    _ _ _ T _ _ E _ _ _ _ - _ _ _ _ X _ _ S _ _ _
    Following the form of https://elephantsalive.org/wp-content/uploads/2021/07/Bedetti-et-al-2020.pdf#page=7
    """
    predictions = segment_image(
        image=elephant_image,
        queries=["trunk", "tusk", "ear", "tail"],
        confidence_threshold=0.5,
        nms=True,
        nms_iou_threshold=0.2,
    )
    
    view = get_view(elephant_image)
