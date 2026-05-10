"""
Generates a SEEK code for one elephant
"""

import numpy as np

def get_seek_code(elephant_image: np.ndarray) -> str:
    """
    Given a cropped image of one elephant, output the corresponding SEEK code in the form:
    _ _ _ T _ _ E _ _ _ _ - _ _ _ _ X _ _ S _ _ _
    Following the form of https://elephantsalive.org/wp-content/uploads/2021/07/Bedetti-et-al-2020.pdf#page=7
    """
    
