from typing import Literal
import numpy as np

from seek.photo import Photo

def get_view(elephant_image: Photo) -> Literal["front", "left", "right", "back"]:
    return "front"