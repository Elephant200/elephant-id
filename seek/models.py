import cv2
import os
from datetime import datetime
import numpy as np

from preprocess.background import remove_background
from vision.model import infer

class Photo:
    """
    One photo of an elephant
    """
    def __init__(self, image_path: str, elephant: str, date: datetime):
        """
        Initialize a Photo object.

        Args:
            image_path: Path to the image file.
            elephant: The name of the elephant in the image.
            date: The date the image was taken.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image at {image_path} does not exist")
        
        self.image: np.ndarray = cv2.imread(image_path)
        self.elephant: str = elephant
        self.date: datetime = date
        self.filename: str = os.path.basename(image_path)

        self._no_bg: np.ndarray | None = None # Image with background removed 
        self._raw_data: dict | None = None

    def _get_no_bg(self) -> np.ndarray:
        """
        Get the image with background removed.
        """
        if self._no_bg is not None:
            return self._no_bg
        # Check cache; located in .cache/ElephantsAlive/<elephant>/<date>_

        return self._no_bg



    def __str__(self) -> str:
        return f"Photo(filename={self.filename}, elephant={self.elephant}, date={self.date})"
    
    def __repr__(self) -> str:
        return f"Photo(filename={self.filename}, elephant={self.elephant}, date={self.date})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Photo):
            return False
        return self.filename == other.filename and self.elephant == other.elephant and self.date == other.date
    
    def __hash__(self) -> int:
        return hash(self.filename)

if __name__ == "__main__":
    photo = Photo(
        image_path="dataset/sample/",
        elephant="Ariel II",
        date=datetime(2011, 2, 1),
    )
    print(photo)