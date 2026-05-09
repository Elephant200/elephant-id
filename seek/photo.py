import cv2
import os
from datetime import datetime
import numpy as np

from preprocess.background import remove_background
from vision.sam3 import segment_image

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
        self.identifier: str = os.path.basename(image_path) # Unique identifier for the photo

        self._raw_data: dict | None = None

    def get_data(self) -> dict:
        if self._raw_data is not None: # Lazy loading
            return self._raw_data
        
        
        self._raw_data = segment_image(
            image=self.image,
            queries=["trunk", "tusk", "ear", "tail"],
            confidence_threshold=0.5,
            nms=True,
            nms_iou_threshold=0.2,
        )



    def __str__(self) -> str:
        return f"Photo(identifier={self.identifier}, elephant={self.elephant}, date={self.date})"
    
    def __repr__(self) -> str:
        return f"Photo(identifier={self.identifier}, elephant={self.elephant}, date={self.date})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Photo):
            return False
        return self.identifier == other.identifier and self.elephant == other.elephant and self.date == other.date
    
    def __hash__(self) -> int:
        return hash(self.identifier)

if __name__ == "__main__":
    photo = Photo(
        image_path="dataset/sample/",
        elephant="Ariel II",
        date=datetime(2011, 2, 1),
    )
    print(photo)