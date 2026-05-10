from pathlib import Path
from typing import Literal

from PIL import Image

from elephant_id.models import Photo


class Dataset:
    """
    Class for interacting with the SEEK elephant ID dataset.
    """

    def __init__(
        self,
        dataset_root: Path,
    ) -> None:
        self.dataset_root: Path = dataset_root

    def read_image(
        self, photo: Photo, size: Literal["original", "thumb"] = "original"
    ) -> Image.Image:
        """
        Read the image for the given photo
        """
        image_path = self.dataset_root / photo.image_path
        image = Image.open(image_path)
        if size == "original":
            return image
        elif size == "thumb":
            return image.resize((128, 128))
        else:
            raise ValueError(f"Invalid size: {size}")
