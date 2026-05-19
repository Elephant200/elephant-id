from pathlib import Path

import numpy as np
from PIL import Image

from elephant_id.cache import CacheManager
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.visualize import decode_rle_mask


class AgeRunner:
    """
    Runner for the age regression CNN. Uses pytorch.
    """

    def __init__(self) -> None:
        # Initialize pytorch model and configure for inference
        ...

    def run(self, image: Image.Image) -> dict:
        # Run model on image
        ...

class AgeService:
    """
    Service for running the age regression CNN and caching the results.
    """

    def __init__(
        self,
        dataset: Dataset,
        cache_root: Path = Path(DEFAULT_CACHE_ROOT)
    ) -> None:
        self.runner = AgeRunner()
        self.dataset = dataset
        self.cache_manager = CacheManager(
            namespace="age",
            cache_root=cache_root,
        )

    def run(self, photo: Photo, body_rle_mask: np.ndarray) -> dict:
        key = f"{photo.identifier}"

        body_mask = decode_rle_mask(body_rle_mask)
        # TODO: crop the image to include only the body
        crop_xyxy = (0, 0, body_mask.shape[1], body_mask.shape[0])

        return self.cache_manager.get_or_compute(
            key=key,
            compute_fn=lambda: self.runner.run(
                image=self.dataset.read_image(photo, mask=body_mask, crop=crop_xyxy),
            ),
        )
