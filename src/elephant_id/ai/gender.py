from pathlib import Path

import numpy as np
from PIL import Image

from elephant_id.cache import CacheManager
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo


class GenderRunner:
    """
    Runner for the gender classification CNN. Uses pytorch.
    """

    def __init__(self) -> None:
        # Initialize pytorch model and configure for inference
        ...

    def run(self, image: Image.Image) -> dict:
        # Run model on image
        ...

class GenderService:
    """
    Service for running the gender classification CNN and caching the results.
    """

    def __init__(
        self,
        dataset: Dataset,
        cache_root: Path = Path(DEFAULT_CACHE_ROOT)
    ) -> None:
        self.runner = GenderRunner()
        self.dataset = dataset
        self.cache_manager = CacheManager(
            namespace="gender",
            cache_root=cache_root,
        )

    def run(self, photo: Photo, body_mask: np.ndarray) -> dict:
        key = f"{photo.identifier}"

        return self.cache_manager.get_or_compute(
            key=key,
            compute_fn=lambda: self.runner.run(
                image=self.dataset.read_image(photo, mask=body_mask),
            ),
        )
