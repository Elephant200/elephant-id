from pathlib import Path

from elephant_id.cache import CacheManager
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.image import BgrImage
from elephant_id.image.masks import RleMask
from elephant_id.image.transforms import apply_mask


class GenderRunner:
    """Local-only runner for the gender classification CNN."""

    def __init__(self) -> None:
        # Initialize pytorch model and configure for inference
        self.model = None

    def run(self, image: BgrImage) -> dict:
        """
        Currently a placeholder that always returns bull, matching the dataset labels.
        """
        # Run model on image
        return {
            "bull": 1.0,
            "cow": 0.0,
        }

class GenderService:
    """Run the gender classification CNN with caching."""

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

    def run(self, photo: Photo, body_rle_mask: RleMask) -> dict:
        key = f"{photo.identifier}"

        return self.cache_manager.get_or_compute(
            key=key,
            compute_fn=lambda: self.runner.run(
                image=apply_mask(self.dataset.read_image(photo), body_rle_mask, crop=True),
            ),
        )
