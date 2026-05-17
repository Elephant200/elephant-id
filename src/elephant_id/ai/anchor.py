from pathlib import Path

from PIL import Image

from elephant_id.cache import CacheManager
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo


class AnchorRunner:
    """
    Runner for the anchor keypoint detection YOLO26 model. Uses ultralytics.
    """

    def __init__(self) -> None:
        # Initialize roboflow client and configure for inference
        ...

    def run(self, image: Image.Image) -> dict:
        # Run model on image
        ...

class AnchorService:
    """
    Service for running the anchor keypoint detection YOLO26 model and caching the results.
    """

    def __init__(
        self,
        dataset: Dataset,
        cache_root: Path = Path(DEFAULT_CACHE_ROOT)
    ) -> None:
        self.runner = AnchorRunner()
        self.dataset = dataset
        self.cache_manager = CacheManager(
            namespace="anchor",
            cache_root=cache_root,
        )

    def run(self, photo: Photo, crop: tuple[int, int, int, int]) -> dict:
        key = f"f{photo.identifier}"

        coords = self.cache_manager.get_or_compute(
            key=key,
            compute_fn=lambda: self.runner.run(
                image=self.dataset.read_image(photo, crop=crop), # TODO: add crop logic to dataset
            ),
        )
        # TODO: translate coords to absolute coordinates
        return coords
