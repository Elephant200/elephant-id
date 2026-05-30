from pathlib import Path

from elephant_id.ai.anchor import AnchorService
from elephant_id.domain import Photo


class _RecordingCache:
    def __init__(self):
        self.key = None

    def get_or_compute(self, key, compute_fn):
        self.key = key
        return {
            "predictions": {
                "x1": 1,
                "y1": 2,
                "x2": 3,
                "y2": 4,
                "keypoints": [[5, 6], [7, 8]],
            }
        }


def test_anchor_cache_key_includes_full_crop_coordinates():
    service = AnchorService.__new__(AnchorService)
    service.cache_manager = _RecordingCache()
    service.dataset = object()
    service.runner = object()
    photo = Photo(
        identifier="Adam_2011-03-31_03",
        image_path=Path("Adam/2011-03-31/Adam_2011-03-31_03.jpg"),
        elephant_name="Adam",
        sighting_id="Adam_2011-03-31",
    )

    service.run(photo, crop_xyxy=(899.2, 93.8, 1260.1, 512.9))

    assert service.cache_manager.key == "Adam_2011-03-31_03__crop_899_93_1260_512"
