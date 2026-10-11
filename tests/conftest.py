from collections.abc import Callable

import numpy as np
import pytest
from pycocotools import mask as coco_mask


def _rle_from_mask(mask: np.ndarray) -> dict:
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


@pytest.fixture
def rle_from_mask() -> Callable[[np.ndarray], dict]:
    return _rle_from_mask
