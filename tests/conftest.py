from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as coco_mask

from elephant_id.domain import Photo, Sighting


def _rle_from_mask(mask: np.ndarray) -> dict:
    encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return encoded


@pytest.fixture
def rle_from_mask() -> Callable[[np.ndarray], dict]:
    return _rle_from_mask


@pytest.fixture
def make_photo() -> Callable[..., Photo]:
    def factory(
        *,
        name: str = "Devin",
        elephant_name: str | None = None,
        sighting_date: str = "2015-11-05",
        sequence: int = 1,
        identifier: str | None = None,
        image_path: Path | None = None,
        sighting_id: str | None = None,
    ) -> Photo:
        resolved_name = elephant_name if elephant_name is not None else name
        resolved_sighting_id = (
            sighting_id
            if sighting_id is not None
            else f"{resolved_name}_{sighting_date}"
        )
        resolved_identifier = (
            identifier
            if identifier is not None
            else f"{resolved_sighting_id}_{sequence:02d}"
        )
        return Photo(
            identifier=resolved_identifier,
            image_path=image_path
            or Path(f"{resolved_name}/{sighting_date}/{resolved_identifier}.jpg"),
            elephant_name=resolved_name,
            sighting_id=resolved_sighting_id,
        )

    return factory


@pytest.fixture
def make_sighting(make_photo: Callable[..., Photo]) -> Callable[..., Sighting]:
    def factory(
        *,
        name: str = "Devin",
        elephant_name: str | None = None,
        sighting_date: date | None = None,
        sighting_id: str | None = None,
        photos: tuple[Photo, ...] | None = None,
    ) -> Sighting:
        resolved_name = elephant_name if elephant_name is not None else name
        resolved_date = sighting_date or date(2015, 11, 5)
        resolved_sighting_id = (
            sighting_id
            if sighting_id is not None
            else f"{resolved_name}_{resolved_date.isoformat()}"
        )
        default_photo_sighting_id = f"{name}_{resolved_date.isoformat()}"
        return Sighting(
            elephant_name=resolved_name,
            sighting_date=resolved_date,
            sighting_id=resolved_sighting_id,
            photos=photos
            if photos is not None
            else (
                make_photo(
                    name=name,
                    sighting_date=resolved_date.isoformat(),
                    sighting_id=default_photo_sighting_id,
                ),
            ),
        )

    return factory
