"""Tests for the sighting store."""

from pathlib import Path

import numpy as np
import pytest

from elephant_id.api.store import SightingStore
from elephant_id.constants import TEAR_PROFILE_BINS


def test_create_update_get_roundtrip(tmp_path: Path) -> None:
    store = SightingStore(tmp_path)
    record = store.create(Path("/somewhere/2020-01-01"))
    sighting_id = record["sighting_id"]

    assert record["status"] == "analyzing"
    store.update(sighting_id, status="ready", profile_count=3)
    assert store.get(sighting_id)["status"] == "ready"
    assert store.list()[0]["profile_count"] == 3

    with pytest.raises(KeyError):
        store.get("missing")


def test_records_persist_across_instances(tmp_path: Path) -> None:
    first = SightingStore(tmp_path)
    sighting_id = first.create(Path("/somewhere/2020-01-01"))["sighting_id"]

    second = SightingStore(tmp_path)
    assert second.get(sighting_id)["folder"] == "/somewhere/2020-01-01"


def test_profiles_roundtrip(tmp_path: Path) -> None:
    store = SightingStore(tmp_path)
    sighting_id = store.create(Path("/somewhere/2020-01-01"))["sighting_id"]
    profiles = np.random.default_rng(0).uniform(0, 0.2, (2, TEAR_PROFILE_BINS))

    store.save_profiles(
        sighting_id,
        profiles,
        ("left", "right"),
        ("p1", "p2"),
        ("/crops/p1.jpg", None),
    )
    loaded, sides, photo_ids, crop_paths = store.load_profiles(sighting_id)

    np.testing.assert_allclose(loaded, profiles)
    assert sides == ("left", "right")
    assert photo_ids == ("p1", "p2")
    assert crop_paths == ("/crops/p1.jpg", None)

    with pytest.raises(FileNotFoundError):
        store.load_profiles("missing")
