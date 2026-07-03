"""Tests for gallery loading."""

from pathlib import Path

import numpy as np
import pytest

from elephant_id.api.gallery import GalleryData, load_gallery
from elephant_id.constants import TEAR_PROFILE_BINS


def write_profile_npz(path: Path) -> None:
    """Write a two-row profile cache in the evaluation script's format."""
    np.savez_compressed(
        path,
        profiles=np.zeros((2, TEAR_PROFILE_BINS)),
        photo_ids=np.asarray(["Ana_2020-01-01_01", "Ana_2020-01-01_02"]),
        identities=np.asarray(["Ana", "Ana"]),
        sides=np.asarray(["left", "right"]),
        dates=np.asarray(["2020-01-01", "2020-01-01"]),
        skipped_count=np.asarray(0),
    )


def test_load_gallery_joins_crop_paths(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.npz"
    write_profile_npz(profiles_path)
    crop = tmp_path / "images" / "left" / "Ana"
    crop.mkdir(parents=True)
    crop_file = crop / "Ana_2020-01-01_01.jpg"
    crop_file.write_bytes(b"jpg")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "side,identity,photo_identifier,exported_path\n"
        "left,Ana,Ana_2020-01-01_01,images/left/Ana/Ana_2020-01-01_01.jpg\n"
        "right,Ana,Ana_2020-01-01_02,images/right/Ana/missing.jpg\n"
    )

    gallery = load_gallery(profiles_path, manifest)

    assert gallery.crop_paths == (str(crop_file), None)
    assert gallery.identities == ("Ana", "Ana")


def test_load_gallery_tolerates_missing_manifest(tmp_path: Path) -> None:
    profiles_path = tmp_path / "profiles.npz"
    write_profile_npz(profiles_path)

    gallery = load_gallery(profiles_path, tmp_path / "absent.csv")

    assert gallery.crop_paths == (None, None)


def test_load_gallery_requires_profiles(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_gallery(tmp_path / "absent.npz", tmp_path / "absent.csv")


def test_gallery_data_validates_alignment() -> None:
    with pytest.raises(ValueError):
        GalleryData(
            profiles=np.zeros((2, TEAR_PROFILE_BINS)),
            photo_ids=("only-one",),
            identities=("Ana", "Ana"),
            sides=("left", "right"),
            dates=("2020-01-01", "2020-01-01"),
            crop_paths=(None, None),
        )
