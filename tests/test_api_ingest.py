"""Tests for sighting folder ingest."""

from pathlib import Path

import numpy as np
import pytest

from elephant_id.api import ingest, overlays
from elephant_id.api.gallery import GalleryData
from elephant_id.constants import TEAR_PROFILE_BINS


class _FakeEar:
    """Minimal ear stand-in for evidence export."""

    side = "left"
    xyxy = (2.0, 2.0, 8.0, 8.0)
    contour = np.array([[3.0, 3.0], [5.0, 4.0], [7.0, 7.0]])
    anchor_points = ((3.0, 3.0), (7.0, 7.0))


class _FakeAnalyzer:
    """Analyzer returning one usable left ear for every photo."""

    def analyze(self, photo: object) -> dict:
        """Return a fixed one-ear analysis package."""
        return {
            "ears": [
                {
                    "ear": _FakeEar(),
                    "tear_profile": type(
                        "P", (), {"profile": np.ones(TEAR_PROFILE_BINS) * 0.1}
                    )(),
                }
            ],
            "view": "left",
            "tusks": [],
            "age": None,
            "gender": None,
        }


class _FakeDataset:
    """Dataset stand-in returning a blank image for any photo."""

    def get_photo(self, identifier: str) -> str:
        """Return the identifier as an opaque photo handle."""
        return identifier

    def read_image(self, photo: object) -> np.ndarray:
        """Return a blank BGR image."""
        return np.zeros((10, 10, 3), dtype=np.uint8)


@pytest.fixture
def gallery() -> GalleryData:
    """Gallery with precomputed profiles for one known photo."""
    return GalleryData(
        profiles=np.ones((2, TEAR_PROFILE_BINS)) * 0.1,
        photo_ids=("Zola_2019-03-03_05", "Zola_2019-03-03_05"),
        identities=("Zola", "Zola"),
        sides=("left", "right"),
        dates=("2019-03-03", "2019-03-03"),
        crop_paths=("/crops/left.jpg", None),
    )


@pytest.fixture
def sighting_folder(tmp_path: Path) -> Path:
    """Folder with one gallery-known photo, one unknown, one bad name."""
    folder = tmp_path / "sighting"
    folder.mkdir()
    for name in ("Zola_2019-03-03_05.jpg", "Zola_2019-03-03_06.jpg", "notes.png"):
        (folder / name).write_bytes(b"\xff\xd8\xff")
    (folder / "readme.txt").write_text("not an image")
    return folder


def test_photo_stem_pattern() -> None:
    match = ingest.PHOTO_STEM_PATTERN.match("Alvin_2014-10-12_08")
    assert match is not None
    assert match["name"] == "Alvin"
    assert match["date"] == "2014-10-12"
    assert ingest.PHOTO_STEM_PATTERN.match("IMG_1234") is None


def test_list_photo_files_filters_and_sorts(sighting_folder: Path) -> None:
    files = ingest.list_photo_files(sighting_folder)
    assert [file.name for file in files] == [
        "Zola_2019-03-03_05.jpg",
        "Zola_2019-03-03_06.jpg",
        "notes.png",
    ]
    with pytest.raises(NotADirectoryError):
        ingest.list_photo_files(sighting_folder / "absent")


def test_ingest_uses_precomputed_fallback(
    sighting_folder: Path,
    gallery: GalleryData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest, "_make_analyzer", lambda dataset, cache_root: None)
    progress_calls: list[tuple[int, int]] = []

    result = ingest.ingest_sighting(
        sighting_folder,
        tmp_path / "work",
        gallery,
        tmp_path / "cache",
        progress=lambda done, total: progress_calls.append((done, total)),
    )

    assert result.profiles.shape == (2, TEAR_PROFILE_BINS)
    assert result.sides == ("left", "right")
    # Product photo_ids are generated and identity-free; both profile rows come
    # from the one precomputed photo, so they share its generated id.
    assert len(set(result.photo_ids)) == 1
    (product_id,) = set(result.photo_ids)
    assert product_id.startswith("P-")
    assert "Zola" not in product_id
    assert result.crop_paths == ("/crops/left.jpg", None)
    statuses = {photo.file_name: photo.status for photo in result.photos}
    assert statuses == {
        "Zola_2019-03-03_05.jpg": "precomputed",
        "Zola_2019-03-03_06.jpg": "skipped",
        "notes.png": "skipped",
    }
    assert progress_calls[-1] == (3, 3)


def test_ingest_skips_unparseable_names_with_reason(
    sighting_folder: Path,
    gallery: GalleryData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest, "_make_analyzer", lambda dataset, cache_root: None)

    result = ingest.ingest_sighting(
        sighting_folder, tmp_path / "work", gallery, tmp_path / "cache"
    )

    bad = next(photo for photo in result.photos if photo.file_name == "notes.png")
    assert bad.photo_id is None
    assert "format" in bad.detail


def test_ingest_generates_identity_free_ids_and_asset_names(
    sighting_folder: Path,
    gallery: GalleryData,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Product ids and exported crop names carry no elephant identity."""
    monkeypatch.setattr(
        ingest, "_make_analyzer", lambda dataset, cache_root: _FakeAnalyzer()
    )
    monkeypatch.setattr(ingest, "_folder_dataset", lambda *a, **k: _FakeDataset())
    monkeypatch.setattr(overlays, "annotate_photo", lambda image, analysis: image)
    monkeypatch.setattr(overlays, "annotate_ear_crop", lambda image, ear: image)

    result = ingest.ingest_sighting(
        sighting_folder, tmp_path / "work", gallery, tmp_path / "cache"
    )

    analyzed = [photo for photo in result.photos if photo.status == "analyzed"]
    assert analyzed, "expected at least one analyzed photo"
    for photo in analyzed:
        assert photo.photo_id is not None
        assert photo.photo_id.startswith("P-")
        assert "Zola" not in photo.photo_id
        assert photo.date == "2019-03-03"
        for ear in photo.ears:
            assert ear.crop_path is not None
            assert "Zola" not in Path(ear.crop_path).name
            assert photo.photo_id in Path(ear.crop_path).name
    for photo_id in result.photo_ids:
        assert "Zola" not in photo_id
    assert len(result.row_geometry) == len(result.photo_ids)
    for geometry in result.row_geometry:
        assert geometry["clean_crop_path"] is not None
        assert "Zola" not in Path(geometry["clean_crop_path"]).name
        assert geometry["contour"], "analyzed rows should carry a contour"
