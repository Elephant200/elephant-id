"""Tests for the Alphaphant matching engine."""

from pathlib import Path

import numpy as np
import pytest

from elephant_id.api.engine import MatchingEngine
from elephant_id.api.gallery import GalleryData
from elephant_id.constants import TEAR_PROFILE_BINS


def bump_profile(center_bin: int, depth: float, rng: np.random.Generator) -> np.ndarray:
    """Return a profile with one Gaussian tear bump plus small noise."""
    bins = np.arange(TEAR_PROFILE_BINS, dtype=np.float64)
    profile = depth * np.exp(-((bins - center_bin) ** 2) / (2 * 12.0**2))
    return profile + rng.uniform(0.0, 0.01, TEAR_PROFILE_BINS)


@pytest.fixture
def synthetic_gallery() -> GalleryData:
    """Three elephants, two photos per side, with distinct tear positions."""
    rng = np.random.default_rng(7)
    centers = {"Amber": 150, "Bumi": 320, "Cira": 520}
    profiles: list[np.ndarray] = []
    photo_ids: list[str] = []
    identities: list[str] = []
    sides: list[str] = []
    dates: list[str] = []
    for identity, center in centers.items():
        for side in ("left", "right"):
            for index in range(2):
                profiles.append(bump_profile(center + 10 * index, 0.25, rng))
                photo_ids.append(f"{identity}_2020-01-0{index + 1}_{index:02d}_{side}")
                identities.append(identity)
                sides.append(side)
                dates.append(f"2020-01-0{index + 1}")
    return GalleryData(
        profiles=np.vstack(profiles),
        photo_ids=tuple(photo_ids),
        identities=tuple(identities),
        sides=tuple(sides),
        dates=tuple(dates),
        crop_paths=tuple([None] * len(photo_ids)),
    )


@pytest.fixture
def engine(synthetic_gallery: GalleryData, tmp_path: Path) -> MatchingEngine:
    """Engine over the synthetic gallery with a temp pairwise cache."""
    return MatchingEngine(synthetic_gallery, tmp_path / "pairwise.npy")


def test_rank_puts_matching_elephant_first(engine: MatchingEngine) -> None:
    rng = np.random.default_rng(11)
    query = np.vstack(
        [bump_profile(325, 0.22, rng), bump_profile(318, 0.28, rng)]
    )
    ranked = engine.rank(query, ["left", "right"], ["q_left", "q_right"], top_n=3)

    assert ranked[0].identity == "Bumi"
    assert ranked[0].score >= ranked[1].score
    assert {evidence.side for evidence in ranked[0].evidence} == {"left", "right"}
    assert 0.0 <= ranked[0].confidence <= 1.0
    for candidate in ranked:
        for evidence in candidate.evidence:
            assert evidence.strength in {"strong", "moderate", "weak"}
            assert len(evidence.query_profile) > 0
            assert len(evidence.gallery_profile) > 0


def test_rank_excludes_exact_photo_twins(engine: MatchingEngine) -> None:
    rng = np.random.default_rng(3)
    query = bump_profile(150, 0.25, rng)[None, :]
    photo_id = "Amber_2020-01-01_00_left"

    ranked = engine.rank(query, ["left"], [photo_id], top_n=6)

    for candidate in ranked:
        for evidence in candidate.evidence:
            assert evidence.gallery_photo_id != photo_id


def test_rank_validates_input(engine: MatchingEngine) -> None:
    with pytest.raises(ValueError):
        engine.rank(np.zeros((0, TEAR_PROFILE_BINS)), [], [], top_n=3)
    with pytest.raises(ValueError):
        engine.rank(np.zeros((1, TEAR_PROFILE_BINS)), ["left", "right"], ["a"], top_n=3)


def test_extend_files_new_elephant(engine: MatchingEngine) -> None:
    rng = np.random.default_rng(5)
    new_profiles = np.vstack(
        [bump_profile(620, 0.3, rng), bump_profile(628, 0.3, rng)]
    )
    engine.extend(
        new_profiles,
        ["left", "left"],
        "Dara",
        "2021-05-05",
        ["Dara_2021-05-05_00", "Dara_2021-05-05_01"],
        [None, None],
    )

    assert engine.has_identity("Dara")
    assert engine.profile_count == 14
    ranked = engine.rank(
        bump_profile(622, 0.28, rng)[None, :], ["left"], ["query"], top_n=4
    )
    assert ranked[0].identity == "Dara"


def test_pairwise_cache_reused_and_rebuilt_when_stale(
    synthetic_gallery: GalleryData, tmp_path: Path
) -> None:
    cache_path = tmp_path / "pairwise.npy"
    first = MatchingEngine(synthetic_gallery, cache_path)
    matrix = np.load(cache_path)
    assert matrix.shape == (12, 12)

    np.save(cache_path, np.zeros((3, 3)))
    rebuilt = MatchingEngine(synthetic_gallery, cache_path)
    assert np.load(cache_path).shape == (12, 12)
    assert first.profile_count == rebuilt.profile_count


def test_catalog_summarizes_elephants(engine: MatchingEngine) -> None:
    catalog = engine.catalog()
    assert [entry["name"] for entry in catalog] == ["Amber", "Bumi", "Cira"]
    assert catalog[0]["photo_count"] == 4
    assert catalog[0]["side_counts"] == {"left": 2, "right": 2}

    detail = engine.elephant_detail("Amber")
    assert len(detail["photos"]) == 4
    with pytest.raises(KeyError):
        engine.elephant_detail("Nobody")
