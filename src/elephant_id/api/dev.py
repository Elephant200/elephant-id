"""Development diagnostics used by the desktop dev pages."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from elephant_id.ai.detection import Detection
from elephant_id.api import ingest
from elephant_id.api.profiles import plot_profile
from elephant_id.constants import TEAR_PROFILE_BINS
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.matching import TearMatch, TearMatcher, tear_mass
from elephant_id.visualize import (
    plot_aligned_tear_profiles,
    plot_tear_profile,
    plot_tear_profile_geometry,
    tear_profile_ymax,
    visualize_predictions,
)

SIDE_SUFFIX_PATTERN = re.compile(r"_(left|right)$")
DEV_SAM3_PRESETS = ("body", "features")


@dataclass(frozen=True)
class DevPhoto:
    """One uploaded development photo indexed as a temporary dataset."""

    identifier: str
    date: str
    side_filter: str | None
    original_filename: str
    path: Path
    work_dir: Path
    dataset: Dataset
    photo: Photo


@dataclass(frozen=True)
class DevTearProfile:
    """Raw tear-profile data extracted from one development upload."""

    side: str
    profile: np.ndarray
    mass: float


@dataclass(frozen=True)
class DevTearMatch:
    """Display-ready aligned tear match data for one shared ear side."""

    side: str
    image_a_profile: np.ndarray
    image_b_profile: np.ndarray
    match: TearMatch
    image_a_mass: float
    image_b_mass: float


def prepare_upload(
    filename: str | None,
    data: bytes,
    data_dir: Path,
    work_parent: Path | None = None,
) -> DevPhoto:
    """Store one upload and return a cache-keyed dataset photo."""
    original_filename = filename or "upload.jpg"
    suffix = Path(original_filename).suffix.lower() or ".jpg"
    if suffix not in ingest.IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {suffix}")
    if not data:
        raise ValueError("Empty upload")

    identifier, side_filter, match = _identifier_from_filename(original_filename, data)
    work_dir = (work_parent or data_dir / "dev") / identifier
    image_path = work_dir / f"{identifier}{suffix}"
    work_dir.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(data)

    dataset = ingest._folder_dataset(
        work_dir,
        work_dir / "metadata",
        {image_path: match},
    )
    if dataset is None:
        raise ValueError(f"Could not create development dataset for {identifier}")
    photo = dataset.get_photo(identifier)
    return DevPhoto(
        identifier=identifier,
        date=match["date"],
        side_filter=side_filter,
        original_filename=original_filename,
        path=image_path,
        work_dir=work_dir,
        dataset=dataset,
        photo=photo,
    )


def sam3_overlay(data: bytes, filename: str | None, data_dir: Path, cache_root: Path) -> dict:
    """Run SAM3 body and feature presets for one uploaded image."""
    from elephant_id.ai.sam3 import Sam3Service

    upload = prepare_upload(filename, data, data_dir)
    service = Sam3Service(dataset=upload.dataset, cache_root=cache_root)
    image = upload.dataset.read_image(upload.photo)
    overlays = []
    for preset in DEV_SAM3_PRESETS:
        cache_key = service.cache_key(upload.photo)
        cache = service.cache_managers[preset]
        was_cached = cache.exists(cache_key)
        detections = service.run(upload.photo, preset)
        overlay_path = _write_image(
            visualize_predictions(image, detections),
            upload.work_dir / "sam3",
            f"{preset}.jpg",
        )
        overlays.append(
            {
                "preset": preset,
                "cache_key": cache_key,
                "cache_status": "cache hit" if was_cached else "computed",
                "overlay_path": str(overlay_path),
                "detection_count": len(detections),
                "class_counts": _class_counts(detections),
                "detections": [detection.to_dict() for detection in detections],
            }
        )
    return _base_result(upload) | {"overlays": overlays}


def tear_profile(data: bytes, filename: str | None, data_dir: Path, cache_root: Path) -> dict:
    """Run only the prerequisite ear path and tear-profile extraction."""
    upload = prepare_upload(filename, data, data_dir)
    image = upload.dataset.read_image(upload.photo)
    analysis = _extract_tear_profiles(upload, cache_root)
    ears = []
    for ear_data, extracted in zip(analysis["ear_data"], analysis["profiles"], strict=True):
        diagnostic_path = _write_tear_profile_figure(
            image,
            ear_data,
            upload.work_dir / "tear_profile",
            f"{extracted.side}.png",
        )
        ears.append(
            {
                "side": extracted.side,
                "diagnostic_path": str(diagnostic_path),
                "profile": plot_profile(extracted.profile),
                "mass": extracted.mass,
            }
        )
    if not ears:
        side_note = f" for {upload.side_filter} ear" if upload.side_filter else ""
        raise ValueError(f"No usable tear profile found{side_note}")
    return _base_result(upload) | {"ears": ears}


def tear_match_pair(
    image_a_data: bytes,
    image_a_filename: str | None,
    image_b_data: bytes,
    image_b_filename: str | None,
    data_dir: Path,
    cache_root: Path,
) -> dict:
    """Extract and match same-side tear profiles from two uploaded images.

    Raises:
        ValueError: If either upload has no usable tear profile, or the two
            uploads share no usable ear side.
    """
    run_id = hashlib.sha1(image_a_data + b"\0" + image_b_data).hexdigest()[:12]
    run_dir = data_dir / "dev" / "tear_match" / run_id
    image_a = prepare_upload(image_a_filename, image_a_data, data_dir, run_dir / "a")
    image_b = prepare_upload(image_b_filename, image_b_data, data_dir, run_dir / "b")

    profiles_a = _profiles_by_side(_extract_tear_profiles(image_a, cache_root)["profiles"])
    profiles_b = _profiles_by_side(_extract_tear_profiles(image_b, cache_root)["profiles"])
    shared_sides = [side for side in ("left", "right") if side in profiles_a and side in profiles_b]
    if not shared_sides:
        raise ValueError("Images share no usable ear side; this won't work")

    matcher = TearMatcher()
    matches = []
    for side in shared_sides:
        profile_a = profiles_a[side]
        profile_b = profiles_b[side]
        aligned_a, aligned_b, match = matcher.align_pair(profile_a.profile, profile_b.profile)
        matches.append(
            DevTearMatch(
                side=side,
                image_a_profile=aligned_a,
                image_b_profile=aligned_b,
                match=match,
                image_a_mass=profile_a.mass,
                image_b_mass=profile_b.mass,
            )
        )

    graph_path = _write_tear_match_figure(matches, run_dir, "aligned_profiles.png")
    return {
        "image_a": _base_result(image_a),
        "image_b": _base_result(image_b),
        "shared_sides": shared_sides,
        "matches": [_match_result(match) for match in matches],
        "aligned_graph_path": str(graph_path),
    }


def photo_analysis(data: bytes, filename: str | None, data_dir: Path, cache_root: Path) -> dict:
    """Run the full analyzer and render a large diagnostic panel."""
    from elephant_id.coding.photo_analyzer import PhotoAnalyzer

    upload = prepare_upload(filename, data, data_dir)
    analyzer = PhotoAnalyzer(dataset=upload.dataset, cache_root=cache_root)
    image = upload.dataset.read_image(upload.photo)
    analysis = analyzer.analyze(upload.photo)
    if analysis is None:
        raise ValueError("No usable elephant evidence found in the photo")

    photo_result, _ = ingest._photo_result_from_analysis(
        upload.path,
        upload.identifier,
        upload.identifier,
        upload.date,
        analysis,
        upload.dataset,
        upload.work_dir,
    )
    diagnostic_path = _write_analyzer_dashboard(
        analysis,
        upload.identifier,
        image,
        upload.work_dir / "photo_analysis",
        "dashboard.png",
    )
    return _base_result(upload) | {
        "photo": photo_result.to_dict(),
        "diagnostic_path": str(diagnostic_path),
    }


def _extract_tear_profiles(upload: DevPhoto, cache_root: Path) -> dict[str, Any]:
    """Return raw tear profiles and source ear data for one upload."""
    from elephant_id.coding.photo_analyzer import PhotoAnalyzer

    analyzer = PhotoAnalyzer(dataset=upload.dataset, cache_root=cache_root)
    analysis = _analyze_tear_profiles_only(analyzer, upload.photo)
    ear_data = []
    profiles = []
    for item in analysis["ears"]:
        ear = item["ear"]
        if upload.side_filter is not None and ear.side != upload.side_filter:
            continue
        profile = np.asarray(item["tear_profile"].profile, dtype=np.float64)
        if profile.shape != (TEAR_PROFILE_BINS,):
            logger.warning(f"Unexpected profile shape for {upload.identifier}: {profile.shape}")
            continue
        ear_data.append(item)
        profiles.append(
            DevTearProfile(
                side=ear.side,
                profile=profile,
                mass=float(tear_mass(profile)[0]),
            )
        )
    if not profiles:
        side_note = f" for {upload.side_filter} ear" if upload.side_filter else ""
        raise ValueError(f"No usable tear profile found{side_note}")
    return {"ear_data": ear_data, "profiles": profiles}


def _identifier_from_filename(
    filename: str,
    data: bytes,
) -> tuple[str, str | None, re.Match[str]]:
    """Return an identifier, optional side filter, and parse match."""
    stem = Path(filename).stem
    side_filter = None
    side_match = SIDE_SUFFIX_PATTERN.search(stem)
    if side_match is not None:
        side_filter = side_match.group(1)
        stem = stem[: side_match.start()]

    match = ingest.PHOTO_STEM_PATTERN.match(stem)
    if match is not None:
        return stem, side_filter, match

    digest = hashlib.sha1(data).hexdigest()[:8]
    generated = f"Lab{digest}_{datetime.now(UTC).date().isoformat()}_01"
    generated_match = ingest.PHOTO_STEM_PATTERN.match(generated)
    if generated_match is None:
        raise ValueError(f"Generated invalid development identifier: {generated}")
    return generated, side_filter, generated_match


def _analyze_tear_profiles_only(analyzer: Any, photo: Photo) -> dict[str, Any]:
    """Run the analyzer stages needed to reach tear-profile extraction."""
    body_detections = analyzer.sam3.run(photo, "body")
    feature_detections = analyzer.sam3.run(photo, "features")
    if not body_detections or not feature_detections:
        raise ValueError("SAM3 found no body/features for this photo")

    body = analyzer._choose_body(photo, body_detections)
    if body is None:
        raise ValueError("Could not choose a single body for this photo")
    features_on_body = analyzer._features_on_body(body, feature_detections)
    trunks, ears, tusks = analyzer._group_features(photo, features_on_body)
    usable_ears = analyzer._choose_usable_ears(photo, ears)
    anchored_ears = analyzer._anchor_ears(photo, usable_ears)
    ear_evidence = analyzer.ear_analyzer.analyze(photo, {"ears": anchored_ears})
    return {
        "shared_data": {
            "body": body,
            "trunks": trunks,
            "ears": anchored_ears,
            "tusks": tusks,
        },
        "ears": ear_evidence,
    }


def _write_image(image: np.ndarray, directory: Path, filename: str) -> Path:
    """Write an image under a diagnostics directory."""
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"Could not write diagnostic image: {output_path}")
    return output_path


def _write_tear_profile_figure(
    image: np.ndarray,
    ear_data: dict,
    directory: Path,
    filename: str,
) -> Path:
    """Render one ear crop and tear profile as a PNG."""
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename
    figure = plt.figure(figsize=(10, 4), facecolor="white")
    try:
        axes = figure.subplots(1, 2)
        plot_tear_profile_geometry(axes[0], image, ear_data["ear"], ear_data["tear_profile"])
        plot_tear_profile(axes[1], ear_data["tear_profile"])
        figure.tight_layout(pad=0.4)
        figure.savefig(output_path, format="png", dpi=150)
    finally:
        plt.close(figure)
    return output_path


def _profiles_by_side(profiles: list[DevTearProfile]) -> dict[str, DevTearProfile]:
    """Return at most one extracted profile per side."""
    by_side = {}
    for profile in profiles:
        if profile.side in by_side:
            logger.warning(f"Multiple {profile.side} profiles found; using the first")
            continue
        by_side[profile.side] = profile
    return by_side


def _write_tear_match_figure(
    matches: list[DevTearMatch],
    directory: Path,
    filename: str,
) -> Path:
    """Render stacked aligned tear-profile matches as a PNG."""
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename
    profile_rows = np.vstack(
        [profile for match in matches for profile in (match.image_a_profile, match.image_b_profile)]
    )
    y_max = tear_profile_ymax(profile_rows)
    figure_height = max(5.5, 4.8 * len(matches))
    figure = plt.figure(figsize=(13, figure_height), facecolor="white")
    try:
        axes = figure.subplots(len(matches), 1, squeeze=False)
        for axis, match in zip(axes.ravel(), matches, strict=True):
            plot_aligned_tear_profiles(
                axis,
                match.image_b_profile,
                match.image_a_profile,
                candidate_label=f"Image B {match.side} ear",
                color="#2f7f70" if match.side == "left" else "#9b5c30",
                y_max=y_max,
                shift_fraction=match.match.shift_fraction,
                shift_bins=match.match.shift_bins,
                stretch=match.match.stretch,
                penalty=match.match.penalty,
                overlap_score=match.match.overlap_score,
                score=match.match.score,
                ylabel="tear depth / R",
            )
            axis.plot([], [], color="none", label=f"Image A mass {match.image_a_mass:.1f}")
            axis.plot([], [], color="none", label=f"Image B mass {match.image_b_mass:.1f}")
        figure.suptitle("Aligned same-side tear profile matches", fontsize=15, fontweight="bold")
        figure.tight_layout(rect=(0, 0, 1, 0.97), pad=1.2)
        figure.savefig(output_path, format="png", dpi=170)
    finally:
        plt.close(figure)
    return output_path


def _match_result(match: DevTearMatch) -> dict[str, float | int | str]:
    """Return JSON-ready score metadata for one dev tear match."""
    return {
        "side": match.side,
        "score": float(match.match.score),
        "distance": float(match.match.distance),
        "overlap_score": float(match.match.overlap_score),
        "shift_bins": int(match.match.shift_bins),
        "shift_degrees": float(match.match.shift_fraction * 180.0),
        "stretch": float(match.match.stretch),
        "penalty": float(match.match.penalty),
        "image_a_mass": float(match.image_a_mass),
        "image_b_mass": float(match.image_b_mass),
    }


def _write_analyzer_dashboard(
    analysis: dict,
    identifier: str,
    image: np.ndarray,
    directory: Path,
    filename: str,
) -> Path:
    """Render the full analyzer dashboard as a PNG."""
    from apps.visualization.analyzer_render import dashboard_png

    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename
    output_path.write_bytes(dashboard_png(analysis, identifier, image))
    return output_path


def _class_counts(detections: list[Detection]) -> dict[str, int]:
    """Return detection counts by class name."""
    counts: dict[str, int] = {}
    for detection in detections:
        counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
    return counts


def _base_result(upload: DevPhoto) -> dict[str, str | None]:
    """Return common JSON fields for every dev diagnostic."""
    return {
        "identifier": upload.identifier,
        "side_filter": upload.side_filter,
        "file_name": upload.original_filename,
        "photo_path": str(upload.path),
    }
