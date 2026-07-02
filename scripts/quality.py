"""Explore heuristic quality metrics for segmented elephant ear crops."""

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from elephant_id.coding import PhotoAnalyzer
from elephant_id.coding.ears import AnchoredEar
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.image import BgrImage
from elephant_id.log import configure_logging

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATE_DIR = REPO_ROOT / "outputs/ear_segmentation"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "outputs/ear_segmentation_filtered"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/ear_quality"
DEFAULT_THRESHOLD = 0.99

Side = Literal["left", "right"]
MetricRow = dict[str, object]

SCORE_WEIGHTS = {
    "flatness_score": 0.35,
    "aspect_score": 0.25,
    "detail_score": 0.25,
    "exposure_score": 0.15,
}

CSV_FIELDS = [
    "file_name",
    "photo_identifier",
    "side",
    "in_reference",
    "overall_score",
    "flatness_score",
    "aspect_score",
    "detail_score",
    "exposure_score",
    "crop_width",
    "crop_height",
    "crop_aspect_height_width",
    "detail",
    "dark_fraction",
    "bright_fraction",
    "bbox_width",
    "bbox_height",
    "mask_fill",
    "root_offset",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Score elephant ear crop quality and compare against filtered crops."
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=DEFAULT_CANDIDATE_DIR,
        help="Directory of ear crops to score.",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Directory of already-filtered good ear crops to compare against.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Overall score threshold treated as passing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of candidate crops to score. Zero means all crops.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for quality outputs and selected image copies.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for the metrics CSV. Defaults to output-dir/quality_scores.csv.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Log level for service output. The quality summary is always printed.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    return parser.parse_args()


def parse_ear_crop_name(path: Path) -> tuple[str, Side]:
    """Parse a saved ear crop filename into its photo identifier and side."""
    photo_identifier, separator, side = path.stem.rpartition("_")
    if separator == "" or side not in {"left", "right"}:
        raise ValueError(f"Expected ear crop name ending in _left/_right: {path.name}")
    return photo_identifier, side


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a numeric value into a closed interval."""
    return min(upper, max(lower, value))


def ramp(value: float, low: float, high: float) -> float:
    """Return a linear 0-1 score over a low-to-high interval."""
    if high <= low:
        raise ValueError("ramp high must be greater than low")
    return clamp((value - low) / (high - low))


def band_score(
    value: float,
    low_zero: float,
    low_full: float,
    high_full: float,
    high_zero: float,
) -> float:
    """Return a trapezoid score with a full-credit center band."""
    if not low_zero < low_full <= high_full < high_zero:
        raise ValueError("Invalid band score thresholds")
    if value < low_full:
        return ramp(value, low_zero, low_full)
    if value <= high_full:
        return 1.0
    return 1.0 - ramp(value, high_full, high_zero)


def weighted_geometric_mean(values: dict[str, float]) -> float:
    """Return a weighted geometric mean for component scores."""
    weighted_log_sum = 0.0
    weight_sum = 0.0
    for name, value in values.items():
        weight = SCORE_WEIGHTS[name]
        weighted_log_sum += weight * math.log(max(value, 1e-6))
        weight_sum += weight
    return math.exp(weighted_log_sum / weight_sum)


def read_crop(path: Path) -> BgrImage:
    """Read a saved ear crop as a BGR image."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read ear crop: {path}")
    return image


def downscale_long_side(gray_image: np.ndarray, long_side: int = 800) -> np.ndarray:
    """Downscale a grayscale image to a stable long side for focus metrics."""
    height, width = gray_image.shape
    scale = long_side / max(height, width)
    if scale >= 1.0:
        return gray_image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(gray_image, size, interpolation=cv2.INTER_AREA)


def measure_crop_pixels(image: BgrImage) -> dict[str, float | int]:
    """Measure crop-level pixel quality signals."""
    crop_height, crop_width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized = downscale_long_side(gray)
    laplacian_variance = float(cv2.Laplacian(normalized, cv2.CV_64F).var())

    return {
        "crop_width": crop_width,
        "crop_height": crop_height,
        "crop_aspect_height_width": crop_height / crop_width,
        "detail": math.log1p(laplacian_variance),
        "dark_fraction": float(np.mean(normalized < 30)),
        "bright_fraction": float(np.mean(normalized > 235)),
    }


def measure_ear_geometry(ear: AnchoredEar) -> dict[str, float]:
    """Measure geometry signals from an anchored ear mask."""
    x1, y1, x2, y2 = ear.xyxy
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    if bbox_width <= 0.0 or bbox_height <= 0.0:
        raise ValueError(f"Invalid anchored ear bounds: {ear.xyxy}")

    upper_anchor, lower_anchor = ear.anchor_points
    root_center_x = (upper_anchor[0] + lower_anchor[0]) / 2.0
    box_center_x = (x1 + x2) / 2.0

    return {
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "mask_fill": ear.area / (bbox_width * bbox_height),
        "root_offset": abs(root_center_x - box_center_x) / bbox_width,
    }


def score_metrics(metrics: dict[str, float | int]) -> dict[str, float]:
    """Score each independent quality component."""
    fill_score = ramp(float(metrics["mask_fill"]), low=0.52, high=0.66)
    root_score = ramp(float(metrics["root_offset"]), low=0.28, high=0.43)
    flatness_score = math.sqrt(fill_score * root_score)

    aspect_score = band_score(
        float(metrics["crop_aspect_height_width"]),
        low_zero=1.0,
        low_full=1.18,
        high_full=1.45,
        high_zero=1.85,
    )
    detail_score = ramp(float(metrics["detail"]), low=3.5, high=5.2)
    clipped_fraction = (
        float(metrics["dark_fraction"]) + float(metrics["bright_fraction"])
    )
    exposure_score = 1.0 - ramp(clipped_fraction, low=0.08, high=0.35)

    scores = {
        "flatness_score": flatness_score,
        "aspect_score": aspect_score,
        "detail_score": detail_score,
        "exposure_score": exposure_score,
    }
    scores["overall_score"] = weighted_geometric_mean(scores)
    return scores


def anchored_ears_for_photo(
    analyzer: PhotoAnalyzer,
    photo: Photo,
) -> list[AnchoredEar]:
    """Return usable anchored ears for a photo through the project services."""
    body_detections = analyzer.sam3.run(photo, "body")
    feature_detections = analyzer.sam3.run(photo, "features")
    if not body_detections or not feature_detections:
        return []

    body = analyzer._choose_body(photo, body_detections)
    if body is None:
        return []

    features_on_body = analyzer._features_on_body(body, feature_detections)
    _, ears, _ = analyzer._group_features(photo, features_on_body)
    ears = analyzer._choose_usable_ears(photo, ears)
    return analyzer._anchor_ears(photo, ears)


def choose_ear_for_side(ears: list[AnchoredEar], side: Side) -> AnchoredEar:
    """Choose the largest anchored ear matching the requested side."""
    matching_ears = [ear for ear in ears if ear.side == side]
    if not matching_ears:
        raise ValueError(f"No anchored {side} ear found")
    return max(matching_ears, key=lambda ear: ear.area)


def score_crop(
    path: Path,
    dataset: Dataset,
    analyzer: PhotoAnalyzer,
    reference_names: set[str],
    ear_cache: dict[str, list[AnchoredEar]],
) -> MetricRow:
    """Measure and score one saved ear crop."""
    photo_identifier, side = parse_ear_crop_name(path)
    photo = dataset.get_photo(photo_identifier)
    if photo_identifier not in ear_cache:
        ear_cache[photo_identifier] = anchored_ears_for_photo(analyzer, photo)

    ear = choose_ear_for_side(ear_cache[photo_identifier], side)
    metrics = measure_ear_geometry(ear) | measure_crop_pixels(read_crop(path))
    scores = score_metrics(metrics)

    return {
        "file_name": path.name,
        "path": path,
        "photo_identifier": photo_identifier,
        "side": side,
        "in_reference": path.name in reference_names,
        **metrics,
        **scores,
    }


def iter_candidate_paths(candidate_dir: Path, limit: int) -> list[Path]:
    """Return candidate crop paths in deterministic order."""
    paths = sorted(candidate_dir.glob("*.jpg"))
    if limit > 0:
        return paths[:limit]
    return paths


def numeric(row: MetricRow, field_name: str) -> float:
    """Read a numeric value from a flat metric row."""
    return float(row[field_name])


def summarize_results(results: list[MetricRow], threshold: float) -> None:
    """Print aggregate metric and reference-filter comparison summaries."""
    if not results:
        print("No crops were scored")
        return

    passed = [row for row in results if numeric(row, "overall_score") >= threshold]
    reference = [row for row in results if row["in_reference"]]
    passed_reference = [row for row in passed if row["in_reference"]]
    passed_non_reference = [row for row in passed if not row["in_reference"]]

    print(f"Scored {len(results)} crops")
    print(
        f"Existing filtered crops in scored set: {len(reference)} "
        f"({len(reference) / len(results):.1%})"
    )
    print(
        f"Heuristic pass threshold {threshold:.3f}: {len(passed)} crops "
        f"({len(passed) / len(results):.1%})"
    )
    if reference:
        print(
            f"Filtered crops retained: {len(passed_reference)} / {len(reference)} "
            f"({len(passed_reference) / len(reference):.1%})"
        )
    print(
        f"Non-filtered crops passing: {len(passed_non_reference)} / "
        f"{len(results) - len(reference)}"
    )
    print("")
    print("Threshold sweep")
    for sweep_threshold in (
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.92,
        0.94,
        0.96,
        0.98,
        0.99,
        0.995,
        0.999,
    ):
        sweep_passed = [
            row for row in results if numeric(row, "overall_score") >= sweep_threshold
        ]
        sweep_reference = [row for row in sweep_passed if row["in_reference"]]
        retained = len(sweep_reference) / len(reference) if reference else 0.0
        print(
            f"  {sweep_threshold:.3f}: pass {len(sweep_passed):4d} "
            f"({len(sweep_passed) / len(results):5.1%}), "
            f"retain filtered {len(sweep_reference):3d}/{len(reference):3d} "
            f"({retained:5.1%})"
        )
    print("")

    score_fields = (
        "flatness_score",
        "aspect_score",
        "detail_score",
        "exposure_score",
        "overall_score",
    )
    for field_name in score_fields:
        values = np.array([numeric(row, field_name) for row in results])
        quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        print(
            f"{field_name} quantiles p05/p25/p50/p75/p95: "
            f"{quantiles[0]:.3f}, {quantiles[1]:.3f}, {quantiles[2]:.3f}, "
            f"{quantiles[3]:.3f}, {quantiles[4]:.3f}"
        )
    print("")

    for field_name in score_fields:
        worst = sorted(results, key=lambda row: numeric(row, field_name))[:8]
        names = ", ".join(
            f"{row['file_name']}={numeric(row, field_name):.3f}" for row in worst
        )
        print(f"Lowest {field_name}: {names}")


def csv_value(value: object) -> object:
    """Format floats consistently for CSV output."""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_csv(results: list[MetricRow], output_path: Path) -> None:
    """Write per-crop metrics and scores to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in results:
            writer.writerow({field: csv_value(row[field]) for field in CSV_FIELDS})


def selected_dir_name(threshold: float) -> str:
    """Return a folder name for crops selected at a threshold."""
    threshold_text = f"{threshold:.3f}".replace(".", "_")
    return f"selected_threshold_{threshold_text}"


def copy_selected_images(
    results: list[MetricRow],
    output_dir: Path,
    threshold: float,
) -> int:
    """Copy crops whose overall score passes the threshold."""
    selected_dir = output_dir / selected_dir_name(threshold)
    selected_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for row in results:
        if numeric(row, "overall_score") < threshold:
            continue
        source_path = row["path"]
        if not isinstance(source_path, Path):
            raise TypeError(f"Expected row path to be a Path: {source_path!r}")
        shutil.copy2(source_path, selected_dir / str(row["file_name"]))
        copied += 1
    return copied


def main() -> None:
    """Score candidate ear crops and compare them with existing filtered crops."""
    args = parse_args()
    load_dotenv()
    configure_logging(args.log_level)

    dataset = Dataset(
        dataset_root=REPO_ROOT / "dataset/elephants-alive/coded",
        metadata_path=REPO_ROOT / "dataset/elephants-alive/images.csv",
    )
    analyzer = PhotoAnalyzer(dataset=dataset)

    candidate_paths = iter_candidate_paths(args.candidate_dir, args.limit)
    reference_names = {path.name for path in args.reference_dir.glob("*.jpg")}
    results: list[MetricRow] = []
    skipped: dict[str, int] = {}
    ear_cache: dict[str, list[AnchoredEar]] = {}

    for path in tqdm(
        candidate_paths,
        desc="Scoring ear crops",
        disable=args.no_progress,
    ):
        try:
            results.append(
                score_crop(
                    path=path,
                    dataset=dataset,
                    analyzer=analyzer,
                    reference_names=reference_names,
                    ear_cache=ear_cache,
                )
            )
        except Exception as exc:
            reason = str(exc).split(":", maxsplit=1)[0]
            skipped[reason] = skipped.get(reason, 0) + 1

    summarize_results(results, threshold=args.threshold)
    if skipped:
        print("")
        print("Skipped crops")
        for reason, count in sorted(skipped.items(), key=lambda item: item[1], reverse=True):
            print(f"  {count:4d} {reason}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or args.output_dir / "quality_scores.csv"
    write_csv(results, output_csv)
    copied = copy_selected_images(results, args.output_dir, args.threshold)
    print(f"Wrote quality metrics to {output_csv}")
    print(
        f"Copied {copied} selected images to "
        f"{args.output_dir / selected_dir_name(args.threshold)}"
    )


if __name__ == "__main__":
    main()
