"""Score ears and write outputs/curation/index.html.

Run `uv run python scripts/curation/score.py` for the fixed pilot, or add `--all`
for the archive. Edit rubric.md and rerun; successful grades reuse CacheManager.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from queue import PriorityQueue
from typing import Any
from uuid import UUID

import cv2
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from elephant_id.cache import CacheManager
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.image import BgrImage, decode_image
from elephant_id.image.boxes import BoundingBox
from elephant_id.inference import Detection
from elephant_id.inference.landmarks import (
    CachedEarLandmarkDetector,
    YoloEarLandmarkDetector,
)
from elephant_id.log import configure_logging
from elephant_id.preparation.ear import prepare_ear

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUTPUTS = ROOT / "outputs/curation"
METADATA = ROOT / "dataset/elephants-alive/images.csv"
ARCHIVE = ROOT / "dataset/elephants-alive/coded"
HIGH_QUALITY = ROOT / "outputs/high_quality/manifest.csv"
ANNOTATIONS = Path("/Users/alex/Downloads/30997/annotations/instances_default.json")
MODEL = "gemini-3.8-flash"
PILOT_PER_GROUP = 16
SEED = 42
ATTEMPTS = 3
TRANSIENT_ATTEMPTS = 8
MAX_BACKOFF_S = 60.0
REQUESTS_PER_KEY = 2
SIDE_PROCESSES = os.cpu_count() or 8
PREPARATION_THREADS = 2 * SIDE_PROCESSES


WORKSPACES = (
    (0, "eleid-api-key-1", "gemini"),
    (1, "eleid-api-key-2", "gemini"),
    (2, "eleid-api-key-3", "gemini"),
    (3, "eleid-api-key-4", "gemini"),
    (4, "eleid-api-key-5", "gemini"),
    (5, "eleid-api-key-6", "gemini"),
    (6, "eleid-api-key-7", "gemini"),
    (7, "eleid-api-key-8", "gemini"),
    (8, "eleid-api-key-9", "gemini"),
    (9, "eleid-api-key-10", "gemini"),
    (10, "eleid-api-key-11", "gemini"),
    (11, "eleid-api-key-12", "gemini"),
    (12, "eleid-api-key-13", "gemini"),
    (13, "eleid-api-key-14", "gemini"),
    (14, "eleid-api-key-15", "gemini"),
    (15, "eleid-api-key-16", "gemini"),
)
WORKFLOW_REVISION = "roboflow-google-gemini-1"


MAX_ASPECT_RATIO = 2.5
SLUG = "ear-quality"
ENUMS = {
    "contour_visibility": {"none", "partial", "mostly", "complete"},
    "contour_detail": {"unreadable", "limited", "clear", "excellent"},
    "upper_anchor": {"not_visible", "uncertain", "clear"},
    "lower_anchor": {"not_visible", "uncertain", "clear"},
    "pose_limitation": {"none", "mild", "moderate", "severe"},
}
FACTORS = {
    "occlusion",
    "vegetation",
    "truncation",
    "blur",
    "folding",
    "insufficient_detail",
    "foreshortening",
}
EAR_FIELDS = set(ENUMS) | {"reid_quality", "limiting_factors", "main_limitation"}


def validate_assessment(value: object) -> dict[str, Any]:
    """Validate exact fields and types; raise ValueError for malformed output."""
    if not isinstance(value, dict) or set(value) != {"status", "ear", "reason"}:
        raise ValueError("assessment_fields")
    if value["status"] not in ("assessed", "no_visible_ear", "unresolved"):
        raise ValueError("assessment_status")
    ear = value["ear"]
    texts = [value["reason"]]
    if value["status"] == "assessed":
        if not isinstance(ear, dict) or set(ear) != EAR_FIELDS:
            raise ValueError("ear_fields")
        for field, choices in ENUMS.items():
            if not isinstance(ear[field], str) or ear[field] not in choices:
                raise ValueError(f"ear_{field}")
        if type(ear["reid_quality"]) is not int or not 1 <= ear["reid_quality"] <= 10:
            raise ValueError("ear_reid_quality")
        factors = ear["limiting_factors"]
        if not isinstance(factors, list) or any(
            not isinstance(f, str) or f not in FACTORS for f in factors
        ):
            raise ValueError("ear_limiting_factors")
        if len(factors) != len(set(factors)):
            raise ValueError("duplicate_limiting_factors")
        texts.append(ear["main_limitation"])
    elif ear is not None:
        raise ValueError("assessment_status_ear")
    if any(not isinstance(t, str) or not 1 <= len(t.strip()) <= 500 for t in texts):
        raise ValueError("assessment_text")
    return value


def parse_response(response: object) -> dict[str, Any]:
    """Find exactly one valid assessment in the workflow response envelope."""
    pending, found = [response], []
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            with suppress(ValueError):
                pending.append(
                    json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip()))
                )
        elif isinstance(value, dict):
            if "status" in value and "ear" in value:
                found.append(validate_assessment(value))
            else:
                pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    if len(found) != 1:
        raise ValueError("response_missing_or_ambiguous_assessment")
    return found[0]


@dataclass(frozen=True)
class EarTarget:
    """One padded ear crop's readable key, geometry, and source detection."""

    key: str
    box: tuple[int, int, int, int]
    detection: dict[str, Any]
    source_box: BoundingBox

    @property
    def aspect_ratio(self) -> float:
        """Return the native unpadded box's long/short ratio."""
        x1, y1, x2, y2 = self.source_box.as_tuple()
        return max(x2 - x1, y2 - y1) / min(x2 - x1, y2 - y1)

    @property
    def landmark_key(self) -> str:
        """Mirror `CachedEarLandmarkDetector`'s key for the unpadded source box."""
        photo_id = self.key.split("__", 1)[0]
        return f"{photo_id}__crop_" + "_".join(map(str, self.source_box.as_tuple()))


def ear_targets(
    photo: Photo, detections: list[dict[str, Any]], width: int, height: int
) -> tuple[EarTarget, ...]:
    """Derive padded ear crops from cached detections without decoding pixels."""
    boxes: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for detection in detections:
        if detection.get("class_name") != "ear":
            continue
        x1, y1, x2, y2 = (float(detection[k]) for k in ("x1", "y1", "x2", "y2"))
        px, py = (x2 - x1) * 0.15, (y2 - y1) * 0.15
        box = BoundingBox.from_float(
            (x1 - px, y1 - py, x2 + px, y2 + py), image_width=width, image_height=height
        )
        boxes.setdefault(box.as_tuple(), detection)
    targets = []
    for box in sorted(boxes):
        detection = boxes[box]
        source_box = BoundingBox.from_float(
            tuple(float(detection[k]) for k in ("x1", "y1", "x2", "y2")),
            image_width=width,
            image_height=height,
        )
        key = f"{photo.photo_id}__crop_" + "_".join(map(str, box))
        targets.append(EarTarget(key, box, detection, source_box))
    return tuple(targets)


def encode_crop(image: BgrImage, box: tuple[int, int, int, int]) -> bytes:
    """Encode one padded crop exactly as graded; raise ValueError on JPEG failure."""
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    ratio = min(1024 / max(crop.shape[:2]), 1.0)
    size = tuple(max(1, round(n * ratio)) for n in crop.shape[1::-1])
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError("jpeg_encoding_failed")
    return encoded.tobytes()


def image_score(ears: list[dict[str, Any]]) -> int | None:
    """Use the best assessed ear; missing or filtered evidence has no score."""
    scores = [ear_score(e) for e in ears if ear_score(e) is not None]
    if scores:
        return max(scores)
    if ears and all(
        e.get("assessment", {}).get("status") == "no_visible_ear" for e in ears
    ):
        return 1
    return None


def summaries(records: list[dict[str, Any]], annotated: set[str]) -> dict[str, Any]:
    """Calculate threshold inventory without treating failures as poor images."""
    ears = [ear for record in records for ear in record.get("ears", [])]
    groups = {}
    for group in ("high_quality", "random"):
        rows = [r for r in records if r["group"] == group]
        scores = [r["image_score"] for r in rows if r.get("image_score") is not None]
        groups[group] = {
            "images": len(rows),
            "scored": len(scores),
            "scores": sorted(scores),
            "median": statistics.median(scores) if scores else None,
            "mean": statistics.mean(scores) if scores else None,
            "unresolved": sum(r["status"] == "unresolved" for r in rows),
        }
    sightings: dict[str, dict[str, Any]] = {}
    for record in records:
        sighting = sightings.setdefault(
            record["sighting_id"],
            {
                "identity": record["identity"],
                "left": [],
                "right": [],
            },
        )
        for item in record.get("ears", []):
            score = ear_score(item)
            if score is not None and item.get("side") in ("left", "right"):
                sighting[item["side"]].append(
                    {
                        "photo_id": record["photo_id"],
                        "score": score,
                        "input_key": item["input_key"],
                    }
                )
    for sighting in sightings.values():
        for side in ("left", "right"):
            candidates = sorted(
                sighting[side],
                key=lambda e: (-e["score"], e["photo_id"], e["input_key"]),
            )
            sighting[side] = [
                e for e in candidates if e["score"] >= candidates[0]["score"] - 1
            ]
        sighting["paired_sighting_quality"] = (
            min(sighting[s][0]["score"] for s in ("left", "right"))
            if sighting["left"] and sighting["right"]
            else None
        )
    thresholds = {}
    for threshold in (6, 7, 8, 9):
        qualifying = [r for r in records if (r.get("image_score") or 0) >= threshold]
        pairs = [
            s
            for s in sightings.values()
            if (s["paired_sighting_quality"] or 0) >= threshold
        ]
        counts = Counter(s["identity"] for s in pairs)
        thresholds[str(threshold)] = {
            "qualifying_images": len(qualifying),
            "paired_sightings": len(pairs),
            "identities_with_pairs": {
                str(n): sum(c >= n for c in counts.values()) for n in (2, 3, 4)
            },
            "annotation_filename_overlap_images": sum(
                r["photo_id"] in annotated for r in qualifying
            ),
        }
    return {
        "scope": "Selected input set only; no population extrapolation or cohort allocation. Counts use known evidence, including good ears from partially unresolved photos.",
        "unresolved_breakdown": {
            "images_without_score": sum(r.get("image_score") is None for r in records),
            "input_error_images": sum("error" in r for r in records),
            "request_or_validation_errors": sum("error" in e for e in ears),
            "visual_unresolved_ears": sum(
                e.get("assessment", {}).get("status") == "unresolved" for e in ears
            ),
            "assessed_ears_without_side": sum(
                ear_score(e) is not None and e.get("side") == "unknown" for e in ears
            ),
        },
        "recorded_images": len(records),
        "groups": groups,
        "thresholds": thresholds,
        "filtered_ears": sum(e.get("status") == "filtered" for e in ears),
        "sightings": sightings,
    }


STYLE = """
body{font:16px system-ui;margin:24px;background:#eee;color:#222}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px}
article{background:white;padding:16px;border-radius:8px}
img{width:100%;height:420px;object-fit:contain}pre{white-space:pre-wrap}
table{border-collapse:collapse;background:white;margin:16px 0}
th,td{padding:8px 14px;text-align:left;border-bottom:1px solid #ddd}
small{overflow-wrap:anywhere} .filtered{opacity:.7}
"""


def ear_score(item: dict[str, Any]) -> int | None:
    """Read a valid ear score without mapping missing evidence to zero."""
    return (item.get("assessment", {}).get("ear") or {}).get("reid_quality")


def ear_status(item: dict[str, Any]) -> str:
    """Describe a scored, filtered, or failed crop for human review."""
    return (
        item.get("filter_reason")
        or item.get("error")
        or item.get("assessment", {}).get("status", "unresolved")
    )


def gallery_page(title: str, body: str) -> str:
    """Wrap escaped report content in a standalone responsive page."""
    return f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{html.escape(title)}</title><style>{STYLE}</style><h1>{html.escape(title)}</h1>{body}'


def html_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Render a table, escaping every value and displaying missing scores clearly."""
    heading = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(c)) if c is not None else '—'}</td>" for c in row
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><tr>{heading}</tr>{body}</table>"


def ear_card(record: dict[str, Any], item: dict[str, Any]) -> str:
    """Render one exact crop input with its score, status, and assessment."""
    ear = item.get("assessment", {}).get("ear") or {}
    score = ear_score(item)
    heading = f"Ear {score if score is not None else '—'} · {item.get('side', 'unknown')} · {record['group']}"
    details = html.escape(json.dumps(item.get("assessment", item), indent=2))
    source = html.escape(item["preview"], quote=True)
    status = html.escape(ear_status(item))
    css = "filtered" if item.get("status") == "filtered" else ""
    return f'<article class="{css}"><h3>{heading}</h3><img loading="lazy" src="{source}"><p>{html.escape(ear.get("main_limitation", status))}</p><details><summary>Assessment · {status}</summary><pre>{details}</pre></details><small>{item["input_key"]}</small></article>'


def write_gallery(
    directory: Path, records: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    """Render exact crop inputs, scores, exclusions, and the saved rubric."""
    cards = []
    for record in records:
        if not record.get("ears"):
            cards.append(
                f"<article><h3>No crop</h3><p>{html.escape(record.get('error', 'unresolved'))}</p><small>{record['photo_id']}</small></article>"
            )
        cards.extend(ear_card(record, item) for item in record.get("ears", []))
    rows = [
        [
            group,
            stats["scored"],
            f"{stats['mean']:.2f}" if stats["mean"] is not None else None,
        ]
        for group, stats in summary["groups"].items()
    ]
    table = html_table(["Group", "Scored photos", "Mean best-ear score"], rows)
    header = f'<p>{len(records)} photos · {summary["filtered_ears"]} crops filtered at native long/short &gt;{MAX_ASPECT_RATIO:g}. Filtered crops were not submitted.</p><p><a href="rubric.md">Exact rubric</a> · <a href="summary.json">Summary JSON</a></p>{table}<p>Group labels are local metadata, not human ratings.</p>'
    (directory / "gallery.html").write_text(
        gallery_page(
            "Ear quality evaluation", header + "<main>" + "".join(cards) + "</main>"
        )
    )


def compare(before: Path, after: Path, output: Path) -> None:
    """Compare paired crops, checking that only the grading prompt changed."""
    configurations = [
        json.loads((d / "configuration.json").read_text()) for d in (before, after)
    ]
    for field in ("model", "workspaces", "workflow_revision", "max_aspect_ratio"):
        if configurations[0][field] != configurations[1][field]:
            raise ValueError(f"Comparison differs in {field}")
    records = [json.loads((d / "results.json").read_text()) for d in (before, after)]
    if [r["photo_id"] for r in records[0]] != [r["photo_id"] for r in records[1]]:
        raise ValueError("Comparison photo sets/order differ")
    indexed = [{e["input_key"]: e for r in run for e in r["ears"]} for run in records]
    if indexed[0].keys() != indexed[1].keys():
        raise ValueError("Comparison crop sets differ")
    changes = []
    for key, first in indexed[0].items():
        second = indexed[1][key]
        if (before / first["preview"]).read_bytes() != (
            after / second["preview"]
        ).read_bytes():
            raise ValueError(f"Comparison pixels differ: {key}")
        if first.get("filter_reason") != second.get("filter_reason"):
            raise ValueError(f"Comparison filtering differs: {key}")
        old, new = ear_score(first), ear_score(second)
        changes.append(
            {
                "input_key": key,
                "before": old,
                "after": new,
                "delta": new - old if old is not None and new is not None else None,
                "before_status": ear_status(first),
                "after_status": ear_status(second),
            }
        )
    write_json(output / "comparison.json", changes)
    links = " · ".join(
        f'<a href="{html.escape(os.path.relpath(d / "gallery.html", output), quote=True)}">{label} gallery</a>'
        for label, d in (("Before", before), ("After", after))
    )
    rows = [
        [c["input_key"][:8], c["before"], c["after"], c["delta"], c["after_status"]]
        for c in changes
    ]
    table = html_table(["Photo", "Before", "After", "Change", "After status"], rows)
    body = f"<p>{links}</p><p>Identical model, workflow, photo set, crop bytes, and filter verified. One run per prompt; changes include model variability.</p>{table}"
    (output / "index.html").write_text(gallery_page("Rubric comparison", body))


def finish_reports(
    output: Path, records: list[dict[str, Any]], annotated: set[str]
) -> None:
    """Write review artifacts and link the latest compatible comparison."""
    summary = summaries(records, annotated)
    write_json(output / "summary.json", summary)
    write_gallery(output, records, summary)
    current = json.loads((output / "configuration.json").read_text())
    candidates = sorted(
        output.parent.glob("*/configuration.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    comparison = ""
    for candidate in candidates:
        previous = candidate.parent
        if previous == output:
            continue
        try:
            if (
                json.loads((candidate).read_text()).get("producer_slug")
                == current["producer_slug"]
            ):
                continue
            compare(previous, output, output / "comparison")
        except (ValueError, KeyError, FileNotFoundError):
            continue
        comparison = f'<p><a href="{output.name}/comparison/index.html">Compare with previous compatible run</a></p>'
        break
    body = f'<p><a href="{output.name}/gallery.html">Open current gallery</a> · <a href="{output.name}/summary.json">Summary</a> · <a href="{output.name}/rubric.md">Exact rubric</a></p>{comparison}'
    (output.parent / "index.html").write_text(gallery_page("Ear quality results", body))


def write_json(path: Path, value: object) -> None:
    """Write a final derived report; expensive results persist in CacheManager."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class RequestError(RuntimeError):
    """A failed Roboflow request with the server's optional retry delay."""

    def __init__(self, code: str, retry_after: float | None = None) -> None:
        """Store a loggable error code and the `Retry-After` seconds."""
        super().__init__(code)
        self.retry_after = retry_after


def is_transient(code: str) -> bool:
    """Return whether a failure can clear after the key backs off."""
    return code in ("http_408", "http_429", "network_or_timeout") or code.startswith(
        "http_5"
    )


def request_json(url: str, payload: dict[str, Any]) -> object:
    """POST a Roboflow request without logging credentials or response bodies."""
    request = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        retry_after = None
        with suppress(TypeError, ValueError):
            retry_after = max(0.0, float(error.headers.get("Retry-After")))
        raise RequestError(f"http_{error.code}", retry_after) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RequestError("network_or_timeout") from None
    except (ValueError, UnicodeError):
        raise ValueError("invalid_http_json") from None


class KeyBackoff:
    """Share one API key's backoff among all of its concurrent request slots."""

    def __init__(self) -> None:
        """Start with no failures and no pause."""
        self._lock = threading.Lock()
        self._failures = 0
        self._resume_at = 0.0

    def wait(self) -> None:
        """Sleep until the key may send another request."""
        while (delay := self._resume_at - time.monotonic()) > 0:
            time.sleep(delay)

    def succeeded(self) -> None:
        """Reset the exponential backoff after a successful request."""
        with self._lock:
            self._failures = 0

    def failed(self, retry_after: float | None) -> None:
        """Pause the key for `Retry-After` or a jittered exponential delay."""
        with self._lock:
            self._failures += 1
            delay = retry_after
            if delay is None:
                delay = random.uniform(0.5, 1.0) * min(
                    MAX_BACKOFF_S, 2.0**self._failures
                )
            self._resume_at = max(self._resume_at, time.monotonic() + delay)


def load_sample(
    metadata: list[dict[str, str]], all_images: bool
) -> list[dict[str, Any]]:
    """Use the fixed pilot, or prioritize archive elephants by distinct sightings."""
    path = OUTPUTS / "pilot-sample.json"
    if not all_images and path.exists():
        return sorted(
            json.loads(path.read_text())["photos"], key=lambda r: r["photo_id"]
        )
    by_path = {row["image_path"]: row for row in metadata}
    selected = {
        by_path[row["source_image_path"]]["photo_id"]
        for row in csv.DictReader(HIGH_QUALITY.open())
    }
    records = [
        {
            "photo_id": row["photo_id"],
            "group": "high_quality" if row["photo_id"] in selected else "random",
        }
        for row in metadata
    ]
    records.sort(key=lambda r: r["photo_id"])
    if all_images:
        sightings: dict[str, set[str]] = defaultdict(set)
        by_id = {row["photo_id"]: row for row in metadata}
        for row in metadata:
            sightings[row["name"]].add(row["sighting_id"])
        for record in records:
            record["sighting_count"] = len(sightings[by_id[record["photo_id"]]["name"]])
        records.sort(
            key=lambda r: (
                -r["sighting_count"],
                by_id[r["photo_id"]]["name"],
                by_id[r["photo_id"]]["sighting_id"],
                r["photo_id"],
            )
        )
    else:
        rng = random.Random(SEED)
        groups = [
            [r for r in records if r["group"] == group]
            for group in ("high_quality", "random")
        ]
        records = [r for group in groups for r in rng.sample(group, PILOT_PER_GROUP)]
        records.sort(key=lambda r: r["photo_id"])
        write_json(path, {"seed": SEED, "photos": records})
    return records


def annotation_ids(metadata: list[dict[str, str]], path: Path) -> set[str]:
    """Provisionally reconcile annotation filenames; do not claim verified masks."""
    if not path.exists():
        return set()
    by_stem: dict[str, list[str]] = defaultdict(list)
    for row in metadata:
        by_stem[Path(row["image_path"]).stem].append(row["photo_id"])
    matched = set()
    for image in json.loads(path.read_text())["images"]:
        stem = Path(image["file_name"]).stem.rsplit("_", 1)[0]
        ids = by_stem.get(stem, [])
        if len(ids) == 1:
            matched.add(ids[0])
    return matched


def cached_grade(cache: CacheManager, slug: str, key: str) -> dict[str, Any]:
    """Load a resumed grade without pixels; the slug already fixes preprocessing."""
    outcome: dict[str, Any] = {"cached": True, "attempts": 1, "elapsed_s": 0.0}
    try:
        return {
            "assessment": validate_assessment(cache.load(slug, key).get("assessment")),
            **outcome,
        }
    except Exception as error:
        code = str(error) if isinstance(error, ValueError) else type(error).__name__
        logger.warning("{} unresolved: {}", key[:8], code[:160])
        return {"error": code[:160], **outcome}


def grade_view(
    cache: CacheManager,
    slug: str,
    key: str,
    box: tuple[int, int, int, int],
    jpeg: bytes,
    assess: Callable[[bytes], object],
) -> dict[str, Any]:
    """Validate cached evidence or retry a live assessment; return explicit errors.

    `assess` owns request backoff, so transient failures retry without sleeping here.
    """
    started = time.monotonic()
    fingerprint = hashlib.sha256(jpeg).hexdigest()
    attempts = rate_limited = 0

    def outcome(hit: bool) -> dict[str, Any]:
        """Describe attempts, cache use, rate limits, and elapsed time."""
        return {
            "cached": hit,
            "attempts": attempts,
            "elapsed_s": round(time.monotonic() - started, 2),
            **({"rate_limited": rate_limited} if rate_limited else {}),
        }

    while True:
        attempts += 1
        hit = False
        try:
            hit = cache.exists(slug, key)
            if hit:
                record = cache.load(slug, key)
                if record.get("input_sha256") != fingerprint:
                    raise ValueError("cached_input_changed")
                assessment = validate_assessment(record.get("assessment"))
            else:
                assessment = parse_response(assess(jpeg))
                cache.save(
                    slug,
                    key,
                    {
                        "assessment": assessment,
                        "box": box,
                        "input_sha256": fingerprint,
                        "grader": f"{WORKFLOW_REVISION}/{MODEL}",
                    },
                )
            return {"assessment": assessment, **outcome(hit)}
        except Exception as error:
            code = (
                str(error)
                if isinstance(error, (ValueError, RuntimeError))
                else type(error).__name__
            )
            rate_limited += code == "http_429"
            permanent = hit or code in (
                "missing_credential",
                "http_400",
                "http_401",
                "http_403",
                "http_404",
            )
            limit = TRANSIENT_ATTEMPTS if is_transient(code) else ATTEMPTS
            if permanent or attempts >= limit:
                logger.warning("{} unresolved: {}", key[:8], code[:160])
                return {"error": code[:160], **outcome(hit)}


def log_result(
    photo_id: str, workspace: str, workflow: str, item: dict[str, Any]
) -> None:
    """Write one completed ear outcome, including skipped or failed assessments."""
    assessment = item.get("assessment", {})
    score = image_score([item])
    metadata = {
        "workflow": workflow,
        "side": item.get("side", "unknown"),
        "status": item.get("status", assessment.get("status", "unresolved")),
        **{
            key: item[key]
            for key in (
                "cached",
                "attempts",
                "rate_limited",
                "elapsed_s",
                "input_key",
                "error",
                "filter_reason",
                "side_error",
            )
            if key in item
        },
        **({"reason": assessment["reason"]} if assessment.get("reason") else {}),
    }
    logger.bind(curation=True).info(
        "{} | {} | score={} | {}",
        photo_id,
        workspace,
        score if score is not None else "unresolved",
        " | ".join(
            f"{key}={' '.join(str(value).split())}" for key, value in metadata.items()
        ),
    )


def infer_side(
    photo: Photo, detection: dict[str, Any], points: Detection, source_box: BoundingBox
) -> str:
    """Infer an ear side in a worker process; mask decoding otherwise holds the GIL."""
    return prepare_ear(
        Detection.from_dict(detection),
        points,
        source_photo=photo,
        source_box=source_box,
    ).inferred_side


class LockedYoloEarLandmarkDetector(YoloEarLandmarkDetector):
    """Serialize raw YOLO inference while leaving the outer cache concurrent."""

    def __init__(self) -> None:
        """Initialize the lazy model and its inference lock."""
        super().__init__()
        self._lock = threading.Lock()

    def detect(
        self, photo: Photo, image: BgrImage, ear_box: BoundingBox
    ) -> Detection | None:
        """Run raw inference under the model lock."""
        with self._lock:
            return super().detect(photo, image, ear_box)


def main() -> None:
    """Feed prioritized ear crops to every workspace request slot and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Score the full archive")
    parser.add_argument(
        "--per-key",
        type=int,
        default=REQUESTS_PER_KEY,
        help="Concurrent requests for each API key",
    )
    args = parser.parse_args()
    all_images = args.all
    slots = len(WORKSPACES) * args.per_key
    configure_logging(level="WARNING")
    logger.add(
        ROOT / "run.log",
        mode="w",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} {message}",
        filter=lambda record: record["extra"].get("curation", False),
    )
    load_dotenv(ROOT / ".env")
    credentials = os.environ.get("ROBOFLOW_ELEID_API_KEYS", "").split(",")
    prompt = (HERE / "rubric.md").read_text()
    slug = SLUG
    logger.bind(curation=True).info(
        "run_start mode={} producer={}", "archive" if all_images else "pilot", slug
    )
    output = OUTPUTS / ("archive" if all_images else "pilot")
    (output / "images").mkdir(parents=True, exist_ok=True)
    write_json(
        output / "configuration.json",
        {
            "model": MODEL,
            "producer_slug": slug,
            "workspaces": WORKSPACES,
            "workflow_revision": WORKFLOW_REVISION,
            "max_aspect_ratio": MAX_ASPECT_RATIO,
        },
    )
    (output / "rubric.md").write_text(prompt)
    metadata = list(csv.DictReader(METADATA.open()))
    sample = load_sample(metadata, all_images)
    dataset = Dataset(ARCHIVE, METADATA)
    cache = CacheManager(ROOT / ".cache")
    landmarks = CachedEarLandmarkDetector(LockedYoloEarLandmarkDetector(), cache)
    jobs: PriorityQueue = PriorityQueue(maxsize=2 * slots)
    pending: dict[str, int] = {}

    def finish_photo_part(photo_id: str) -> None:
        """Count a photo once every queued ear has finished."""
        with progress.get_lock():
            pending[photo_id] -= 1
            if pending[photo_id] == 0:
                del pending[photo_id]
                progress.update(1)

    def worker(route: tuple[int, str, str], backoff: KeyBackoff) -> None:
        """Use one key slot to grade the next prepared ear."""
        index, workspace, workflow = route
        api_key = credentials[index].strip() if index < len(credentials) else ""

        def assess(jpeg: bytes) -> object:
            """Submit only pixels, model, and the rubric once the key may send."""
            if not api_key:
                raise ValueError("missing_credential")
            backoff.wait()
            try:
                response = request_json(
                    f"https://serverless.roboflow.com/infer/workflows/{workspace}/{workflow}",
                    {
                        "api_key": api_key,
                        "inputs": {
                            "image": {
                                "type": "base64",
                                "value": base64.b64encode(jpeg).decode(),
                            },
                            "model": MODEL,
                            "prompt": prompt,
                        },
                    },
                )
            except RequestError as error:
                if is_transient(str(error)):
                    backoff.failed(error.retry_after)
                raise
            backoff.succeeded()
            return response

        while True:
            _, _, job = jobs.get()
            if job is None:
                return
            photo_id, key, box, jpeg, item = job
            item["workspace"] = workspace
            item["workflow"] = workflow
            try:
                item.update(grade_view(cache, slug, key, box, jpeg, assess))
            except Exception as error:
                item["error"] = type(error).__name__
            log_result(photo_id, workspace, workflow, item)
            finish_photo_part(photo_id)

    def prepare_photo(
        selected: dict[str, Any],
    ) -> tuple[dict[str, Any], list[tuple[str, str, Any, bytes, dict[str, Any]]]]:
        """Resolve cached ears without pixels and prepare only requests that remain."""
        photo_id = selected["photo_id"]
        record = dict(selected, sighting_id="", identity="", ears=[])
        requests = []
        try:
            photo = dataset.photo(UUID(photo_id))
            record.update(
                sighting_id=str(photo.sighting_id),
                identity=dataset.known_elephant_name(photo.sighting_id),
            )
            detections = cache.load("sam3-features", photo_id)["detections"]
            image: BgrImage | None = None

            def pixels() -> BgrImage:
                """Decode the original photo once, only when a step needs pixels."""
                nonlocal image
                if image is None:
                    image = decode_image(dataset.photo_store.read(photo))
                return image

            # SAM3 masks were computed on the decoded photo, so they carry its size.
            size = next(
                (d["rle_mask"]["size"] for d in detections if d.get("rle_mask")), None
            )
            height, width = size if size else pixels().shape[:2]
            targets = ear_targets(photo, detections, width, height)
            if not targets:
                record["error"] = "no_cached_ear_detections"
            for target in targets:
                preview = output / "images" / f"{target.key}.jpg"
                item: dict[str, Any] = {
                    "input_key": target.key,
                    "box": target.box,
                    "preview": f"images/{target.key}.jpg",
                    "side": "unknown",
                    "aspect_ratio": target.aspect_ratio,
                }
                record["ears"].append(item)
                jpeg = None
                if not preview.exists():
                    jpeg = encode_crop(pixels(), target.box)
                    preview.write_bytes(jpeg)
                if target.aspect_ratio > MAX_ASPECT_RATIO:
                    item.update(
                        status="filtered",
                        filter_reason=f"aspect_ratio_gt_{MAX_ASPECT_RATIO:g}",
                    )
                    log_result(photo_id, "none", "none", item)
                    continue
                try:
                    landmark_image = (
                        None
                        if cache.exists(landmarks.producer_slug, target.landmark_key)
                        else pixels()
                    )
                    points = landmarks.detect(photo, landmark_image, target.source_box)
                    if points is None:
                        item["side_error"] = "no_landmarks"
                    else:
                        item["side"] = sides.submit(
                            infer_side,
                            photo,
                            target.detection,
                            points,
                            target.source_box,
                        ).result()
                except Exception as error:
                    item["side_error"] = type(error).__name__
                if cache.exists(slug, target.key):
                    item.update(cached_grade(cache, slug, target.key))
                    log_result(photo_id, "cache", "cache", item)
                else:
                    # An existing preview holds the exact bytes this crop encodes to.
                    jpeg = jpeg if jpeg is not None else preview.read_bytes()
                    requests.append((photo_id, target.key, target.box, jpeg, item))
        except Exception as error:
            record["error"] = type(error).__name__
        if "error" in record:
            log_result(photo_id, "none", "none", {"error": record["error"]})
        return record, requests

    ordered = []
    with (
        tqdm(total=len(sample), unit="photo", desc="Grading") as progress,
        ThreadPoolExecutor(max_workers=slots) as pool,
        ProcessPoolExecutor(SIDE_PROCESSES) as sides,
    ):
        backoffs = {route: KeyBackoff() for route in WORKSPACES}
        futures = [
            pool.submit(worker, route, backoffs[route])
            for route in WORKSPACES
            for _ in range(args.per_key)
        ]
        sequence = 0
        try:
            with ThreadPoolExecutor(PREPARATION_THREADS) as preparation:
                for chunk in batched(sample, 4 * PREPARATION_THREADS):
                    for record, requests in preparation.map(prepare_photo, chunk):
                        ordered.append(record)
                        with progress.get_lock():
                            if requests:
                                pending[record["photo_id"]] = len(requests)
                            else:
                                progress.update(1)
                        for request in requests:
                            priority = -record.get("sighting_count", 0)
                            jobs.put((priority, sequence, request))
                            sequence += 1
        finally:
            for index in range(slots):
                jobs.put((float("inf"), index, None))
            for future in as_completed(futures):
                future.result()
    for record in ordered:
        record["image_score"] = image_score(record["ears"])
        record["status"] = (
            "assessed" if record["image_score"] is not None else "unresolved"
        )
        if record["ears"] and all(
            e.get("status") == "filtered" for e in record["ears"]
        ):
            record["status"] = "filtered"
    ordered.sort(key=lambda r: r["photo_id"])
    write_json(output / "results.json", ordered)
    finish_reports(output, ordered, annotation_ids(metadata, ANNOTATIONS))
    rate_limited = sum(e.get("rate_limited", 0) for r in ordered for e in r["ears"])
    logger.bind(curation=True).info(
        "run_finished photos={} per_key={} rate_limited={} report={}",
        len(ordered),
        args.per_key,
        rate_limited,
        OUTPUTS / "index.html",
    )
    print(
        f"Finished {len(ordered)} photos with {rate_limited} HTTP 429 responses. "
        f"Report: {OUTPUTS / 'index.html'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
