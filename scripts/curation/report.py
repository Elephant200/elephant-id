"""Write an interim progress report for a running score.py.

Run `uv run python scripts/curation/report.py` while score.py writes run.log.
The report reads run.log and cached grades only; it never sends requests.
"""

from __future__ import annotations

import csv
import html
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from score import (
    ANNOTATIONS,
    METADATA,
    OUTPUTS,
    ROOT,
    annotation_ids,
    ear_card,
    ear_score,
    gallery_page,
    html_table,
    image_score,
    load_sample,
    summaries,
)

from elephant_id.cache import CacheManager

RECENT_WINDOW = timedelta(minutes=10)
BUCKET_MINUTES = 10
RECENT_CARDS = 60
BEST_EAR_WEIGHT = 0.2
IDENTITY_ROWS = 50


def parse_log(path: Path) -> tuple[dict[str, str], list[tuple[datetime, str, dict]]]:
    """Return run-start fields and every per-photo outcome line in log order."""
    start: dict[str, str] = {}
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        timestamp, message = datetime.fromisoformat(line[:23]), line[24:]
        if message.startswith("run_start"):
            start = dict(re.findall(r"(\w+)=(\S+)", message))
            start["time"] = line[:23]
            continue
        parts = message.split(" | ")
        if len(parts) < 3:
            continue
        fields = {"workspace": parts[1]}
        for index, part in enumerate(parts[2:], start=2):
            key, _, value = part.partition("=")
            if key == "reason":
                # Reasons are free text and may contain the field separator.
                fields[key] = " | ".join([value, *parts[index + 1 :]])
                break
            fields[key] = value
        lines.append((timestamp, parts[0], fields))
    return start, lines


def build_records(
    lines: list[tuple[datetime, str, dict]],
    cache: CacheManager,
    slug: str,
    metadata: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Rebuild score.py result records from logged outcomes and cached grades."""
    by_id = {row["photo_id"]: row for row in metadata}
    groups = {r["photo_id"]: r["group"] for r in load_sample(metadata, True)}
    records: dict[str, dict[str, Any]] = {}
    for _, photo_id, fields in lines:
        row = by_id[photo_id]
        record = records.setdefault(
            photo_id,
            {
                "photo_id": photo_id,
                "group": groups[photo_id],
                "sighting_id": row["sighting_id"],
                "identity": row["name"],
                "ears": [],
            },
        )
        if "input_key" not in fields:
            record["error"] = fields.get("error", "unresolved")
            continue
        item = {
            "input_key": fields["input_key"],
            "preview": f"images/{fields['input_key']}.jpg",
            **{
                key: fields[key]
                for key in ("side", "error", "filter_reason", "cached")
                if key in fields
            },
        }
        if fields["status"] == "filtered":
            item["status"] = "filtered"
        record["ears"].append(item)

    graded = [
        item
        for record in records.values()
        for item in record["ears"]
        if "error" not in item and item.get("status") != "filtered"
    ]

    def attach(item: dict[str, Any]) -> None:
        """Attach the cached assessment, or mark the grade as missing."""
        try:
            item["assessment"] = cache.load(slug, item["input_key"])["assessment"]
        except FileNotFoundError:
            item["error"] = "missing_cached_grade"

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(attach, graded))
    for record in records.values():
        record["image_score"] = image_score(record["ears"])
        record["status"] = (
            "assessed" if record["image_score"] is not None else "unresolved"
        )
    return list(records.values())


def speed_section(
    lines: list[tuple[datetime, str, dict]], total_photos: int
) -> tuple[str, list[str]]:
    """Summarize recent throughput, retries, errors, and the remaining time."""
    end = lines[-1][0]
    start = max(lines[0][0], end - RECENT_WINDOW)
    minutes = max((end - start).total_seconds() / 60, 1 / 60)
    new = [f for t, _, f in lines if t >= start and f.get("cached") == "False"]
    elapsed = [float(f["elapsed_s"]) for f in new]
    seen_photos = {photo_id for _, photo_id, _ in lines}
    recent_photos = {p for t, p, _ in lines if t >= start}
    photo_rate = len(recent_photos) / minutes
    remaining = total_photos - len(seen_photos)
    eta = timedelta(minutes=remaining / photo_rate) if photo_rate else None
    rows = [
        ["Photos logged", f"{len(seen_photos):,} of {total_photos:,}"],
        ["Window", f"last {minutes:.1f} min, ending {end:%H:%M:%S}"],
        ["New ears / min", f"{len(new) / minutes:.0f}"],
        ["Photos / min", f"{photo_rate:.0f}"],
        [
            "Estimated remaining",
            f"{eta.total_seconds() / 3600:.1f} h" if eta else None,
        ],
        ["Busy slots", f"{sum(elapsed) / (60 * minutes):.1f}"],
        [
            "Median / p90 elapsed_s",
            f"{statistics.median(elapsed):.1f} / {sorted(elapsed)[int(0.9 * len(elapsed))]:.1f}"
            if elapsed
            else None,
        ],
        [
            "Retried / unresolved new ears",
            f"{sum(f['attempts'] != '1' for f in new)} / {sum('error' in f for f in new)}",
        ],
        ["HTTP 429", sum(int(f.get("rate_limited", 0)) for f in new)],
    ]
    warnings = []
    if new and sum("error" in f for f in new) / len(new) > 0.02:
        warnings.append("More than 2% of recent new ears are unresolved.")
    if sum(int(f.get("rate_limited", 0)) for f in new):
        warnings.append("Recent requests received HTTP 429.")
    if end < datetime.now() - timedelta(minutes=2):
        warnings.append(f"run.log has had no new line since {end:%H:%M:%S}.")
    return html_table(["Measurement", "Value"], rows), warnings


def bucket_section(lines: list[tuple[datetime, str, dict]]) -> str:
    """Tabulate new ears, retries, and unresolved codes in fixed time buckets."""
    buckets: dict[datetime, list[dict]] = defaultdict(list)
    for timestamp, _, fields in lines:
        if fields.get("cached") == "False":
            minute = timestamp.minute - timestamp.minute % BUCKET_MINUTES
            buckets[timestamp.replace(minute=minute, second=0, microsecond=0)].append(
                fields
            )
    rows = [
        [
            f"{start:%H:%M}",
            len(ears),
            sum(f["attempts"] != "1" for f in ears),
            ", ".join(
                f"{code} {count}"
                for code, count in Counter(
                    f["error"] for f in ears if "error" in f
                ).most_common()
            )
            or "none",
        ]
        for start, ears in sorted(buckets.items())
    ]
    return html_table(["Start", "New ears", "Retried", "Unresolved"], rows)


def histogram(
    label: str, unit: str, counts: dict[int, int], bins: dict[int, str]
) -> str:
    """Render labeled bins as a count table with proportional bars."""
    peak = max(counts.values(), default=0) or 1
    heading = "".join(f"<th>{h}</th>" for h in (label, unit, ""))
    body = "".join(
        f"<tr><td>{name}</td><td>{counts.get(key, 0):,}</td>"
        f'<td><span class="bar" style="width:{240 * counts.get(key, 0) / peak:.0f}px"></span></td></tr>'
        for key, name in bins.items()
    )
    return f"<table><tr>{heading}</tr>{body}</table>"


def distribution_section(records: list[dict[str, Any]]) -> str:
    """Show assessed-ear scores, model statuses, and inferred sides."""
    ears = [item for record in records for item in record["ears"]]
    scores = Counter(ear_score(item) for item in ears if ear_score(item) is not None)
    outcomes = Counter(
        "filtered"
        if item.get("status") == "filtered"
        else item.get("error") or item.get("assessment", {}).get("status", "unresolved")
        for item in ears
    )
    sides = Counter(
        item.get("side", "unknown") for item in ears if "assessment" in item
    )
    return (
        histogram("Score", "Ears", scores, {n: str(n) for n in range(1, 11)})
        + html_table(
            ["Ear outcome", "Ears"], [[k, f"{v:,}"] for k, v in outcomes.most_common()]
        )
        + html_table(
            ["Inferred side", "Graded ears"],
            [[k, f"{v:,}"] for k, v in sides.most_common()],
        )
    )


def sighting_grade(left: int, right: int) -> float:
    """Weight the better side's best ear lightly and the weaker side's heavily."""
    grade = BEST_EAR_WEIGHT * max(left, right) + (1 - BEST_EAR_WEIGHT) * min(
        left, right
    )
    # Rounding keeps whole-number grades from falling below their bin or threshold.
    return round(grade, 6)


def sighting_section(records: list[dict[str, Any]]) -> str:
    """Grade sightings from their best left and right ears and summarize them."""
    best: dict[str, dict[str, int]] = defaultdict(dict)
    identities: dict[str, str] = {}
    for record in records:
        sides = best[record["sighting_id"]]
        identities[record["sighting_id"]] = record["identity"]
        for item in record["ears"]:
            score = ear_score(item)
            side = item.get("side")
            if score is not None and side in ("left", "right"):
                sides[side] = max(sides.get(side, 0), score)
    grades = {
        sighting: sighting_grade(sides["left"], sides["right"])
        for sighting, sides in best.items()
        if len(sides) == 2
    }
    one_side = sum(len(sides) == 1 for sides in best.values())
    counts = html_table(
        ["Sightings", "Count"],
        [
            ["Graded (both sides scored)", f"{len(grades):,}"],
            ["One side only", f"{one_side:,}"],
            ["No scored left or right ear", f"{len(best) - len(grades) - one_side:,}"],
        ],
    )
    bins = {n: f"{n}.0&ndash;{n}.9" for n in range(1, 10)} | {10: "10.0"}
    spread = histogram(
        "Grade", "Sightings", Counter(int(grade) for grade in grades.values()), bins
    )
    by_identity: dict[str, list[float]] = defaultdict(list)
    for sighting, grade in grades.items():
        by_identity[identities[sighting]].append(grade)
    thresholds = html_table(
        ["Grade ≥", "Sightings", "Identities with ≥2 / 3 / 4 such sightings"],
        [
            [
                threshold,
                f"{sum(g >= threshold for g in grades.values()):,}",
                " / ".join(
                    str(
                        sum(
                            sum(g >= threshold for g in values) >= n
                            for values in by_identity.values()
                        )
                    )
                    for n in (2, 3, 4)
                ),
            ]
            for threshold in (6, 7, 8, 9)
        ],
    )
    logged = Counter(identities.values())
    rows = [
        [
            identity,
            logged[identity],
            len(by_identity[identity]),
            f"{statistics.median(by_identity[identity]):.1f}"
            if by_identity[identity]
            else None,
            f"{statistics.mean(by_identity[identity]):.2f}"
            if by_identity[identity]
            else None,
        ]
        for identity, _ in logged.most_common(IDENTITY_ROWS)
    ]
    identity_table = html_table(
        ["Identity", "Sightings logged", "Graded", "Median grade", "Mean grade"], rows
    )
    weights = f"{BEST_EAR_WEIGHT:g} &times; better side + {1 - BEST_EAR_WEIGHT:g} &times; weaker side"
    return (
        f"<p>Grade = {weights}, from the best-scored ear on each side. "
        "Ears with an unknown side are not used. Photos still in progress count "
        "with the ears graded so far.</p>"
        f"{counts}{spread}{thresholds}"
        f"<h3>Top {len(rows)} identities by sightings logged</h3>{identity_table}"
    )


def main() -> None:
    """Write progress.html beside the running output's preview images."""
    start, lines = parse_log(ROOT / "run.log")
    if not start or not lines:
        raise SystemExit("run.log has no run_start line or no photo outcomes yet")
    slug = start["producer"]
    output = OUTPUTS / start["mode"]
    metadata = list(csv.DictReader(METADATA.open()))
    records = build_records(lines, CacheManager(ROOT / ".cache"), slug, metadata)
    summary = summaries(records, annotation_ids(metadata, ANNOTATIONS))
    total = (
        len(metadata)
        if start["mode"] == "archive"
        else len(load_sample(metadata, False))
    )

    speed, warnings = speed_section(lines, total)
    groups = html_table(
        ["Group", "Scored photos", "Mean best-ear score"],
        [
            [
                group,
                f"{s['scored']:,}",
                f"{s['mean']:.2f}" if s["mean"] is not None else None,
            ]
            for group, s in summary["groups"].items()
        ],
    )
    thresholds = html_table(
        ["Score ≥", "Images", "Paired sightings", "Identities with ≥2 / 3 / 4 pairs"],
        [
            [
                threshold,
                f"{t['qualifying_images']:,}",
                f"{t['paired_sightings']:,}",
                " / ".join(str(n) for n in t["identities_with_pairs"].values()),
            ]
            for threshold, t in summary["thresholds"].items()
        ],
    )
    by_key = {
        item["input_key"]: (record, item)
        for record in records
        for item in record["ears"]
    }
    recent = [
        by_key[f["input_key"]]
        for _, _, f in reversed(lines)
        if f.get("cached") == "False"
        and "error" not in f
        and f.get("input_key") in by_key
    ][:RECENT_CARDS]
    status = (
        "".join(f'<p class="warning">{html.escape(w)}</p>' for w in warnings)
        or '<p class="ok">No warnings.</p>'
    )
    body = (
        f"<p>Run started {start['time'][:19]} · report written {datetime.now():%Y-%m-%d %H:%M:%S} · "
        f"<code>{html.escape(slug)}</code></p>{status}"
        f"<h2>Speed</h2>{speed}<h2>New ears every {BUCKET_MINUTES} minutes</h2>{bucket_section(lines)}"
        f"<h2>Scores</h2>{distribution_section(records)}{groups}"
        f"<h2>Sighting grades (interim)</h2>{sighting_section(records)}"
        f"<h2>Threshold inventory so far</h2>{thresholds}"
        f"<h2>Latest {len(recent)} new grades</h2><main>"
        + "".join(ear_card(record, item) for record, item in recent)
        + "</main>"
    )
    page = gallery_page("Ear quality progress", body).replace(
        "</style>",
        ".bar{display:inline-block;height:12px;background:#4a7}"
        ".warning{color:#a40;font-weight:600}.ok{color:#270}</style>",
    )
    path = output / "progress.html"
    path.write_text(page)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
