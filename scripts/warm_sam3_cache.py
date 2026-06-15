"""Warm the SAM3 cache for every photo in the dataset."""

from __future__ import annotations

import concurrent.futures
import os
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from elephant_id.ai.sam3 import Sam3Service
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.log import configure_logging

SAM3_PRESETS = ("features", "body")
DATASET_ROOT = Path("dataset/elephants-alive/coded")
METADATA_PATH = Path("dataset/elephants-alive/images.csv")
CACHE_ROOT = Path(DEFAULT_CACHE_ROOT)
MAX_RETRIES = 5
INITIAL_DELAY = 1.0


@dataclass(frozen=True)
class RoboflowProfile:
    """Roboflow credentials for one warmup worker."""

    workspace: str
    api_key: str


@dataclass
class WarmupStats:
    """Thread-safe summary of warmup progress."""

    completed_photos: int = 0
    succeeded: int = 0
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_success(self) -> None:
        """Record one successful SAM3 preset run."""
        with self.lock:
            self.succeeded += 1

    def record_failure(self, photo: Photo, preset: str, error: Exception) -> None:
        """Record one failed SAM3 preset run."""
        with self.lock:
            self.failures.append((photo.identifier, preset, str(error)))

    def record_photo_done(self) -> None:
        """Record one fully attempted photo."""
        with self.lock:
            self.completed_photos += 1

    def snapshot(self) -> tuple[int, int, list[tuple[str, str, str]]]:
        """Return a stable copy of the current warmup status."""
        with self.lock:
            return self.completed_photos, self.succeeded, list(self.failures)


def load_profiles() -> list[RoboflowProfile]:
    """Load ordered Roboflow profiles used for SAM3 warmup."""
    api_keys = [
        api_key.strip()
        for api_key in os.getenv("ROBOFLOW_ELEID_API_KEYS", "").split(",")
        if api_key.strip()
    ]
    if not api_keys:
        raise ValueError("ROBOFLOW_ELEID_API_KEYS is not set")

    return [
        RoboflowProfile(workspace=f"eleid-api-key-{index}", api_key=api_key)
        for index, api_key in enumerate(api_keys, start=1)
    ]


def run_sam3_preset(
    sam3: Sam3Service,
    photo: Photo,
    preset: str,
) -> Exception | None:
    """Run one SAM3 preset and return the final error, if any.

    Transient failures are retried with exponential backoff.
    Deterministic failures are returned immediately.
    """
    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            sam3.run(photo, preset)
            return None
        except ValueError as exc:
            return exc
        except FileNotFoundError as exc:
            return exc
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                return exc
            time.sleep(delay)
            delay *= 2

    return None  # Unreachable: the loop always returns for MAX_RETRIES >= 1.


def warm_photos(
    sam3: Sam3Service,
    photos: Sequence[Photo],
    *,
    progress: tqdm,
    stats: WarmupStats,
) -> None:
    """Warm SAM3 cache entries assigned to one Roboflow workspace."""
    for photo in photos:
        for preset in SAM3_PRESETS:
            error = run_sam3_preset(sam3, photo, preset)
            if error is None:
                stats.record_success()
            else:
                stats.record_failure(photo, preset, error)
                logger.error(
                    f"SAM3 {preset} failed for {photo.identifier} "
                    f"with workspace {sam3.runner.workspace_name}: {error}"
                )
        stats.record_photo_done()
        progress.update(1)


def split_photos(photos: Sequence[Photo], worker_count: int) -> list[list[Photo]]:
    """Split photos across workers in round-robin order."""
    return [list(photos[index::worker_count]) for index in range(worker_count)]


def summary_text(
    *,
    completed_photos: int,
    total_photos: int,
    succeeded: int,
    failures: list[tuple[str, str, str]],
    interrupted: bool,
) -> str:
    """Return a plain-text SAM3 warmup summary."""
    total_runs = total_photos * len(SAM3_PRESETS)
    prefix = "SAM3 cache warmup interrupted" if interrupted else "SAM3 cache warmup finished"
    lines = [
        (
            f"{prefix}: {completed_photos}/{total_photos} photos completed, "
            f"{succeeded}/{total_runs} runs succeeded, {len(failures)} failed"
        )
    ]
    if failures:
        lines.append(f"Failed SAM3 runs ({len(failures)}):")
        lines.extend(
            f"  {identifier} [{preset}]: {message}"
            for identifier, preset, message in failures
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Warm the SAM3 cache across all configured Roboflow workspaces."""
    load_dotenv()
    configure_logging(level="WARNING")

    profiles = load_profiles()
    photos = list(
        Dataset(dataset_root=DATASET_ROOT, metadata_path=METADATA_PATH).iter_photos()
    )
    photo_groups = split_photos(photos, worker_count=len(profiles))
    stats = WarmupStats()

    # Build one service per profile up front, in the main thread, so an invalid
    # key or unreachable workspace fails here rather than inside a worker.
    services = [
        Sam3Service(
            dataset=Dataset(dataset_root=DATASET_ROOT, metadata_path=METADATA_PATH),
            cache_root=CACHE_ROOT,
            api_key=profile.api_key,
            workspace_name=profile.workspace,
        )
        for profile in profiles
    ]

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(profiles))
    try:
        with tqdm(total=len(photos)) as progress:
            futures = [
                executor.submit(
                    warm_photos,
                    service,
                    group,
                    progress=progress,
                    stats=stats,
                )
                for service, group in zip(services, photo_groups, strict=True)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        # Workers may be blocked in API calls and cannot be cancelled, so report
        # what we have and exit hard rather than waiting for them to drain.
        completed_photos, succeeded, failures = stats.snapshot()
        print("\nInterrupted by user; reporting progress so far.", file=sys.stderr, flush=True)
        print(
            summary_text(
                completed_photos=completed_photos,
                total_photos=len(photos),
                succeeded=succeeded,
                failures=failures,
                interrupted=True,
            ),
            end="",
            flush=True,
        )
        sys.exit(130)  # KeyboardInterrupt
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    completed_photos, succeeded, failures = stats.snapshot()
    print(
        summary_text(
            completed_photos=completed_photos,
            total_photos=len(photos),
            succeeded=succeeded,
            failures=failures,
            interrupted=False,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
