"""Pre-warm the SAM3 cache for every photo in the dataset."""

import time
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from elephant_id.ai import Sam3Service
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo
from elephant_id.log import configure_logging

SAM3_PRESETS = ("features", "body")


def with_exponential_backoff[T](
    func: Callable[..., T],
    *args: object,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    **kwargs: object,
) -> T:
    """Call ``func`` with exponential backoff on transient failures."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("exponential backoff exhausted without raising")


def run_sam3_preset(
    sam3: Sam3Service,
    photo: Photo,
    preset: str,
    *,
    max_retries: int = 5,
    initial_delay: float = 1.0,
) -> Exception | None:
    """Run one SAM3 preset with backoff.

    Returns:
        The final exception if all retries failed, otherwise ``None``.
    """
    try:
        with_exponential_backoff(
            sam3.run,
            photo,
            preset,
            max_retries=max_retries,
            initial_delay=initial_delay,
        )
    except Exception as exc:
        return exc
    return None


if __name__ == "__main__":
    load_dotenv()
    configure_logging(level="WARNING")

    dataset = Dataset(
        dataset_root=Path("dataset/elephants-alive/coded"),
        metadata_path=Path("dataset/elephants-alive/images.csv"),
    )

    sam3 = Sam3Service(dataset=dataset)

    all_photos = list(dataset.iter_photos())
    failures: list[tuple[str, str, str]] = []
    succeeded = 0
    interrupted = False

    try:
        for photo in tqdm(all_photos):
            for preset in SAM3_PRESETS:
                error = run_sam3_preset(sam3, photo, preset)
                if error is None:
                    succeeded += 1
                    continue
                message = str(error)
                failures.append((photo.identifier, preset, message))
                logger.error(
                    f"SAM3 {preset} failed for {photo.identifier} "
                    f"after exponential backoff: {message}"
                )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Interrupted by user; reporting progress so far.")

    total_runs = len(all_photos) * len(SAM3_PRESETS)
    print(
        f"SAM3 cache warm-up finished: {succeeded}/{total_runs} runs succeeded, "
        f"{len(failures)} failed"
    )
    if failures:
        logger.error(f"Failed SAM3 runs ({len(failures)}):")
        for identifier, preset, message in failures:
            logger.error(f"  {identifier} [{preset}]: {message}")

    if interrupted:
        raise SystemExit(130)
