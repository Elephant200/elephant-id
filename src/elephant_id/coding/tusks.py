"""Tusk field analyzer."""

import numpy as np
from loguru import logger

from elephant_id.ai import Detection
from elephant_id.domain import Photo


class TuskFieldAnalyzer:
    """Infer tusk presence and side from tusk detections."""

    def __init__(self) -> None:
        ...

    def analyze(self, photo: Photo, shared_data: dict) -> list[dict]:
        """Return tusk evidence for the prepared photo."""
        tusks: list[Detection] = shared_data["tusks"]
        if not tusks:
            return []

        if len(tusks) > 2:
            logger.warning(f"Found too many tusks in photo {photo}: {len(tusks)}")
            tusks = sorted(tusks, key=lambda t: t.area() * t.confidence, reverse=True)
            tusks = tusks[:2]

        if len(tusks) == 2:
            return [
                {
                    "side": "unknown",
                    "confidence": tusk.confidence,
                    "area": tusk.area(),
                    "x1": tusk.x1,
                    "y1": tusk.y1,
                    "x2": tusk.x2,
                    "y2": tusk.y2,
                    "rle_mask": tusk.rle_mask,
                }
                for tusk in tusks
            ]

        # Single tusk
        tusk = tusks[0]
        trunks: list[Detection] = shared_data["trunks"]
        view = shared_data["view"]
        side = view if view in ["left", "right"] else "unknown"

        if not trunks:
            logger.warning(f"No trunk found for single tusk in photo {photo}")
        else:
            trunk = trunks[0]
            trunk_mask = trunk.get_mask()
            y1 = max(0, int(tusk.y1))
            y2 = min(trunk_mask.shape[0], int(tusk.y2))
            trunk_band = trunk_mask[y1:y2, :]
            _, trunk_xs = np.where(trunk_band)

            if trunk_xs.size == 0:
                logger.warning(f"No trunk pixels overlap tusk y-range in photo {photo}")
            else:
                trunk_centroid_x = float(trunk_xs.mean())
                if view == "front":
                    tusk_mask = tusk.get_mask()
                    _, tusk_xs = np.where(tusk_mask)
                    tusk_centroid_x = float(tusk_xs.mean())
                    side = "right" if tusk_centroid_x < trunk_centroid_x else "left"
                elif view == "right":
                    side = "left" if tusk.x1 >= trunk_centroid_x else "right"
                elif view == "left":
                    side = "right" if tusk.x2 <= trunk_centroid_x else "left"
                else:
                    logger.warning(f"Unknown view {view} in photo {photo}")

        return [
            {
                "side": side,
                "confidence": tusk.confidence,
                "area": tusk.area(),
                "x1": tusk.x1,
                "y1": tusk.y1,
                "x2": tusk.x2,
                "y2": tusk.y2,
                "rle_mask": tusk.rle_mask,
            }
        ]
