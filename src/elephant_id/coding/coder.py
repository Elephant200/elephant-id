"""
Generates a SEEK code for a sighting of an elephant
"""

from loguru import logger

from elephant_id.coding.photo_analyzer import PhotoAnalyzer
from elephant_id.dataset import Dataset
from elephant_id.domain import Sighting


class SeekCoder:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.photo_analyzer = PhotoAnalyzer(dataset)

    def code(self, sighting: Sighting) -> dict:
        """Analyze every photo and aggregate into one sighting result.

        Returns a loose dict holding the per-photo analyses and a preview SEEK
        code. Aggregation is currently a stub.
        """
        logger.info(f"Coding sighting {sighting.sighting_id} ({len(sighting.photos)} photos)")
        photo_analyses: list[dict] = []
        for photo in sighting.photos:
            photo_analysis = self.photo_analyzer.analyze(photo)
            if photo_analysis is None:
                continue
            photo_analyses.append(photo_analysis)

        return {
            "sighting_id": sighting.sighting_id,
            "photos": photo_analyses,
            "preview_seek": ...,
            # TODO: combine age/gender, infer tusk presence and side, choose the
            # best left/right ear, select representative images, and build the
            # preview SEEK code from the combined fields.
        }
