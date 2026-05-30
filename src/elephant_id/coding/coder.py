"""
Generates a SEEK code for one elephant
"""

from elephant_id.coding.photo_analyzer import PhotoAnalyzer
from elephant_id.dataset import Dataset
from elephant_id.domain import SeekCode, Sighting


class SeekCoder:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.photo_analyzer = PhotoAnalyzer(dataset)

    def code(self, sighting: Sighting) -> SeekCode:
        for photo in sighting.photos:
            photo_analysis = self.photo_analyzer.analyze(photo)

        # Use photo analysis to generate SEEK codes
