"""
Generates a SEEK code for a sighting of an elephant
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
            if photo_analysis is None:
                continue

        # Use photo analysis to generate SEEK codes
        return SeekCode(
            g=...,
            a=...,
            tr=...,
            tl=...,
            ert1=...,
            erh1=...,
            ert2=...,
            erh2=...,
            elt1=...,
            elh1=...,
            elt2=...,
            elh2=...,
            xr=...,
            xl=...,
            sr=...,
            sl=...,
            sb=...,
        )
