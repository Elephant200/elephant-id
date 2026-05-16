"""
Generates a SEEK code for one elephant
"""

from elephant_id.dataset import Dataset
from elephant_id.domain import SeekCode, Sighting


class SeekCoder:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.sam3 = ... # Should this be an argument or should it be initialized here?
        self.anchor_model = ... # Should this be an argument or should it be initialized here?
        self.gender_model = ... # Should this be an argument or should it be initialized here?
        self.age_model = ... # Should this be an argument or should it be initialized here?

    def code(self, sighting: Sighting) -> SeekCode:
        # Compute all localization data for the sighting. Where should this be done? In ai/ or coding/ ?
        pass
