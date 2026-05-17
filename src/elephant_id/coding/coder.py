"""
Generates a SEEK code for one elephant
"""

from elephant_id.coding.features import FeatureComputeService
from elephant_id.dataset import Dataset
from elephant_id.domain import SeekCode, Sighting


class SeekCoder:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.feature_compute_service = FeatureComputeService(dataset)

    def code(self, sighting: Sighting) -> SeekCode:
        all_features = []
        for photo in sighting.photos:
            all_features.append(self.feature_compute_service.compute(photo))
        # Use features to generate view and SEEK codes
