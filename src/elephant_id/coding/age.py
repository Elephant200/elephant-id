"""Age field analyzer."""

from elephant_id.ai import AgeService
from elephant_id.domain import Photo


class AgeFieldAnalyzer:
    """Run the age model on the masked body and shape its output."""

    def __init__(self, age_service: AgeService) -> None:
        self.age_service = age_service

    def analyze(self, photo: Photo, shared_data: dict) -> dict:
        return self.age_service.run(photo, shared_data["body"].rle_mask)
