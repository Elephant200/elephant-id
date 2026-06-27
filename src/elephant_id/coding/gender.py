"""Gender field analyzer."""

from elephant_id.ai.gender import GenderService
from elephant_id.domain import Photo


class GenderFieldAnalyzer:
    """Run the gender model on the masked body and shape its output."""

    def __init__(self, gender_service: GenderService) -> None:
        self.gender_service = gender_service

    def analyze(self, photo: Photo, shared_data: dict) -> dict:
        return self.gender_service.run(photo, shared_data["body"].rle_mask)
