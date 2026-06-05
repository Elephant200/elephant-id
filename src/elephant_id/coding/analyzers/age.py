"""Age field analyzer."""

from elephant_id.ai import AgeService
from elephant_id.domain import Photo


class AgeAnalyzer:
    """Run the age model on the masked body and shape its output."""

    def __init__(self, age_service: AgeService) -> None:
        self.age_service = age_service

    def analyze(self, photo: Photo, prep: dict) -> dict:
        ...
