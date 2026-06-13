"""Tusk field analyzer."""

from elephant_id.domain import Photo


class TuskFieldAnalyzer:
    """Infer tusk presence and side from tusk detections."""

    def __init__(self) -> None:
        ...

    def analyze(self, photo: Photo, shared_data: dict) -> dict:
        """Return tusk evidence for the prepared photo."""
