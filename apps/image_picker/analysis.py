"""Ear-candidate generation for the matching image picker.

Candidates come from the production :class:`PhotoAnalyzer` so the picker's
body detection, feature-on-body filtering, and usable-ear selection are the
exact same code path as the rest of the project. Each accepted anchored ear
becomes one ranked candidate scored by its ``quality`` prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from elephant_id.coding.ears.anchored_ear import AnchoredEar
from elephant_id.coding.photo_analyzer import PhotoAnalyzer
from elephant_id.constants import DEFAULT_CACHE_ROOT
from elephant_id.dataset import Dataset
from elephant_id.domain import Photo, Sighting
from elephant_id.image import BgrImage
from elephant_id.image.transforms import apply_crop


def encode_crop_jpeg(
    image: BgrImage,
    crop_xyxy: tuple[float, float, float, float],
) -> bytes:
    """Crop an ear region and encode it as JPEG bytes.

    Raises:
        RuntimeError: If OpenCV fails to encode the crop.
    """
    ok, encoded = cv2.imencode(".jpg", apply_crop(image, crop_xyxy))
    if not ok:
        raise RuntimeError("Could not encode ear crop")
    return encoded.tobytes()


@dataclass(frozen=True)
class EarCandidate:
    """One anchored-ear crop candidate for a photo and side."""

    candidate_id: str
    side: str
    identity: str
    sighting_id: str
    sighting_date: str
    photo_identifier: str
    image_path: str
    crop_xyxy: tuple[float, float, float, float]
    quality: float
    confidence: float

    def to_json(self, *, picked: bool = False) -> dict:
        """Return the JSON representation consumed by the frontend."""
        return {
            "candidateId": self.candidate_id,
            "side": self.side,
            "photoIdentifier": self.photo_identifier,
            "quality": self.quality,
            "picked": picked,
        }


@dataclass(frozen=True)
class SightingCandidates:
    """A sighting's ear candidates, pooled across its photos and split by side.

    ``left`` and ``right`` are each ranked by quality score descending.
    """

    sighting_id: str
    sighting_date: str
    left: tuple[EarCandidate, ...]
    right: tuple[EarCandidate, ...]

    def qualifies(self, threshold: float) -> bool:
        """Whether both sides have a candidate scoring above ``threshold``."""
        return (
            bool(self.left)
            and self.left[0].quality > threshold
            and bool(self.right)
            and self.right[0].quality > threshold
        )

    def side(self, side: str) -> tuple[EarCandidate, ...]:
        """Return the ranked candidates for one side."""
        return self.left if side == "left" else self.right


class CandidateAnalyzer:
    """Generate ranked ear candidates using the production photo analyzer."""

    def __init__(
        self,
        dataset: Dataset,
        cache_root: Path = Path(DEFAULT_CACHE_ROOT),
    ) -> None:
        """Wrap a cache-backed :class:`PhotoAnalyzer`."""
        self._analyzer = PhotoAnalyzer(dataset=dataset, cache_root=cache_root)

    def analyze_sighting(self, sighting: Sighting) -> SightingCandidates:
        """Pool one sighting's ear candidates and rank each side by quality."""
        left: list[EarCandidate] = []
        right: list[EarCandidate] = []
        for photo in sighting.photos:
            for index, ear in enumerate(self._ears(photo)):
                candidate = self._candidate(sighting, photo, ear, index)
                (left if candidate.side == "left" else right).append(candidate)
        left.sort(key=lambda candidate: candidate.quality, reverse=True)
        right.sort(key=lambda candidate: candidate.quality, reverse=True)
        return SightingCandidates(
            sighting_id=sighting.sighting_id,
            sighting_date=sighting.sighting_date.isoformat(),
            left=tuple(left),
            right=tuple(right),
        )

    def _ears(self, photo: Photo) -> list[AnchoredEar]:
        """Return the anchored ears the production analyzer accepts for a photo."""
        analysis = self._analyzer.analyze(photo)
        if analysis is None:
            return []
        return analysis["shared_data"]["ears"]

    def _candidate(
        self,
        sighting: Sighting,
        photo: Photo,
        ear: AnchoredEar,
        index: int,
    ) -> EarCandidate:
        """Build one candidate from an anchored ear."""
        return EarCandidate(
            candidate_id=f"{photo.identifier}__{ear.side}__{index}",
            side=ear.side,
            identity=sighting.elephant_name,
            sighting_id=sighting.sighting_id,
            sighting_date=sighting.sighting_date.isoformat(),
            photo_identifier=photo.identifier,
            image_path=photo.image_path.as_posix(),
            crop_xyxy=tuple(float(value) for value in ear.xyxy),
            quality=ear.quality,
            confidence=ear.confidence,
        )
