"""Ear field analyzer."""

from elephant_id.coding.ears.contour import AnchoredEar
from elephant_id.coding.ears.tear_profile import compute_tear_profile
from elephant_id.domain import Photo


class EarFieldAnalyzer:
    """Analyze all anchored ears prepared for one photo."""

    def __init__(self) -> None:
        ...

    def analyze(self, photo: Photo, shared_data: dict) -> dict:
        """Return ear diagnostics plus tear and hole feature lists."""
        ears: list[dict] = []
        tears: list[dict] = []
        holes: list[dict] = []
        for ear_index, ear in enumerate(shared_data["ears"]):
            ear_evidence = self._analyze_ear(photo, ear, ear_index)
            ears.append(ear_evidence)
        return {
            "ears": ears,
            "tears": tears,
            "holes": holes,
        }

    def _analyze_ear(
        self,
        photo: Photo,
        ear: AnchoredEar,
        ear_index: int,
    ) -> dict:
        """Return traceable evidence for one anchored ear."""
        profile = compute_tear_profile(ear.resampled_contour())
        return {
            "ear": ear,
            "ear_index": ear_index,
            "photo_identifier": photo.identifier,
            "side": ear.side,
            "tear_profile": profile.profile,
            "scale": profile.scale,
        }
