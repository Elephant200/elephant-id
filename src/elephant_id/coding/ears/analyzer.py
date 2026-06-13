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
        ears: list[AnchoredEar] = shared_data["ears"] # TODO: after adding typing, remove this type here

        ear_data = []
        for ear in ears:
            profile = compute_tear_profile(ear.resampled_contour(1024))
            ear_data.append({
                "ear": ear,
                "ear_side": ear.side,
                "scale": profile.scale,
                "tear_profile": profile.profile,
                "tears": ..., # TODO: list of tears
            })

        profile = compute_tear_profile(ear.resampled_contour())
        return {
            "ear": ear,
            "photo_identifier": photo.identifier,
            "side": ear.side,
            "tear_profile": profile.profile,
            "scale": profile.scale,
        }
