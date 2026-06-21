"""Ear field analyzer."""

from elephant_id.coding.ears.anchored_ear import AnchoredEar
from elephant_id.coding.ears.tear_profile import tear_profile
from elephant_id.domain import Photo


class EarFieldAnalyzer:
    """Analyze all anchored ears prepared for one photo."""

    def __init__(self) -> None:
        ...

    def analyze(self, photo: Photo, shared_data: dict) -> list[dict]:
        """Return ear diagnostics plus tear and hole feature lists."""
        ears: list[AnchoredEar] = shared_data["ears"] # TODO: after adding typing, remove this type here
        if not ears:
            return []

        ear_data = []
        for ear in ears:
            contour = ear.resampled_contour(1024)
            profile = tear_profile(
                contour,
                ear.area,
                ear.side,
                ear.original_anchor_points,
            )
            ear_data.append({
                "ear": ear,
                "ear_side": ear.side,
                "tear_profile": profile,
                "tears": ..., # TODO: list of tear objects
            })

        return ear_data
