"""Persistence decorator for tear-profile extraction."""

from elephant_id.cache import CacheManager
from elephant_id.matching.alphaphant.extraction import TearProfile, TearProfileExtractor
from elephant_id.preparation.ear import PreparedEar


class CachedTearProfileExtractor:
    """Persist output from a settled tear-profile extractor."""

    def __init__(
        self,
        inner: TearProfileExtractor,
        cache: CacheManager,
        *,
        segmentation_producer_slug: str,
        landmark_producer_slug: str,
    ) -> None:
        """Cache an extractor's output with its inference producer identities.

        Raises:
            ValueError: If the extractor has no settled producer slug.
        """
        if inner.producer_slug is None:
            raise ValueError("A cached profile extractor requires a producer slug")
        self._inner = inner
        self._cache = cache
        self._producer_slug = inner.producer_slug
        self._segmentation_producer_slug = segmentation_producer_slug
        self._landmark_producer_slug = landmark_producer_slug

    @property
    def producer_slug(self) -> str:
        """Return the wrapped extractor's settled slug."""
        return self._producer_slug

    @property
    def alpha_scales(self) -> tuple[float, ...]:
        """Return the wrapped extractor's alpha scales."""
        return self._inner.alpha_scales

    def extract(self, ear: PreparedEar) -> tuple[TearProfile, ...]:
        """Return cached tear profiles, or extract and store new ones.

        Each record contains the alpha scales and the tear profiles. Each
        stored tear profile contains its measured interval and its depths.

        Raises:
            ValueError: If a stored record has other alpha scales, in value or
                order, or does not contain one valid tear profile for each alpha scale.
        """
        crop = "_".join(map(str, ear.source_box.as_tuple()))
        key = (
            f"{ear.source_photo.photo_id}"
            f"__seg_{self._segmentation_producer_slug}"
            f"__landmarks_{self._landmark_producer_slug}"
            f"__side_{ear.inferred_side}__crop_{crop}"
        )
        alpha_scales = list(self.alpha_scales)
        record = self._cache.get_or_compute(
            self.producer_slug,
            key,
            lambda: {
                "alpha_scales": alpha_scales,
                "profiles": [
                    {
                        "start_degrees": profile.start_degrees,
                        "end_degrees": profile.end_degrees,
                        "depths": profile.depths.tolist(),
                    }
                    for profile in self._inner.extract(ear)
                ],
            },
        )
        if record.get("alpha_scales") != alpha_scales:
            raise ValueError(
                f"Cached tear profiles use alpha scales {record.get('alpha_scales')}; "
                f"the extractor uses {alpha_scales}"
            )
        profiles = record.get("profiles")
        if not isinstance(profiles, list) or len(profiles) != len(alpha_scales):
            raise ValueError("Tear-profile cache record must contain one profile for each alpha scale")
        try:
            return tuple(
                TearProfile(profile["depths"], profile["start_degrees"], profile["end_degrees"])
                for profile in profiles
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Cached tear profiles are invalid") from error
