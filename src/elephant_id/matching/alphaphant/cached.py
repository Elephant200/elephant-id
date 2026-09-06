"""Persistence decorator for tear-profile extraction."""

from elephant_id.cache import CacheManager
from elephant_id.matching.alphaphant.profile import TearProfile, TearProfileExtractor
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
        """Wrap a settled extractor with its upstream processor lineage."""
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

    def extract(self, ear: PreparedEar) -> TearProfile:
        """Return cached or newly extracted tear depths."""
        crop = "_".join(map(str, ear.source_box.as_tuple()))
        key = (
            f"{ear.source_photo.photo_id}"
            f"__seg_{self._segmentation_producer_slug}"
            f"__landmarks_{self._landmark_producer_slug}"
            f"__side_{ear.inferred_side}__crop_{crop}"
        )
        record = self._cache.get_or_compute(
            self.producer_slug,
            key,
            lambda: {"depths": self._inner.extract(ear).depths.tolist()},
        )
        depths = record.get("depths")
        if not isinstance(depths, list):
            raise ValueError("Tear-profile cache record must contain a depths list")
        try:
            return TearProfile(depths)
        except (TypeError, ValueError) as error:
            raise ValueError("Cached tear-profile depths are invalid") from error
