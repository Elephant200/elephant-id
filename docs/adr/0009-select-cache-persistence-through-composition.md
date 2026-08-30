# Select cache persistence through processor composition

**Status:** superseded by [ADR 0010](0010-cache-reusable-processing-through-composition.md)

Cache policy is selected only when the processing pipeline is composed. Each deterministic processor exposes a stable `producer_id`; a thin cached decorator preserves the processor interface and identity while adding persistence through the generic CacheManager. Standard runs decorate segmentation, landmark detection, and tear-profile extraction, while parameter-tuning runs leave only the profile extractor undecorated so upstream model output remains cached. Sighting analysis and catalog matching contain no cache flags or cache-policy branches.
