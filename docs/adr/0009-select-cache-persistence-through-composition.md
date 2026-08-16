# Select cache persistence through processor composition

**Status:** accepted; supersedes [ADR 0006](0006-cache-immutable-expensive-producers.md)

Cache policy is selected only when the processing pipeline is composed. Each deterministic processor exposes a stable `producer_id`; a thin cached decorator preserves the processor interface and identity while adding persistence through the generic CacheManager. Standard runs decorate segmentation, landmark detection, and tear-profile extraction, while parameter-tuning runs leave only the profile extractor undecorated so upstream model output remains cached. Sighting analysis and ranking contain no cache flags or cache-policy branches.
