# Cache reusable processing through composition

**Status:** accepted; supersedes [ADR 0009](0009-select-cache-persistence-through-composition.md)

Cache policy is selected only when the processing pipeline is composed. Each
settled deterministic processor exposes a stable human-readable
`producer_slug`; a thin cached decorator preserves the processor interface and
slug while adding persistence through the generic CacheManager. CacheManager
only stores JSON and has no permission modes, tuning mode, or processor
knowledge.

The useful SAM3 cache boundary is its complete multi-feature computation, not
the ear-only semantic adapter. Standard construction therefore caches the full
SAM3 feature result before filtering it to ears, caches full-image-relative ear
landmark results, and caches final AlphaTear profiles. Parameter-tuning
construction leaves only the experimental AlphaTear extractor uncached so
upstream inference remains reusable. Sighting analysis and ranking contain no
cache-policy options.
