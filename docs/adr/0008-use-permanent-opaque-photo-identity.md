# Use permanent opaque photo identity and separate photo storage

**Status:** amends [ADR 0005](0005-separate-retrieval-evaluation-from-implementation.md) and [ADR 0006](0006-cache-immutable-expensive-producers.md)

AlphaPhant uses UUIDv4 photo and sighting IDs throughout the system. A Photo carries only its permanent photo ID and parent sighting ID. Dataset owns identity-aware metadata and a PhotoStore; the PhotoStore resolves a Photo to immutable original encoded bytes without exposing known-elephant resolution. Catalog matchers receive neutral Photo and SightingEarPair values plus the PhotoStore, never Dataset.

Cache producer namespaces use readable keys beginning with the permanent photo UUID and followed by actual dependent inputs. A producer name carries model and configuration identity.
