"""Alphaphant desktop sidecar: a local FastAPI service over the elephant_id core.

The Electron desktop app spawns this API to ingest sighting folders, rank
likely matches against the known-elephant catalog, and record review
decisions. Everything runs locally; model inference is served from the
on-disk cache plus local model weights.
"""
