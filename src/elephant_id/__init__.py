"""Elephant ID: identify individual African elephants from photographs.

This package turns a sighting folder into a structured, SEEK-coded
identification record. It is organised into:

* :mod:`elephant_id.domain` -- core immutable data models (``Photo``,
  ``Sighting``, ``SeekCode``).
* :mod:`elephant_id.dataset` -- on-disk access to the SEEK dataset.
* :mod:`elephant_id.ai` -- cached segmentation, keypoint, gender, and
  age model wrappers.
* :mod:`elephant_id.coding` -- SEEK-code feature extraction.
* :mod:`elephant_id.image` -- ``BgrImage`` plus box, mask, and
  pixel-transform utilities.
* :mod:`elephant_id.cache` -- JSON cache manager for model responses.
* :mod:`elephant_id.visualize` -- supporting visualization utilities.
"""
