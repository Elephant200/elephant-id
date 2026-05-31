"""Elephant ID: identify individual African elephants from photographs.

This package turns a sighting (a folder of photos of one elephant) into a
structured, SEEK-coded identification record. It is organised into:

* :mod:`elephant_id.domain` -- core immutable data models (``Photo``,
  ``Sighting``, ``SeekCode``).
* :mod:`elephant_id.dataset` -- on-disk access to the SEEK dataset.
* :mod:`elephant_id.ai` -- cached wrappers around the segmentation, keypoint,
  gender, and age models.
* :mod:`elephant_id.coding` -- feature extraction and SEEK-code generation.
* :mod:`elephant_id.cache`, :mod:`elephant_id.image_utils`,
  :mod:`elephant_id.visualize` -- supporting utilities.
"""
