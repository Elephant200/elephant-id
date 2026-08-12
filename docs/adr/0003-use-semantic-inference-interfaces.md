# Use semantic inference interfaces

## Decision

Analysis depends on the semantic capabilities of ear localization and segmentation and ear landmark detection, not on particular model architectures. Model implementations are replaceable behind those interfaces, allowing SAM3 to be replaced by a composite detector and segmentation network, or the current landmark model by another neural network, without changing tear-profile extraction.
