from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Photo:
    """Metadata for one photo of an elephant."""

    identifier: str         # Unique identifier; consists of elephant name, sighting date, and sequential number separated by underscores. Does not include extension.
    image_path: Path        # Relative to dataset root
    elephant_name: str      # Unique elephant name
    sighting_id: str        # Unique sighting identifier; consists of elephant name and sighting date separated by underscore.

    def __post_init__(self) -> None:
        """Validate identity fields and image path."""
        if not self.identifier:
            raise ValueError(f"Identifier is empty: {self.identifier}")
        if not self.elephant_name:
            raise ValueError(f"Elephant name is empty: {self.elephant_name}")
        if not self.sighting_id:
            raise ValueError(f"Sighting id is empty: {self.sighting_id}")
        if self.image_path.is_absolute():
            raise ValueError(f"Image path must be relative: {self.image_path}")
        if ".." in self.image_path.parts:
            raise ValueError(f"Image path must not contain '..': {self.image_path}")
        if self.identifier != self.image_path.stem:
            raise ValueError(f"Identifier does not match image path stem: {self.identifier} != {self.image_path.stem}")
        if not self.identifier.startswith(self.sighting_id + "_"):
            raise ValueError(f"Identifier does not start with sighting id: {self.identifier} does not start with {self.sighting_id}_")
        if not self.sighting_id.startswith(self.elephant_name + "_"):
            raise ValueError(f"Sighting id does not start with elephant name: {self.sighting_id} does not start with {self.elephant_name}_")

    def __str__(self) -> str:
        """Return a compact, identifier-based representation."""
        return f"Photo({self.identifier})"
