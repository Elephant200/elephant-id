from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, slots=True)
class Photo:
    """
    Metadata for one photo of an elephant
    """
    filename: str           # Unique identifier; consists of name, sighting date, and sequential number separated by underscores. Does not include extension.
    image_path: Path        # Relative to dataset root
    elephant_name: str      # Unique elephant name
    sighting_id: str        # Unique sighting identifier; consists of elephant name and sighting date separated by underscore.

    def __post_init__(self) -> None:
        if not self.filename:
            raise ValueError(f"Filename is empty: {self.filename}")
        if not self.elephant_name:
            raise ValueError(f"Elephant name is empty: {self.elephant_name}")
        if not self.sighting_id:
            raise ValueError(f"Sighting id is empty: {self.sighting_id}")
        if self.image_path.is_absolute():
            raise ValueError(f"Image path must be relative: {self.image_path}")
        if self.filename != self.image_path.stem:
            raise ValueError(f"Filename does not match image path: {self.filename} != {self.image_path.stem}")
        if not self.filename.startswith(self.sighting_id + "_"):
            raise ValueError(f"Filename does not start with sighting id: {self.filename} does not start with {self.sighting_id}_")
        if not self.sighting_id.startswith(self.elephant_name + "_"):
            raise ValueError(f"Sighting id does not start with elephant name: {self.sighting_id} does not start with {self.elephant_name}_")

    def __str__(self) -> str:
        return f"{self.filename}"