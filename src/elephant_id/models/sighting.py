from dataclasses import dataclass
from .photo import Photo
from datetime import datetime

@dataclass(frozen=True, slots=True)
class Sighting:
    """
    A sighting of an elephant
    """
    elephant_name: str
    sighting_date: datetime
    sighting_id: str
    photos: tuple[Photo, ...]
    
    def __post_init__(self) -> None:
        if not self.elephant_name:
            raise ValueError(f"Elephant name is empty: {self.elephant_name}")
        if not self.sighting_date:
            raise ValueError(f"Sighting date is empty: {self.sighting_date}")
        if not self.sighting_id:
            raise ValueError(f"Sighting id is empty: {self.sighting_id}")
        if not self.sighting_id == f"{self.elephant_name}_{self.sighting_date.strftime('%Y-%m-%d')}":
            raise ValueError(f"Sighting id does not match elephant name and sighting date: {self.sighting_id} != {self.elephant_name}_{self.sighting_date.strftime('%Y-%m-%d')}")
        if not self.photos:
            raise ValueError(f"At least one photo is required: {self.photos}")
        
        _filenames = set()
        for photo in self.photos:
            if photo.filename in _filenames:
                raise ValueError(f"Photo {photo.filename} is duplicated")
            _filenames.add(photo.filename)
            if photo.sighting_id != self.sighting_id:
                raise ValueError(f"Photo {photo.filename} has sighting_id {photo.sighting_id}, expected {self.sighting_id}")
            if photo.elephant_name != self.elephant_name:
                raise ValueError(f"Photo {photo.filename} has elephant name {photo.elephant_name}, expected {self.elephant_name}")
        
    def __len__(self) -> int:
        return len(self.photos)