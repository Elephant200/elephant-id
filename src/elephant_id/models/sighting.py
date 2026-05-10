from elephant_id.models.photo import Photo
from elephant_id.models.seek_code import SeekCode

class Sighting:
    """
    A sighting of an elephant
    """
    def __init__(self, photo: Photo, seek_code: SeekCode):
        self.photo = photo
        self.seek_code = seek_code