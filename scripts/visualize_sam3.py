from datetime import date
from pathlib import Path
from elephant_id.ai import Sam3Service
from elephant_id.dataset import Dataset
from elephant_id.models import Photo
from elephant_id.visualization import draw_rle_mask_overlay

if __name__ == "__main__":
    dataset = Dataset(dataset_root=Path("dataset/elephants-alive/coded"), metadata_path=Path("dataset/elephants-alive/images.csv"))
    sighting = dataset.get_photo("Devin_2015-11-05_09.JPG")