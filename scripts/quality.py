"""
Exploratory script to determine a good heuristic for ear image quality.
"""

from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from elephant_id.coding import PhotoAnalyzer
from elephant_id.dataset import Dataset
from elephant_id.log import configure_logging

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    load_dotenv()
    configure_logging()

    dataset = Dataset(
        dataset_root=REPO_ROOT / "dataset/elephants-alive/coded",
        metadata_path=REPO_ROOT / "dataset/elephants-alive/images.csv",
    )
    analyzer = PhotoAnalyzer(dataset=dataset)

    for photo_path in tqdm((REPO_ROOT / "outputs/ear_segmentation_filtered").glob("*.jpg")):
        photo_identifier = "_".join(photo_path.stem.split("_")[:-1])
        photo = dataset.get_photo(photo_identifier)
        analysis = analyzer.analyze(photo)

# TODO: Consider lowering tear profile precision to 180 bins and rounding tear profile values

if __name__ == "__main__":
    main()
