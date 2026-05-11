from datetime import date
import os
from pathlib import Path
import time

from dotenv import load_dotenv

from elephant_id.ai import Sam3Service
from elephant_id.dataset import Dataset
from elephant_id.visualize import visualize_predictions

if __name__ == "__main__":
    load_dotenv()
    
    dataset = Dataset(
        dataset_root=Path("dataset/elephants-alive/coded"),
        metadata_path=Path("dataset/elephants-alive/images.csv"),
        image_cache_size=32,
    )
    sighting = dataset.get_sighting("Devin", date(2015, 11, 5))
    sam3 = Sam3Service(
        api_key=os.getenv("ROBOFLOW_API_KEY"),
        dataset=dataset,
    )
    print(sighting)
    sample_photo = sighting.photos[0]
    print(sample_photo)
    start_time = time.perf_counter()
    for image in sighting.photos:
        original_image = dataset.read_image(image)
        predictions = sam3.run(image, "features")
        visualized_image = visualize_predictions(original_image, predictions["predictions"])
        visualized_image.show()
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time} seconds ({(end_time-start_time) / len(sighting.photos)} seconds per image)")