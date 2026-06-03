import os
from pathlib import Path

from dotenv import load_dotenv

from elephant_id.ai import AnchorService, Sam3Service
from elephant_id.dataset import Dataset
from elephant_id.log import configure_logging

if __name__ == "__main__":
    load_dotenv()
    configure_logging()
    dataset = Dataset(
        dataset_root=Path("dataset/elephants-alive/coded"),
        metadata_path=Path("dataset/elephants-alive/images.csv"),
    )

    sam3 = Sam3Service(
        api_key=os.getenv("ROBOFLOW_API_KEY"),
        dataset=dataset,
    )

    anchor = AnchorService(
        dataset=dataset,
    )

    test_photo = dataset.get_photo("Adam_2011-03-31_03")
    dataset.read_image(test_photo).show()

    sam3_predictions = sam3.run(test_photo, "features")["predictions"]
    ears = []
    for prediction in sam3_predictions:
        if prediction["class"].strip() == "ear":
            ears.append(prediction)
    for ear in ears:
        crop_xyxy = (ear["x1"], ear["y1"], ear["x2"], ear["y2"])
        print(crop_xyxy)
    anchor_predictions = anchor.run(test_photo, crop_xyxy)
    print(anchor_predictions)
