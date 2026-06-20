from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from elephant_id.coding.photo_analyzer import PhotoAnalyzer
from elephant_id.dataset import Dataset
from elephant_id.image.transforms import apply_crop
from elephant_id.log import configure_logging

if __name__ == "__main__":
    load_dotenv()
    configure_logging()

    dataset = Dataset(
        dataset_root=Path("dataset/elephants-alive/coded"),
        metadata_path=Path("dataset/elephants-alive/images.csv"),
    )
    analyzer = PhotoAnalyzer(dataset=dataset)

    photo_name = "Adam_2011-03-31_03"
    photo = dataset.get_photo(photo_name)
    analysis = analyzer.analyze(photo)

    image = dataset.read_image(photo)

    tusks = analysis["tusks"]
    for tusk in tusks:
        cv2.rectangle(image, (int(tusk["x1"]), int(tusk["y1"])), (int(tusk["x2"]), int(tusk["y2"])), (0, 0, 255), 2)
        cv2.putText(image, f"{tusk["side"]} {tusk["confidence"]:.2f}", (int(tusk["x1"]), int(tusk["y1"])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    ears = analysis["ears"]
    ear_images = []
    for ear in ears:
        cv2.rectangle(image, (int(ear["ear"].xyxy[0]), int(ear["ear"].xyxy[1])), (int(ear["ear"].xyxy[2]), int(ear["ear"].xyxy[3])), (0, 255, 0), 2)
        cv2.drawContours(image, [ear["ear"].contour], 0, (0, 255, 0), 2)
        cv2.putText(image, f"{ear["ear"].side}", (int(ear["ear"].xyxy[0]), int(ear["ear"].xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ear_image = apply_crop(image, ear["ear"].xyxy, pad=0.15)
        ear_images.append(ear_image)

    print(analysis)

    age_idx = np.argmax(analysis["age"]["probabilities"])
    age_probability = analysis["age"]["probabilities"][age_idx]
    age_bucket = analysis["age"]["buckets"][age_idx]
    next_age_bucket = analysis["age"]["buckets"][age_idx + 1]
    gender = "bull" if analysis["gender"]["bull"] > 0.5 else "cow"
    gender_conf = analysis["gender"]["bull"] if gender == "bull" else analysis["gender"]["cow"]

    cv2.imshow(f"Image {photo_name}\t(View: {analysis["view"]:<12} Age: {age_probability:.0%}, {age_bucket:.0f}-{next_age_bucket:.0f} years         Gender: {gender:<5} {gender_conf:.0%})", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
