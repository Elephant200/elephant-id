import time
from pathlib import Path

from ultralytics import SAM

from elephant_id.dataset import Dataset

dataset = Dataset(
    dataset_root=Path("dataset/elephants-alive/coded"),
    metadata_path=Path("dataset/elephants-alive/images.csv"),
)
sam3 = SAM("model_weights/sam3/sam3.pt")

print(sam3)

target_photo = dataset.get_photo("Acdc_2021-08-27_05")
image = dataset.read_image(target_photo)

start_time = time.perf_counter()
result = sam3.predict(
    source=image,
    labels=["ear"],
    device="mps",
)[0]
end_time = time.perf_counter()
print(f"Time taken: {end_time - start_time} seconds")

# Show the mask on the image using cv2
masked_image = result.plot(pil=True)
masked_image.show()
