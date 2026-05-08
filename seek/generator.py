"""
Generates a SEEK code for one elephant
"""

import numpy as np

from preprocess.background import remove_background
from vision.sam3 import segment_image
from seek.gender import get_gender_code
from seek.age import get_age_code
from seek.tusks import get_tusk_code
from seek.ears import get_ears_code, get_ear_contour
from seek.view import get_view


def get_seek_code(elephant_image: np.ndarray) -> str:
    """
    Given a cropped image of one elephant, output the corresponding SEEK code in the form:
    _ _ _ T _ _ E _ _ _ _ - _ _ _ _ X _ _ S _ _ _
    Following the form of https://elephantsalive.org/wp-content/uploads/2021/07/Bedetti-et-al-2020.pdf#page=7
    """
    predictions = segment_image(
        image=elephant_image,
        queries=["trunk", "tusk", "ear", "tail"],
        confidence_threshold=0.5,
        nms=True,
        nms_iou_threshold=0.2,
    )
    
    no_background_image = remove_background(elephant_image)
    view = get_view(no_background_image)

    gender_code = get_gender_code(no_background_image)
    age_code = get_age_code(no_background_image)
    tusk_code = get_tusk_code(elephant_image, view)
    (
        left_ear_code,
        right_ear_code,
        extreme_features_code,
        special_features_ear_code,
    ) = get_ears_code(elephant_image)
    return f"{gender_code}{age_code}T{tusk_code}E{left_ear_code}-{right_ear_code}X{extreme_features_code}S{special_features_ear_code}_"

if __name__ == "__main__":
    import cv2
    from dotenv import load_dotenv
    from vision.visualization import visualize_predictions

    load_dotenv()
    image = cv2.imread("sample.jpg")
    if image is None:
        raise FileNotFoundError("Image path is invalid or unreadable.")

    predictions = get_seek_code(image)
    annotated = visualize_predictions(image, predictions)

    output_path = "sam3_predictions_visualization.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"Saved visualization to {output_path}")
    print("Press any key in the image window to close.")

    cv2.imshow("SAM3 Predictions", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()