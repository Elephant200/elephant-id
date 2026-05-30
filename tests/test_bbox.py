from elephant_id.ai.bbox import center_to_xyxy, prediction_center_to_xyxy


def test_center_to_xyxy_matches_cached_sam3_geometry():
    # Derived from .cache/sam3/body/Adam_2011-03-31_02.
    assert center_to_xyxy(
        x=1072.5,
        y=1319.5,
        width=315.0,
        height=357.0,
    ) == (915.0, 1141.0, 1230.0, 1498.0)


def test_prediction_center_to_xyxy_preserves_non_bbox_fields():
    prediction = {
        "class": "elephant",
        "class_id": 0,
        "confidence": 0.75,
        "x": 1072.5,
        "y": 1319.5,
        "width": 315.0,
        "height": 357.0,
    }

    converted = prediction_center_to_xyxy(prediction)

    assert converted == {
        "class": "elephant",
        "class_id": 0,
        "confidence": 0.75,
        "x1": 915.0,
        "y1": 1141.0,
        "x2": 1230.0,
        "y2": 1498.0,
    }
    assert prediction["x"] == 1072.5
