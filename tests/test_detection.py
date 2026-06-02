import numpy as np
import pytest

from elephant_id.ai.detection import Detection


def _masked(mask: np.ndarray, rle_from_mask) -> Detection:
    return Detection(
        xyxy=(0.0, 0.0, float(mask.shape[1]), float(mask.shape[0])),
        class_name="ear",
        class_id=2,
        confidence=0.9,
        rle_mask=rle_from_mask(mask),
    )


def test_coordinate_properties_read_xyxy_values():
    detection = Detection(
        xyxy=(1.0, 2.0, 3.0, 4.0), class_name="ear", class_id=2, confidence=0.9
    )

    assert detection.x1 == 1.0
    assert detection.y1 == 2.0
    assert detection.x2 == 3.0
    assert detection.y2 == 4.0


def test_area_uses_box_when_unmasked():
    detection = Detection(
        xyxy=(0.0, 0.0, 4.0, 3.0), class_name="ear", class_id=2, confidence=0.9
    )

    assert detection.area() == 12.0


def test_area_uses_mask_pixel_count_when_masked(rle_from_mask):
    mask = np.array([[True, True, False], [False, True, False]])

    assert _masked(mask, rle_from_mask).area() == 3.0


def test_mask_property_decodes_rle(rle_from_mask):
    mask = np.array([[True, False], [False, True]])

    decoded = _masked(mask, rle_from_mask).mask

    assert decoded.dtype == bool
    assert decoded.tolist() == mask.tolist()


def test_mask_property_without_rle_raises():
    detection = Detection(
        xyxy=(0.0, 0.0, 1.0, 1.0), class_name="ear", class_id=2, confidence=0.9
    )

    with pytest.raises(ValueError, match="no mask"):
        _ = detection.mask


def test_intersection_union_and_iou(rle_from_mask):
    a = _masked(np.array([[True, True], [False, False]]), rle_from_mask)
    b = _masked(np.array([[False, True], [False, True]]), rle_from_mask)

    assert a.intersection_area(b) == 1.0
    assert a.union_area(b) == 3.0
    assert a.iou(b) == pytest.approx(1 / 3)


def test_iou_of_disjoint_masks_is_zero(rle_from_mask):
    a = _masked(np.array([[True, False], [False, False]]), rle_from_mask)
    b = _masked(np.array([[False, False], [False, True]]), rle_from_mask)

    assert a.iou(b) == 0.0


def test_intersection_requires_both_masks(rle_from_mask):
    a = _masked(np.array([[True]]), rle_from_mask)
    b = Detection(xyxy=(0.0, 0.0, 1.0, 1.0), class_name="ear", class_id=2, confidence=0.9)

    with pytest.raises(ValueError, match="need a mask"):
        a.intersection_area(b)


def test_translate_shifts_box_and_keypoints_returning_new_instance():
    detection = Detection(
        xyxy=(1.0, 2.0, 3.0, 4.0),
        class_name="anchor",
        class_id=0,
        confidence=0.9,
        keypoints=((5.0, 6.0),),
    )

    moved = detection.translate(10.0, 20.0)

    assert moved.xyxy == (11.0, 22.0, 13.0, 24.0)
    assert moved.keypoints == ((15.0, 26.0),)
    assert detection.xyxy == (1.0, 2.0, 3.0, 4.0)  # original unchanged


def test_clip_clamps_box_to_image_bounds():
    detection = Detection(
        xyxy=(-2.0, 1.0, 5.0, 100.0), class_name="ear", class_id=2, confidence=0.9
    )

    assert detection.clip(8, 8).xyxy == (0, 1, 5, 8)


def test_to_dict_from_dict_round_trips(rle_from_mask):
    detection = Detection(
        xyxy=(1.0, 2.0, 3.0, 4.0),
        class_name="ear",
        class_id=2,
        confidence=0.9,
        rle_mask=rle_from_mask(np.array([[True, False]])),
        keypoints=((5.0, 6.0), (7.0, 8.0)),
    )

    assert Detection.from_dict(detection.to_dict()) == detection
