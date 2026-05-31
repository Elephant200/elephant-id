import numpy as np

from elephant_id.image.masks import decode_rle_mask, mask_bounds


def test_decode_rle_mask_returns_boolean_mask(rle_from_mask):
    mask = np.array(
        [
            [False, True, False],
            [False, True, True],
        ]
    )

    decoded = decode_rle_mask(rle_from_mask(mask))

    assert decoded.dtype == bool
    assert decoded.tolist() == mask.tolist()


def test_decode_rle_mask_collapses_single_mask_stack(monkeypatch):
    stacked = np.array(
        [
            [[0], [1], [0]],
            [[1], [0], [1]],
        ],
        dtype=np.uint8,
    )

    monkeypatch.setattr(
        "elephant_id.image.masks.coco_mask.decode",
        lambda _encoded: stacked,
    )

    decoded = decode_rle_mask({"size": [2, 3], "counts": b"unused"})

    assert decoded.dtype == bool
    assert decoded.tolist() == [
        [False, True, False],
        [True, False, True],
    ]


def test_mask_bounds_wrap_true_pixels():
    mask = np.array(
        [
            [False, False, False, False],
            [False, True, False, False],
            [False, True, True, False],
        ]
    )

    assert mask_bounds(mask) == (1, 1, 3, 3)
