import numpy as np
import pytest

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


def test_decode_rle_mask_accepts_bytes_counts(rle_from_mask):
    mask = np.array([[True, False], [False, True]])
    rle = rle_from_mask(mask)
    rle["counts"] = rle["counts"].encode("utf-8")

    decoded = decode_rle_mask(rle)

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


@pytest.mark.parametrize(
    "rle",
    [
        {},
        {"size": [2, 3]},
        {"counts": "abc"},
    ],
)
def test_decode_rle_mask_rejects_missing_fields(rle):
    with pytest.raises(ValueError, match="must have size and counts"):
        decode_rle_mask(rle)


@pytest.mark.parametrize(
    "rle",
    [
        {"size": [], "counts": "abc"},
        {"size": [2, 3], "counts": ""},
    ],
)
def test_decode_rle_mask_rejects_empty_fields(rle):
    with pytest.raises(ValueError, match="must be non-empty"):
        decode_rle_mask(rle)


def test_decode_rle_mask_rejects_non_string_counts():
    with pytest.raises(ValueError, match="counts must be a string or bytes"):
        decode_rle_mask({"size": [2, 3], "counts": [1, 2, 3]})


@pytest.mark.parametrize(
    "size",
    [123, [2, 3, 1], (2,)],
)
def test_decode_rle_mask_rejects_bad_size(size):
    with pytest.raises(ValueError, match="size must be a list or tuple of length 2"):
        decode_rle_mask({"size": size, "counts": "abc"})


@pytest.mark.parametrize(
    "size",
    [
        [0, 3],
        [2, 0],
        [-1, 3],
        [2, -1],
        [2.5, 3],
        [2, "3"],
        [True, 3],
    ],
)
def test_decode_rle_mask_rejects_non_positive_integer_size_values(size):
    with pytest.raises(ValueError, match="positive integers"):
        decode_rle_mask({"size": size, "counts": "abc"})


def test_mask_bounds_wrap_true_pixels():
    mask = np.array(
        [
            [False, False, False, False],
            [False, True, False, False],
            [False, True, True, False],
        ]
    )

    assert mask_bounds(mask) == (1, 1, 3, 3)


def test_mask_bounds_single_pixel():
    mask = np.zeros((4, 5), dtype=bool)
    mask[2, 3] = True

    assert mask_bounds(mask) == (3, 2, 4, 3)


def test_mask_bounds_full_mask():
    mask = np.ones((3, 4), dtype=bool)

    assert mask_bounds(mask) == (0, 0, 4, 3)


def test_mask_bounds_rejects_empty_mask():
    with pytest.raises(ValueError, match="empty mask"):
        mask_bounds(np.zeros((3, 4), dtype=bool))


@pytest.mark.parametrize(
    "shape",
    [(4,), (2, 3, 1)],
)
def test_mask_bounds_rejects_non_2d_mask(shape):
    with pytest.raises(ValueError, match="must be 2D"):
        mask_bounds(np.ones(shape, dtype=bool))


def test_decode_rle_mask_converts_fortran_layout_to_contiguous_booleans(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decoded COCO storage becomes contiguous canonical booleans for OpenCV."""
    source = np.asfortranarray(np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8))
    monkeypatch.setattr('elephant_id.image.masks.coco_mask.decode', lambda _: source)
    result = decode_rle_mask({'size': [2, 3], 'counts': b'unused'})
    assert result.flags.c_contiguous
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result.view(np.uint8), source)
    np.testing.assert_array_equal(source, [[0, 1, 0], [1, 0, 1]])
