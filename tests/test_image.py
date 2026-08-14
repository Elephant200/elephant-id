"""Tests for encoded-byte image decoding."""

import cv2
import numpy as np
import pytest

from elephant_id.image import decode_image


def test_decode_image_returns_original_bgr_pixels() -> None:
    """Decode encoded bytes into the canonical OpenCV-native BGR image."""
    original = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    success, encoded = cv2.imencode(".png", original)
    assert success

    decoded = decode_image(encoded.tobytes())

    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, original)


def test_decode_image_rejects_invalid_bytes() -> None:
    """Invalid encoded image bytes fail clearly at the decode seam."""
    with pytest.raises(ValueError, match="valid image"):
        decode_image(b"not an encoded image")
