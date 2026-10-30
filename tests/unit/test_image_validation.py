"""
tests/unit/test_image_validation.py
────────────────────────────────────────────────────────────────
Unit tests for src/utils/image_utils.py

Tests cover:
  - Valid image bytes (happy path)
  - Empty bytes
  - Oversized image
  - Unsupported extension
  - Dimensions below minimum
  - Path-based loading
  - Base64 encoding round-trip
  - Resize behaviour
"""

import io
from pathlib import Path

import pytest
from PIL import Image

from src.utils.errors import (
    EmptyImageError,
    ImageDimensionError,
    ImageTooLargeError,
    UnsupportedImageFormatError,
)
from src.utils.image_utils import (
    MAX_FILE_SIZE_BYTES,
    MIN_DIMENSION,
    image_to_base64,
    image_to_bytes,
    resize_image,
    validate_image_bytes,
    validate_image_path,
)


# ── Happy path ─────────────────────────────────────────────────────────────────

class TestValidateImageBytesHappyPath:
    def test_valid_jpeg_returns_pil_image(self, minimal_image_bytes):
        img = validate_image_bytes(minimal_image_bytes, filename="test.jpg")
        assert isinstance(img, Image.Image)

    def test_valid_jpeg_is_rgb(self, minimal_image_bytes):
        img = validate_image_bytes(minimal_image_bytes, filename="test.jpg")
        assert img.mode == "RGB"

    def test_valid_jpeg_dimensions(self, minimal_image_bytes):
        img = validate_image_bytes(minimal_image_bytes, filename="test.jpg")
        assert img.size == (100, 100)

    def test_valid_png_accepted(self):
        buf = io.BytesIO()
        Image.new("RGB", (200, 200), color=(0, 128, 255)).save(buf, format="PNG")
        img = validate_image_bytes(buf.getvalue(), filename="image.png")
        assert img.mode == "RGB"

    def test_no_filename_still_validates(self, minimal_image_bytes):
        """Extension check is skipped when no extension is present."""
        img = validate_image_bytes(minimal_image_bytes, filename="imagewithoutextension")
        assert isinstance(img, Image.Image)


# ── Empty data ─────────────────────────────────────────────────────────────────

class TestEmptyData:
    def test_empty_bytes_raises_empty_image_error(self, empty_bytes):
        with pytest.raises(EmptyImageError):
            validate_image_bytes(empty_bytes, filename="test.jpg")


# ── File size ─────────────────────────────────────────────────────────────────

class TestFileSizeValidation:
    def test_oversized_bytes_raises_too_large_error(self):
        oversized = b"A" * (MAX_FILE_SIZE_BYTES + 1)
        with pytest.raises(ImageTooLargeError) as exc_info:
            validate_image_bytes(oversized, filename="huge.jpg")
        assert "MB" in str(exc_info.value)


# ── Unsupported format ────────────────────────────────────────────────────────

class TestUnsupportedFormat:
    def test_unsupported_extension_raises_error(self, minimal_image_bytes):
        with pytest.raises(UnsupportedImageFormatError):
            validate_image_bytes(minimal_image_bytes, filename="file.tiff")

    def test_non_image_bytes_raises_error(self):
        fake_bytes = b"This is not an image file at all."
        with pytest.raises((UnsupportedImageFormatError, EmptyImageError)):
            validate_image_bytes(fake_bytes, filename="notanimage.jpg")


# ── Dimensions ────────────────────────────────────────────────────────────────

class TestDimensionValidation:
    def test_tiny_image_raises_dimension_error(self, tiny_image_bytes):
        with pytest.raises(ImageDimensionError) as exc_info:
            validate_image_bytes(tiny_image_bytes, filename="tiny.jpg")
        assert str(MIN_DIMENSION) in str(exc_info.value)


# ── Path-based loading ────────────────────────────────────────────────────────

class TestValidateImagePath:
    def test_valid_path_loads_image(self, tmp_path, minimal_rgb_image):
        p = tmp_path / "test.jpg"
        minimal_rgb_image.save(str(p), format="JPEG")
        img = validate_image_path(p)
        assert isinstance(img, Image.Image)

    def test_missing_path_raises_empty_error(self, tmp_path):
        p = tmp_path / "nonexistent.jpg"
        with pytest.raises(EmptyImageError):
            validate_image_path(p)


# ── Utilities ─────────────────────────────────────────────────────────────────

class TestImageUtilities:
    def test_image_to_bytes_returns_nonempty(self, minimal_rgb_image):
        b = image_to_bytes(minimal_rgb_image)
        assert len(b) > 0

    def test_image_to_base64_is_string(self, minimal_rgb_image):
        b64 = image_to_base64(minimal_rgb_image)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_resize_does_not_upscale(self, minimal_rgb_image):
        """A 100×100 image should not be enlarged when max_size=200."""
        resized = resize_image(minimal_rgb_image, max_size=200)
        assert resized.size == (100, 100)

    def test_resize_reduces_large_image(self):
        large = Image.new("RGB", (2000, 1000))
        resized = resize_image(large, max_size=512)
        assert resized.size[0] == 512

    def test_resize_preserves_aspect_ratio(self):
        large = Image.new("RGB", (2000, 1000))
        resized = resize_image(large, max_size=512)
        w, h = resized.size
        # 2:1 ratio → 512×256
        assert w == 512
        assert h == 256
