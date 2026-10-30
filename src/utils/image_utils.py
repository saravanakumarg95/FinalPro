"""
src/utils/image_utils.py
────────────────────────────────────────────────────────────────
Image validation and preprocessing utilities.

Validation checks:
  1. File extension / MIME type
  2. File size
  3. Image dimensions
  4. Image readability (PIL can open it)
  5. Non-empty content

All validation errors raise specific subclasses of ImageValidationError
so callers can distinguish the failure mode.
"""

import base64
import io
from pathlib import Path
from typing import Union

from PIL import Image, UnidentifiedImageError

from src.utils.errors import (
    EmptyImageError,
    ImageDimensionError,
    ImageTooLargeError,
    UnsupportedImageFormatError,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Configuration constants ────────────────────────────────────────────────────
SUPPORTED_FORMATS: set[str] = {"JPEG", "JPG", "PNG", "WEBP", "GIF", "BMP"}
SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
MAX_FILE_SIZE_MB: float = 20.0
MAX_FILE_SIZE_BYTES: int = int(MAX_FILE_SIZE_MB * 1024 * 1024)
MIN_DIMENSION: int = 64  # pixels


def validate_image_bytes(
    data: bytes,
    filename: str = "image",
) -> Image.Image:
    """
    Validate raw image bytes and return an opened PIL Image.

    Performs checks in order:
      1. Non-empty data
      2. File size
      3. Readability (PIL open)
      4. Format / extension
      5. Minimum dimensions

    Args:
        data: Raw image bytes.
        filename: Original filename (used for extension check and logging).

    Returns:
        Opened PIL Image in RGB mode.

    Raises:
        EmptyImageError: Data is empty.
        ImageTooLargeError: Data exceeds MAX_FILE_SIZE_BYTES.
        UnsupportedImageFormatError: Format not in SUPPORTED_FORMATS.
        ImageDimensionError: Image smaller than MIN_DIMENSION in either axis.
    """
    if not data:
        raise EmptyImageError("Image data is empty.")

    if len(data) > MAX_FILE_SIZE_BYTES:
        size_mb = len(data) / (1024 * 1024)
        raise ImageTooLargeError(
            f"Image is {size_mb:.1f} MB; maximum allowed is {MAX_FILE_SIZE_MB} MB."
        )

    ext = Path(filename).suffix.lower()
    if ext and ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedImageFormatError(
            f"File extension '{ext}' is not supported. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        pil_image = Image.open(io.BytesIO(data))
        pil_image.verify()  # check for truncation / corruption
        # Re-open after verify (verify closes the file pointer)
        pil_image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise UnsupportedImageFormatError(
            "File could not be identified as a supported image format."
        )
    except Exception as exc:
        raise EmptyImageError(f"Image could not be read: {exc}") from exc

    fmt = (pil_image.format or "").upper()
    if fmt and fmt not in SUPPORTED_FORMATS:
        raise UnsupportedImageFormatError(
            f"Image format '{fmt}' is not supported. Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    w, h = pil_image.size
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        raise ImageDimensionError(
            f"Image dimensions {w}×{h} are too small. "
            f"Minimum: {MIN_DIMENSION}×{MIN_DIMENSION} pixels."
        )

    log.debug(f"Image validated: {w}×{h} px, format={fmt}, size={len(data)} bytes")
    return pil_image.convert("RGB")


def validate_image_path(path: Union[str, Path]) -> Image.Image:
    """
    Load and validate an image from a filesystem path.

    Args:
        path: Path to the image file.

    Returns:
        Opened PIL Image in RGB mode.
    """
    path = Path(path)
    if not path.exists():
        raise EmptyImageError(f"Image file not found: {path}")
    data = path.read_bytes()
    return validate_image_bytes(data, filename=path.name)


def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """
    Serialize a PIL Image to bytes.

    Args:
        image: PIL Image object.
        format: Output format (JPEG, PNG, etc.).

    Returns:
        Image as bytes.
    """
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Serialize a PIL Image to a base64-encoded string.

    Used when passing images to APIs that expect base64 encoding.

    Args:
        image: PIL Image object.
        format: Output format.

    Returns:
        Base64-encoded string (no data URI prefix).
    """
    return base64.b64encode(image_to_bytes(image, format=format)).decode("utf-8")


def resize_image(
    image: Image.Image,
    max_size: int = 1024,
) -> Image.Image:
    """
    Resize image so the longest edge is at most max_size pixels.

    Preserves aspect ratio. Does not upscale.

    Args:
        image: PIL Image object.
        max_size: Maximum dimension in pixels.

    Returns:
        Resized PIL Image.
    """
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    if w >= h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    return image.resize((new_w, new_h), Image.LANCZOS)
