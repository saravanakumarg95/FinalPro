"""app/components/image_upload.py — Image upload and validation component."""

import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from PIL import Image
from src.utils.image_utils import validate_image_bytes
from src.utils.errors import ImageValidationError


def render_image_upload():
    """Render the image upload widget with validation. Returns (file, bytes, PIL Image) or Nones."""
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a JPG, PNG, or WebP image. Maximum 20 MB.",
        key="image_uploader",
    )

    if not uploaded:
        return None, None, None

    try:
        image_bytes = uploaded.read()
        pil_image = validate_image_bytes(image_bytes, filename=uploaded.name)
        st.success(f"✅ Image loaded: {uploaded.name} ({len(image_bytes) // 1024} KB)")
        return uploaded, image_bytes, pil_image
    except ImageValidationError as e:
        st.error(f"❌ Image validation failed: {e}")
        return uploaded, None, None
    except Exception as e:
        st.error(f"❌ Unexpected error loading image: {e}")
        return uploaded, None, None
