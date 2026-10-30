"""
tests/conftest.py
────────────────────────────────────────────────────────────────
Shared pytest fixtures for all test modules.

Fixtures defined here are automatically available to all tests
without explicit imports.
"""

import io
import os
from pathlib import Path

import pytest
from PIL import Image

# Ensure the project root is on sys.path so that `src.*` imports work
# when running pytest from the project root.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Environment helpers ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def force_mock_mode(monkeypatch):
    """
    Force APP_MODE=mock for all tests.

    This ensures no test ever makes a real API call.
    Tests that explicitly test live mode should override this fixture.
    """
    monkeypatch.setenv("APP_MODE", "mock")
    # Clear the settings cache so the patched env is picked up
    from src.config.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── Image fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_rgb_image() -> Image.Image:
    """A valid 100×100 red RGB PIL Image."""
    img = Image.new("RGB", (100, 100), color=(200, 50, 50))
    return img


@pytest.fixture
def valid_pil_image() -> Image.Image:
    """Alias for minimal_rgb_image — used by integration tests."""
    img = Image.new("RGB", (200, 200), color=(180, 80, 40))
    return img


@pytest.fixture
def mock_pil_image() -> Image.Image:
    """A larger 300×300 PIL image for pipeline component tests."""
    img = Image.new("RGB", (300, 300), color=(100, 60, 20))
    return img


@pytest.fixture
def minimal_image_bytes(minimal_rgb_image) -> bytes:
    """Valid JPEG bytes for a 100×100 image."""
    buf = io.BytesIO()
    minimal_rgb_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def tiny_image_bytes() -> bytes:
    """JPEG bytes for a 10×10 image (below minimum dimension threshold)."""
    img = Image.new("RGB", (10, 10), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def empty_bytes() -> bytes:
    """Empty byte string for testing empty-file handling."""
    return b""


# ── Config fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def app_config():
    """Return an AppConfig instance in mock mode."""
    from src.config.settings import AppConfig
    return AppConfig(app_mode="mock")


# ── Sample KB record fixture ───────────────────────────────────────────────────

@pytest.fixture
def sample_kb_record_dict() -> dict:
    """A valid KB record dictionary for testing ingestion."""
    return {
        "record_id": "test-001",
        "topic": "Test Festival",
        "cultural_region": "Test Region",
        "description": "A test cultural festival for unit testing purposes.",
        "evidence_text": "The Test Festival involves gathering and celebration.",
        "source_title": "Test Source",
        "source_url": "https://example.com/test",
        "source_type": "synthetic_demo",
        "confidence": "synthetic_demo",
        "known_limitations": "This is a synthetic record for testing only.",
        "tags": ["test", "festival"],
    }
