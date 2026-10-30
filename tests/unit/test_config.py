"""
tests/unit/test_config.py
────────────────────────────────────────────────────────────────
Unit tests for src/config/settings.py

Tests cover:
  - Default values in mock mode
  - Environment variable override
  - Validation constraints (top_k range)
  - require_gemini_key raises without key
  - is_mock_mode / is_live_mode properties
  - Settings cache clearing between tests
"""

import pytest

from src.config.settings import AppConfig, AppMode, get_settings


class TestDefaultValues:
    def test_default_app_mode_is_mock(self, app_config):
        assert app_config.app_mode == AppMode.MOCK

    def test_default_top_k_is_five(self, app_config):
        assert app_config.retrieval_top_k == 5

    def test_default_story_words(self, app_config):
        assert app_config.story_target_words == 400

    def test_default_vlm_model(self, app_config):
        assert "gemini" in app_config.vlm_model.lower()


class TestModeProperties:
    def test_is_mock_mode_true_when_mock(self):
        cfg = AppConfig(app_mode="mock")
        assert cfg.is_mock_mode is True
        assert cfg.is_live_mode is False

    def test_is_live_mode_true_when_live(self):
        cfg = AppConfig(app_mode="live")
        assert cfg.is_live_mode is True
        assert cfg.is_mock_mode is False


class TestGeminiKeyRequirement:
    def test_require_gemini_key_raises_without_key(self):
        cfg = AppConfig(gemini_api_key="")
        with pytest.raises(EnvironmentError) as exc_info:
            cfg.require_gemini_key()
        assert "GEMINI_API_KEY" in str(exc_info.value)

    def test_require_gemini_key_returns_key_when_set(self):
        cfg = AppConfig(gemini_api_key="test-key-abc123")
        key = cfg.require_gemini_key()
        assert key == "test-key-abc123"

    def test_has_gemini_key_false_when_empty(self):
        cfg = AppConfig(gemini_api_key="")
        assert cfg.has_gemini_key is False

    def test_has_gemini_key_true_when_set(self):
        cfg = AppConfig(gemini_api_key="any-key")
        assert cfg.has_gemini_key is True


class TestValidation:
    def test_top_k_below_minimum_raises(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            AppConfig(retrieval_top_k=0)

    def test_top_k_above_maximum_raises(self):
        with pytest.raises(Exception):
            AppConfig(retrieval_top_k=21)

    def test_top_k_boundary_values_accepted(self):
        cfg_low = AppConfig(retrieval_top_k=1)
        assert cfg_low.retrieval_top_k == 1
        cfg_high = AppConfig(retrieval_top_k=20)
        assert cfg_high.retrieval_top_k == 20


class TestGetSettingsSingleton:
    def test_get_settings_returns_config(self):
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, AppConfig)

    def test_get_settings_cached(self):
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
