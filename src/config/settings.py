"""
src/config/settings.py
────────────────────────────────────────────────────────────────
Central application configuration loaded from environment
variables and .env file via pydantic-settings.

All modules import AppConfig (or get_settings()) from here.
No secrets are hardcoded; all come from environment variables.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppMode(str, Enum):
    """Controls whether real model calls or deterministic mocks are used."""

    MOCK = "mock"
    LIVE = "live"


class VectorStore(str, Enum):
    """Which vector store backend to use for retrieval."""

    CHROMADB = "chromadb"
    FAISS = "faiss"


class ReadingLevel(str, Enum):
    """Target reading level for generated stories."""

    GENERAL = "general"
    SIMPLE = "simple"
    ACADEMIC = "academic"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# Project root is three levels up from this file:
# src/config/settings.py  →  src/config/  →  src/  →  project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AppConfig(BaseSettings):
    """
    Application configuration.

    Values are loaded from environment variables (and .env file).
    Defaults are chosen to run safely in mock mode without API keys.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unrecognised env vars
    )

    # ── API Keys ──────────────────────────────────────────────────────────────
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key. Required for APP_MODE=live.",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. Optional alternative LLM provider.",
    )

    # ── Model Names ───────────────────────────────────────────────────────────
    vlm_model: str = Field(
        default="gemini-2.5-flash",
        description="VLM model identifier.",
    )
    llm_model: str = Field(
        default="gemini-2.5-flash",
        description="LLM model identifier for story generation.",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model for embeddings.",
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    vector_store: VectorStore = Field(
        default=VectorStore.CHROMADB,
        description="Vector store backend.",
    )
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top-k cultural records to retrieve.",
    )

    # ── Story Generation ──────────────────────────────────────────────────────
    story_target_words: int = Field(
        default=400,
        ge=100,
        le=2000,
        description="Target word count for generated stories.",
    )
    story_reading_level: ReadingLevel = Field(
        default=ReadingLevel.GENERAL,
        description="Target reading level for generated stories.",
    )

    # ── Application Mode ──────────────────────────────────────────────────────
    app_mode: AppMode = Field(
        default=AppMode.MOCK,
        description=(
            "mock: all pipeline stages use deterministic mocks (no API). "
            "live: real model calls are made."
        ),
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    kb_raw_path: Path = Field(
        default=PROJECT_ROOT / "data" / "raw" / "sample_kb.jsonl",
        description="Path to raw knowledge base JSONL file.",
    )
    kb_processed_path: Path = Field(
        default=PROJECT_ROOT / "data" / "processed",
        description="Directory for processed KB files.",
    )
    chroma_db_path: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "chroma_db",
        description="Path to ChromaDB persistence directory.",
    )
    experiments_path: Path = Field(
        default=PROJECT_ROOT / "experiments" / "results",
        description="Directory for saving experiment results.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Application log level.",
    )

    # ── Derived helpers ───────────────────────────────────────────────────────
    @field_validator("retrieval_top_k", mode="before")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        v = int(v)
        if not 1 <= v <= 20:
            raise ValueError("retrieval_top_k must be between 1 and 20")
        return v

    @property
    def is_mock_mode(self) -> bool:
        """True when running in deterministic mock mode."""
        return self.app_mode == AppMode.MOCK

    @property
    def is_live_mode(self) -> bool:
        """True when real model API calls are enabled."""
        return self.app_mode == AppMode.LIVE

    @property
    def has_gemini_key(self) -> bool:
        """True when a non-empty Gemini API key is configured."""
        return bool(self.gemini_api_key.strip())

    def require_gemini_key(self) -> str:
        """Return the Gemini API key or raise a clear error."""
        if not self.has_gemini_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or set the environment variable. "
                "See .env.example for details."
            )
        return self.gemini_api_key


@lru_cache(maxsize=1)
def get_settings() -> AppConfig:
    """
    Return the cached application settings singleton.

    Using lru_cache ensures settings are loaded once and reused
    across the application. In tests, call get_settings.cache_clear()
    before patching environment variables.
    """
    return AppConfig()
