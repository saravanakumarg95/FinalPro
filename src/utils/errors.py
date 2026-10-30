"""
src/utils/errors.py
────────────────────────────────────────────────────────────────
Custom exception hierarchy for the Cultural Story Generation
pipeline.

Design principle: every module raises a specific, catchable
exception so that calling code can distinguish failure modes
(e.g., bad input vs. API failure vs. retrieval failure) and
respond appropriately without silent fallbacks.
"""


# ─── Base ─────────────────────────────────────────────────────────────────────

class CulturalStoryError(Exception):
    """Base exception for all project-specific errors."""


# ─── Input Validation ─────────────────────────────────────────────────────────

class ImageValidationError(CulturalStoryError):
    """Raised when an uploaded image fails validation checks."""


class UnsupportedImageFormatError(ImageValidationError):
    """Raised when the image file format is not supported."""


class ImageTooLargeError(ImageValidationError):
    """Raised when the image exceeds the maximum allowed file size."""


class ImageDimensionError(ImageValidationError):
    """Raised when image dimensions are below the required minimum."""


class EmptyImageError(ImageValidationError):
    """Raised when the image file is empty or unreadable."""


# ─── Configuration ────────────────────────────────────────────────────────────

class ConfigurationError(CulturalStoryError):
    """Raised when a required configuration value is missing or invalid."""


class MissingAPIKeyError(ConfigurationError):
    """Raised when a required API key is not set in the environment."""


# ─── Knowledge Base ───────────────────────────────────────────────────────────

class KnowledgeBaseError(CulturalStoryError):
    """Base exception for knowledge base errors."""


class KBRecordValidationError(KnowledgeBaseError):
    """Raised when a KB record fails schema validation."""


class KBIngestionError(KnowledgeBaseError):
    """Raised when the KB ingestion pipeline encounters a fatal error."""


class EmptyKnowledgeBaseError(KnowledgeBaseError):
    """Raised when the knowledge base contains no records."""


# ─── Embeddings ───────────────────────────────────────────────────────────────

class EmbeddingError(CulturalStoryError):
    """Raised when embedding generation fails."""


class EmbeddingModelLoadError(EmbeddingError):
    """Raised when the embedding model cannot be loaded."""


# ─── Retrieval ────────────────────────────────────────────────────────────────

class RetrievalError(CulturalStoryError):
    """Base exception for retrieval pipeline errors."""


class VectorStoreError(RetrievalError):
    """Raised when the vector store operation fails."""


class NoEvidenceFoundError(RetrievalError):
    """
    Raised when retrieval finds no relevant cultural evidence.

    This is an expected, handled condition — not a system failure.
    The pipeline must handle this gracefully without fabricating evidence.
    """


# ─── Visual Understanding ─────────────────────────────────────────────────────

class VLMError(CulturalStoryError):
    """Base exception for vision-language model errors."""


class VLMAPIError(VLMError):
    """Raised when the VLM API call fails (network, rate limit, auth)."""


class VLMResponseParseError(VLMError):
    """Raised when the VLM response cannot be parsed into VisualObservation."""


class VLMRateLimitError(VLMAPIError):
    """Raised when the VLM API rate limit is exceeded."""


# ─── Reasoning ────────────────────────────────────────────────────────────────

class ReasoningError(CulturalStoryError):
    """Raised when the reasoning pipeline fails."""


# ─── Story Generation ─────────────────────────────────────────────────────────

class GenerationError(CulturalStoryError):
    """Base exception for story generation errors."""


class LLMAPIError(GenerationError):
    """Raised when the LLM API call fails."""


class LLMRateLimitError(LLMAPIError):
    """Raised when the LLM API rate limit is exceeded."""


class LLMResponseParseError(GenerationError):
    """Raised when the LLM response cannot be parsed into GeneratedStory."""


# ─── Verification ─────────────────────────────────────────────────────────────

class VerificationError(CulturalStoryError):
    """Raised when the consistency checker encounters a system error."""


# ─── Evaluation ───────────────────────────────────────────────────────────────

class EvaluationError(CulturalStoryError):
    """Raised when the evaluation pipeline fails."""


class ExperimentLogError(EvaluationError):
    """Raised when experiment results cannot be saved."""
