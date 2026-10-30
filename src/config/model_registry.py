"""
src/config/model_registry.py
────────────────────────────────────────────────────────────────
Registry of available VLM, LLM, and embedding model adapters.

Adding a new adapter:
  1. Implement the adapter class in its module.
  2. Register it here under a string key.
  3. Update AppConfig to accept the new key as a valid option.

This file is intentionally minimal at Phase 1 — adapters are
registered here as strings that are resolved lazily at runtime.
"""

from typing import Any

# Registered VLM adapter keys and their module paths.
VLM_REGISTRY: dict[str, str] = {
    "gemini": "src.vision.gemini_vlm.GeminiVLM",
    "mock": "src.vision.mock_vlm.MockVLM",
}

# Registered LLM adapter keys and their module paths.
LLM_REGISTRY: dict[str, str] = {
    "gemini": "src.generation.gemini_llm.GeminiLLM",
    "mock": "src.generation.mock_llm.MockLLM",
}

# Registered embedding adapter keys and their module paths.
EMBEDDING_REGISTRY: dict[str, str] = {
    "all-MiniLM-L6-v2": "src.embeddings.minilm_embedder.MiniLMEmbedder",
    "mock": "src.embeddings.mock_embedder.MockEmbedder",
}


def resolve_class(module_path: str) -> Any:
    """
    Dynamically import and return a class from a dotted module path.

    Args:
        module_path: Dotted path like "src.vision.gemini_vlm.GeminiVLM".

    Returns:
        The class object.

    Raises:
        ImportError: If the module or class cannot be found.
    """
    import importlib
    parts = module_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ImportError(f"Invalid module path: '{module_path}'")
    module_name, class_name = parts
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
