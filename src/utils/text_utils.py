"""src/utils/text_utils.py — Text cleaning and chunking utilities."""

import re
from typing import Generator


def clean_text(text: str) -> str:
    """
    Normalise whitespace and remove control characters from text.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text with normalised whitespace.
    """
    if not text:
        return ""
    # Remove null bytes and other control characters except newlines/tabs
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)
    # Collapse multiple whitespace (preserving single newlines for structure)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> Generator[str, None, None]:
    """
    Split text into overlapping chunks suitable for embedding.

    Uses sentence-boundary-aware splitting: splits on full stops
    first, then falls back to word boundaries.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks.

    Yields:
        Text chunks.
    """
    text = clean_text(text)
    if not text:
        return

    if len(text) <= chunk_size:
        yield text
        return

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current_chunk: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len + sentence_len > chunk_size and current_chunk:
            chunk = " ".join(current_chunk)
            yield chunk
            # Keep overlap: retain last sentences that fit within overlap chars
            overlap_chunk: list[str] = []
            overlap_len = 0
            for s in reversed(current_chunk):
                if overlap_len + len(s) <= overlap:
                    overlap_chunk.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk = overlap_chunk
            current_len = overlap_len

        current_chunk.append(sentence)
        current_len += sentence_len + 1  # +1 for space

    if current_chunk:
        yield " ".join(current_chunk)


def count_words(text: str) -> int:
    """Return the approximate word count of a text string."""
    return len(text.split()) if text.strip() else 0


def truncate_text(text: str, max_chars: int = 300, suffix: str = "…") -> str:
    """
    Truncate text to max_chars characters, breaking at a word boundary.

    Args:
        text: Input text.
        max_chars: Maximum character count.
        suffix: String appended when truncation occurs.

    Returns:
        Truncated (or original) text.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    return truncated + suffix
