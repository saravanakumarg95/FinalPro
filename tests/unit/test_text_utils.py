"""
tests/unit/test_text_utils.py
────────────────────────────────────────────────────────────────
Unit tests for src/utils/text_utils.py
"""

import pytest

from src.utils.text_utils import chunk_text, clean_text, count_words, truncate_text


class TestCleanText:
    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self):
        result = clean_text("hello    world")
        assert result == "hello world"

    def test_removes_null_bytes(self):
        result = clean_text("hello\x00world")
        assert "\x00" not in result

    def test_preserves_newlines(self):
        result = clean_text("line one\nline two")
        assert "\n" in result

    def test_collapses_triple_newlines(self):
        result = clean_text("a\n\n\n\nb")
        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in result


class TestChunkText:
    def test_short_text_yields_single_chunk(self):
        text = "A short piece of text."
        chunks = list(chunk_text(text, chunk_size=512))
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_yields_nothing(self):
        chunks = list(chunk_text("", chunk_size=512))
        assert chunks == []

    def test_long_text_splits_into_multiple_chunks(self):
        # Create text longer than chunk_size
        long_text = " ".join(["This is a sentence."] * 50)
        chunks = list(chunk_text(long_text, chunk_size=100, overlap=10))
        assert len(chunks) > 1

    def test_all_chunks_are_strings(self):
        long_text = " ".join(["A short sentence."] * 30)
        for chunk in chunk_text(long_text, chunk_size=80, overlap=10):
            assert isinstance(chunk, str)

    def test_chunks_not_empty(self):
        long_text = " ".join(["A sentence here."] * 30)
        for chunk in chunk_text(long_text, chunk_size=80, overlap=10):
            assert len(chunk.strip()) > 0


class TestCountWords:
    def test_empty_string(self):
        assert count_words("") == 0

    def test_single_word(self):
        assert count_words("hello") == 1

    def test_multiple_words(self):
        assert count_words("hello world foo bar") == 4

    def test_whitespace_only(self):
        assert count_words("   ") == 0


class TestTruncateText:
    def test_short_text_unchanged(self):
        text = "Short text."
        assert truncate_text(text, max_chars=100) == text

    def test_long_text_truncated(self):
        text = "A " * 200  # 400 chars
        result = truncate_text(text, max_chars=50)
        assert len(result) <= 53  # 50 + len("…")

    def test_suffix_appended(self):
        text = "A " * 200
        result = truncate_text(text, max_chars=50, suffix="...")
        assert result.endswith("...")

    def test_truncation_at_word_boundary(self):
        text = "Hello world this is a test sentence for truncation."
        result = truncate_text(text, max_chars=20)
        # Result must end with the ellipsis (truncation occurred)
        assert result.endswith("…")
        # Result without ellipsis must not end mid-word (last char before … is a space boundary)
        body = result[:-1]  # strip the ellipsis
        # Body should not end with a partial word: last char should be a space or end of word
        assert not body[-1].isalpha() or body.endswith("is") or body.endswith("world") or body.endswith("Hello")
