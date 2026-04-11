"""Tests for edgevox.core.pipeline — sentence splitting and streaming, no mocks needed."""

from __future__ import annotations

import pytest

from edgevox.core.pipeline import (
    ABBREVIATIONS,
    MAX_CHUNK_CHARS,
    _find_sentence_break,
    _is_sentence_boundary,
    stream_sentences,
)


def _tokens(text: str):
    """Simulate a token stream — one character at a time."""
    yield from text


def _word_tokens(text: str):
    """Simulate a token stream — one word at a time."""
    for word in text.split(" "):
        yield word + " "


class TestStreamSentences:
    def test_simple_two_sentences(self):
        result = list(stream_sentences(_tokens("Hello world. How are you?")))
        assert result == ["Hello world.", "How are you?"]

    def test_exclamation_and_question(self):
        result = list(stream_sentences(_tokens("Wow! Really?")))
        assert result == ["Wow!", "Really?"]

    def test_single_sentence_no_punct(self):
        result = list(stream_sentences(_tokens("Hello world")))
        assert result == ["Hello world"]

    def test_single_sentence_with_period(self):
        result = list(stream_sentences(_tokens("Hello.")))
        assert result == ["Hello."]

    def test_empty_stream(self):
        result = list(stream_sentences(iter([])))
        assert result == []

    def test_abbreviation_dr_not_split(self):
        result = list(stream_sentences(_tokens("Dr. Smith is here. Hello.")))
        assert result == ["Dr. Smith is here.", "Hello."]

    def test_abbreviation_eg_not_split(self):
        result = list(stream_sentences(_tokens("Use e.g. this one. Done.")))
        assert result == ["Use e.g. this one.", "Done."]

    def test_abbreviation_us_not_split(self):
        result = list(stream_sentences(_tokens("The U.S. is large. Yes.")))
        assert result == ["The U.S. is large.", "Yes."]

    def test_decimal_not_split(self):
        result = list(stream_sentences(_tokens("The value is 3.14 exactly. Done.")))
        assert result == ["The value is 3.14 exactly.", "Done."]

    def test_ellipsis_handling(self):
        # Ellipsis followed by space triggers a break after the dots
        result = list(stream_sentences(_tokens("Wait.. no. Yes.")))
        assert len(result) >= 2
        # The first chunk contains the ellipsis
        assert ".." in result[0]

    def test_multiple_sentences(self):
        text = "First sentence. Second sentence! Third sentence? Final."
        result = list(stream_sentences(_tokens(text)))
        assert len(result) == 4

    def test_long_clause_breaks_at_separator(self):
        # Create a string > MAX_CHUNK_CHARS with a comma near the middle
        half = "x" * (MAX_CHUNK_CHARS // 2 + 10)
        text = f"{half}, {half}"
        result = list(stream_sentences(_word_tokens(text)))
        assert len(result) >= 2

    def test_whitespace_only_tokens(self):
        result = list(stream_sentences(iter(["   ", "  "])))
        assert result == []


class TestIsSentenceBoundary:
    def test_normal_word_is_boundary(self):
        assert _is_sentence_boundary("hello") is True

    def test_empty_is_boundary(self):
        assert _is_sentence_boundary("") is True

    @pytest.mark.parametrize("abbr", sorted(ABBREVIATIONS))
    def test_abbreviations_not_boundary(self, abbr):
        bare = abbr.rstrip(".")
        assert _is_sentence_boundary(f"the {bare}") is False

    def test_digit_not_boundary(self):
        assert _is_sentence_boundary("value is 3") is False

    def test_double_dot_not_boundary(self):
        assert _is_sentence_boundary("wait..") is False

    def test_short_word_not_boundary(self):
        # Single-letter "words" before a period are treated as abbreviations
        assert _is_sentence_boundary("letter A") is False


class TestFindSentenceBreak:
    def test_simple_break(self):
        pos = _find_sentence_break("Hello world. Next")
        assert pos is not None
        assert "Hello world. Next"[:pos].strip() == "Hello world."

    def test_no_break(self):
        assert _find_sentence_break("Hello world") is None

    def test_question_mark(self):
        pos = _find_sentence_break("Really? Yes")
        assert pos is not None

    def test_exclamation(self):
        pos = _find_sentence_break("Wow! Next")
        assert pos is not None
