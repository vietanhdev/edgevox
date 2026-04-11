"""Integration tests for LLM with real model.

Downloads Gemma 4 E2B Q4_K_M (~1.8GB) and runs real inference.
Extremely slow — only run manually or on release.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def llm():
    """Load real LLM (Gemma 4 E2B Q4_K_M, ~1.8GB download)."""
    try:
        from edgevox.llm import LLM

        return LLM(language="en")
    except Exception as e:
        pytest.skip(f"LLM not available: {e}")


class TestLLMReal:
    def test_loads_successfully(self, llm):
        assert llm._llm is not None
        assert llm._language == "en"

    def test_chat_returns_text(self, llm):
        reply = llm.chat("Say hello in exactly two words.")
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_chat_stream_yields_tokens(self, llm):
        llm.reset()
        tokens = list(llm.chat_stream("Say hi in one word."))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_history_grows(self, llm):
        llm.reset()
        initial_len = len(llm._history)
        llm.chat("Test message.")
        assert len(llm._history) == initial_len + 2  # user + assistant

    def test_reset_clears_history(self, llm):
        llm.chat("Another message.")
        llm.reset()
        assert len(llm._history) == 1
        assert llm._history[0]["role"] == "system"

    def test_set_language_updates_prompt(self, llm):
        llm.set_language("fr")
        assert "French" in llm._history[0]["content"]
        reply = llm.chat("Dites bonjour.")
        assert isinstance(reply, str)
        # Restore
        llm.set_language("en")

    @pytest.mark.parametrize("lang", ["vi", "ko", "ja", "de", "th"])
    def test_language_hints(self, llm, lang):
        llm.set_language(lang)
        assert len(llm._history[0]["content"]) > 0
        llm.set_language("en")
