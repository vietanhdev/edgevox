"""Local fixtures for tool-parser tests.

Re-exports :class:`ScriptedLLM` from the harness conftest so these tests
can drive the full :class:`LLMAgent` loop without depending on the
harness package layout.
"""

from __future__ import annotations

import pytest

from tests.harness.conftest import ScriptedLLM


@pytest.fixture
def scripted_llm_factory():
    def _make(script=None, *, language="en"):
        return ScriptedLLM(script, language=language)

    return _make
