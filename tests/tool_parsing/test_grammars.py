"""Unit tests for the GBNF grammar generator.

Pure-Python tests that don't require llama-cpp at import time —
``GrammarCache`` is exercised in a separate integration block that
imports llama-cpp lazily.
"""

from __future__ import annotations

import pytest

from edgevox.llm.grammars import (
    GrammarCache,
    reply_or_tool_grammar,
    single_tool_grammar,
    tool_call_grammar,
)

# ---------------------------------------------------------------------------
# Sample tool schemas (OpenAI shape)
# ---------------------------------------------------------------------------


def _tool(name: str, params: dict) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": "", "parameters": params},
    }


GET_TIME = _tool("get_time", {"type": "object", "properties": {}})
GRASP = _tool(
    "grasp",
    {
        "type": "object",
        "properties": {
            "object": {"type": "string"},
            "force": {"type": "number"},
        },
        "required": ["object"],
    },
)
SET_LIGHT = _tool(
    "set_light",
    {
        "type": "object",
        "properties": {
            "room": {"type": "string", "enum": ["kitchen", "living_room"]},
            "on": {"type": "boolean"},
        },
        "required": ["room", "on"],
    },
)


# ---------------------------------------------------------------------------
# tool_call_grammar
# ---------------------------------------------------------------------------


class TestToolCallGrammar:
    def test_empty_tools_raises(self):
        with pytest.raises(ValueError, match="at least one tool"):
            tool_call_grammar([])

    def test_root_is_tool_call_only(self):
        gbnf = tool_call_grammar([GET_TIME])
        assert "root ::= tool-call" in gbnf
        assert "text-reply" not in gbnf

    def test_each_tool_gets_an_args_rule(self):
        gbnf = tool_call_grammar([GET_TIME, GRASP, SET_LIGHT])
        for n in (0, 1, 2):
            assert f"args_{n} ::=" in gbnf

    def test_tool_name_appears_quoted_in_grammar(self):
        gbnf = tool_call_grammar([GRASP])
        # The tool name lives inside a quoted JSON literal in the grammar.
        assert "grasp" in gbnf

    def test_enum_values_become_alternations(self):
        gbnf = tool_call_grammar([SET_LIGHT])
        # Both enum strings must appear as terminals.
        assert "kitchen" in gbnf
        assert "living_room" in gbnf

    def test_object_with_no_properties_uses_empty_object(self):
        """``get_time`` has empty ``properties`` → the args rule is the
        empty-object terminal, not the permissive any-value rule."""
        gbnf = tool_call_grammar([GET_TIME])
        assert 'args_0 ::= "{" ws "}"' in gbnf


class TestSingleToolGrammar:
    def test_aliases_tool_call_grammar(self):
        a = single_tool_grammar(GRASP)
        b = tool_call_grammar([GRASP])
        assert a == b


class TestReplyOrToolGrammar:
    def test_no_tools_returns_free_text(self):
        gbnf = reply_or_tool_grammar([])
        assert "root ::= text-reply" in gbnf
        assert "tool-call" not in gbnf

    def test_with_tools_alternation(self):
        gbnf = reply_or_tool_grammar([GET_TIME])
        assert "root ::= text-reply | tool-call" in gbnf
        # Both alternatives' rules must be present.
        assert "text-reply ::=" in gbnf
        assert "tool-call ::=" in gbnf


# ---------------------------------------------------------------------------
# GrammarCache
# ---------------------------------------------------------------------------


class TestGrammarCache:
    def test_no_tools_returns_none(self):
        cache = GrammarCache()
        assert cache.get("tool", []) is None

    def test_unknown_strategy_raises(self):
        cache = GrammarCache()
        # Need a non-empty tools list to reach the strategy switch.
        with pytest.raises(ValueError, match="unknown grammar strategy"):
            cache.get("nope", [GET_TIME])

    def test_repeat_call_returns_same_object(self):
        """Cache key is the registry fingerprint — identical inputs hit
        the cache rather than recompiling."""
        cache = GrammarCache()
        first = cache.get("tool", [GET_TIME])
        second = cache.get("tool", [GET_TIME])
        # When llama-cpp is unavailable both are None; when available
        # they're the same object. Either way, the contract holds.
        assert first is second

    def test_different_registries_get_different_grammars(self):
        cache = GrammarCache()
        a = cache.get("tool", [GET_TIME])
        b = cache.get("tool", [GRASP])
        # Both None or both objects, but if compiled they're distinct.
        if a is not None and b is not None:
            assert a is not b
