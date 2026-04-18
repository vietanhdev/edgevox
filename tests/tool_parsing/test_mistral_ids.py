"""Regression: Mistral's 9-char tool-call id must round-trip.

Mistral Nemo / Ministral templates require the tool-call id emitted in
``[TOOL_CALLS] [{"name":..., "arguments":..., "id":"abc123xyz"}]`` to
match the ``tool_call_id`` of the subsequent ``role="tool"`` message.
When the agent loop synthesised a new id, the model couldn't pair its
call with the result and the conversation degraded.

These tests lock in:

1. ``MistralDetector.detect_and_parse`` preserves the id on the
   ``ToolCallItem``.
2. ``_to_openai_tool_call`` surfaces the id through to the outer dict.
3. ``LLMAgent._drive`` threads the id into the follow-up ``role="tool"``
   message verbatim.
"""

from __future__ import annotations

import json

import pytest

from edgevox.llm.tool_parsers import MistralDetector, coerce_tools, parse_tool_calls

_GET_TIME_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Return the current time.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def test_detector_preserves_9char_id():
    """Canonical Mistral format carries an id on the JSON payload."""
    raw = '[TOOL_CALLS] [{"name": "get_time", "arguments": {}, "id": "abc123xyz"}]'
    detector = MistralDetector()
    result = detector.detect_and_parse(raw, coerce_tools(_GET_TIME_TOOL))
    assert len(result.calls) == 1
    item = result.calls[0]
    assert item.name == "get_time"
    assert item.id == "abc123xyz"


def test_openai_wrapping_uses_model_id():
    raw = '[TOOL_CALLS] [{"name": "get_time", "arguments": {}, "id": "abc123xyz"}]'
    openai_calls = parse_tool_calls(raw, _GET_TIME_TOOL, detectors=["mistral"])
    assert openai_calls and openai_calls[0]["id"] == "abc123xyz"


def test_openai_wrapping_synthesises_when_absent():
    """No id on the payload → a deterministic synthetic id is used."""
    raw = '[TOOL_CALLS] [{"name": "get_time", "arguments": {}}]'
    openai_calls = parse_tool_calls(raw, _GET_TIME_TOOL, detectors=["mistral"])
    assert openai_calls
    assert openai_calls[0]["id"] == "get_time_0"


@pytest.fixture
def _llm_agent_env():
    from edgevox.agents import AgentContext, LLMAgent
    from edgevox.llm.tools import tool

    @tool
    def get_time() -> str:
        """Return the current time."""
        return "noon"

    return AgentContext, LLMAgent, get_time


def test_tool_result_round_trip_preserves_id(_llm_agent_env, scripted_llm_factory):
    """Full loop: assistant emits a tool_call with id=abc123xyz, the
    next message must be role=tool with tool_call_id=abc123xyz — not a
    synthesised ``call_0_get_time``."""
    AgentContext, LLMAgent, get_time = _llm_agent_env

    mistral_call = {
        "content": None,
        "tool_calls": [
            {
                "id": "abc123xyz",
                "function": {"name": "get_time", "arguments": "{}"},
            }
        ],
    }
    final_reply = {"content": "It's noon.", "tool_calls": None}
    llm = scripted_llm_factory([mistral_call, final_reply])

    agent = LLMAgent(
        name="tester",
        description="",
        instructions="",
        tools=[get_time],
        llm=llm,
    )
    ctx = AgentContext()
    agent.run("What time is it?", ctx)

    # Find the role=tool message in the session history.
    tool_msgs = [m for m in ctx.session.messages if m.get("role") == "tool"]
    assert tool_msgs, "expected a role=tool message after dispatch"
    assert tool_msgs[0]["tool_call_id"] == "abc123xyz"
    # And the result payload still round-trips.
    assert json.loads(tool_msgs[0]["content"])["ok"] is True
