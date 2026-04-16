"""Integration test: real Gemma LLM drives the agent tool loop.

Downloads the default Gemma 4 E2B model and exercises the full
round-trip: register tools → model emits a tool_call → we dispatch →
model relays the result in plain text.

Slow — only run on release. Skips automatically if the model or its
chat template don't support tools.
"""

from __future__ import annotations

import pytest

from edgevox.llm import LLM, ToolCallResult, tool


@tool
def get_current_temperature(city: str) -> dict:
    """Return the current temperature for a city.

    Args:
        city: the city name
    """
    # Hard-coded so the model is forced to actually call the tool.
    return {"city": city, "temp_c": 17, "conditions": "cloudy"}


@tool
def add_numbers(a: float, b: float) -> float:
    """Return the sum of two numbers.

    Args:
        a: first number
        b: second number
    """
    return a + b


@pytest.fixture(scope="module")
def agent_llm():
    try:
        seen: list[ToolCallResult] = []
        llm = LLM(
            language="en",
            tools=[get_current_temperature, add_numbers],
            on_tool_call=seen.append,
        )
        llm._seen = seen  # type: ignore[attr-defined]
        return llm
    except Exception as e:
        pytest.skip(f"LLM not available: {e}")


class TestLLMAgentReal:
    def test_system_prompt_mentions_tools(self, agent_llm):
        assert "tools available" in agent_llm._history[0]["content"]

    def test_temperature_tool_roundtrip(self, agent_llm):
        agent_llm.reset()
        agent_llm._seen.clear()
        reply = agent_llm.chat("What's the temperature in Paris right now? Answer in one sentence.")
        assert isinstance(reply, str)
        assert reply
        # Model should have called the tool at least once
        assert any(r.name == "get_current_temperature" for r in agent_llm._seen), (
            f"Model did not call the tool. Reply={reply!r}"
        )

    def test_arithmetic_tool_roundtrip(self, agent_llm):
        agent_llm.reset()
        agent_llm._seen.clear()
        reply = agent_llm.chat("Use the add_numbers tool to compute 17 plus 25.")
        assert isinstance(reply, str)
        # Either the model called the tool, or at least produced the correct answer
        called = any(r.name == "add_numbers" for r in agent_llm._seen)
        assert called or "42" in reply, f"No tool call and wrong answer: {reply!r}"

    def test_chit_chat_does_not_call_tools(self, agent_llm):
        agent_llm.reset()
        agent_llm._seen.clear()
        reply = agent_llm.chat("Say hi in one word.")
        assert isinstance(reply, str)
        assert len(agent_llm._seen) == 0
