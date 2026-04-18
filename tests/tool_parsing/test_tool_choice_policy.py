"""Integration tests for ``LLMAgent(tool_choice_policy=...)``.

Validates the OpenAI Agents SDK loop-break: under
``required_first_hop`` the LLM gets ``tool_choice="required"`` + a
GBNF grammar on hop 0, then ``"auto"`` on subsequent hops so the
final reply can land. ``ScriptedLLM`` records the kwargs passed to
``complete()`` so we can assert the policy without needing llama-cpp.
"""

from __future__ import annotations

from edgevox.agents.base import LLMAgent
from edgevox.llm.tools import tool
from tests.harness.conftest import ScriptedLLM, call, reply


@tool
def get_time() -> str:
    """Return the current time."""
    return "noon"


def _agent(llm, *, policy="auto"):
    return LLMAgent(
        name="t",
        description="",
        instructions="",
        llm=llm,
        tools=[get_time],
        tool_choice_policy=policy,
    )


class TestToolChoicePolicy:
    def test_auto_does_not_force_or_constrain(self):
        llm = ScriptedLLM([reply("ok")])
        _agent(llm, policy="auto").run("hi")

        assert llm.calls
        kw = llm.calls[0]
        # Default for "auto": no tool_choice, no grammar.
        assert kw["tool_choice"] in (None, "auto")

    def test_required_first_hop_forces_tool_then_releases(self):
        # Hop 0 must end with a tool call; hop 1 (after tool runs) is the
        # final reply.
        llm = ScriptedLLM([call("get_time"), reply("It's noon.")])
        _agent(llm, policy="required_first_hop").run("what time?")

        assert len(llm.calls) == 2
        assert llm.calls[0]["tool_choice"] == "required"
        # Second hop is unconstrained so the model can answer.
        assert llm.calls[1]["tool_choice"] in (None, "auto")

    def test_required_always_keeps_forcing_each_hop(self):
        # Two tool hops + a final reply on the last allowed hop.
        llm = ScriptedLLM(
            [
                call("get_time"),
                call("get_time"),
                reply("done"),
            ]
        )
        _agent(llm, policy="required_always").run("loop?")

        # Every hop until the budget exhausts had tool_choice="required".
        for kw in llm.calls[:-1]:
            assert kw["tool_choice"] == "required"

    def test_no_tools_means_no_constraint(self):
        """An agent with no tools registered must not crash under a
        ``required_*`` policy — the policy gracefully no-ops."""
        llm = ScriptedLLM([reply("hi")])
        agent = LLMAgent(
            name="t",
            description="",
            instructions="",
            llm=llm,
            tool_choice_policy="required_first_hop",
        )
        agent.run("hi")
        assert llm.calls[0]["tool_choice"] in (None, "auto")
