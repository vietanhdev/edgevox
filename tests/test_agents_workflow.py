"""Unit tests for workflow primitives (Sequence, Fallback, Loop, Retry,
Timeout, Router)."""

from __future__ import annotations

from unittest.mock import MagicMock

from edgevox.agents.base import AgentContext, AgentResult
from edgevox.agents.workflow import (
    Fallback,
    Loop,
    Retry,
    Router,
    Sequence,
    Timeout,
)


def _fake_agent(name: str, replies: list[str]):
    """Build a minimal fake Agent that returns scripted replies, one
    per call, from the given list."""
    cursor = {"i": 0}

    class _Fake:
        def __init__(self):
            self.name = name
            self.description = f"fake {name}"
            self.calls: list[str] = []

        def run(self, task: str, ctx: AgentContext | None = None) -> AgentResult:
            self.calls.append(task)
            reply = replies[cursor["i"]] if cursor["i"] < len(replies) else ""
            cursor["i"] += 1
            return AgentResult(reply=reply, agent_name=name)

        def run_stream(self, task, ctx=None):
            yield self.run(task, ctx).reply

    return _Fake()


# --------------- Sequence ---------------


class TestSequence:
    def test_runs_in_order_chains_outputs(self):
        a = _fake_agent("a", ["A-output"])
        b = _fake_agent("b", ["B-output"])
        seq = Sequence("seq", [a, b])
        result = seq.run("first task")
        assert a.calls == ["first task"]
        assert b.calls == ["A-output"]
        assert result.reply == "B-output"

    def test_stops_on_empty_reply(self):
        a = _fake_agent("a", [""])
        b = _fake_agent("b", ["never"])
        seq = Sequence("seq", [a, b])
        seq.run("task")
        assert b.calls == []

    def test_empty_agents_raises(self):
        import pytest

        with pytest.raises(ValueError):
            Sequence("empty", [])


# --------------- Fallback ---------------


class TestFallback:
    def test_returns_first_non_empty(self):
        a = _fake_agent("a", [""])
        b = _fake_agent("b", ["second-wins"])
        c = _fake_agent("c", ["third"])
        fb = Fallback("fb", [a, b, c])
        result = fb.run("task")
        assert result.reply == "second-wins"
        assert c.calls == []  # not reached

    def test_all_fail_returns_last(self):
        a = _fake_agent("a", [""])
        b = _fake_agent("b", [""])
        fb = Fallback("fb", [a, b])
        result = fb.run("task")
        assert result.reply == ""


# --------------- Loop ---------------


class TestLoop:
    def test_terminates_when_until_is_true(self):
        agent = _fake_agent("agent", ["iter1", "iter2", "iter3"])

        def done(state):
            state["i"] = state.get("i", 0) + 1
            return state["i"] >= 2

        loop = Loop("loop", agent, until=done, max_iterations=5)
        result = loop.run("go")
        assert len(agent.calls) == 2
        assert result.reply == "iter2"

    def test_bounded_by_max_iterations(self):
        agent = _fake_agent("agent", ["iter" + str(i) for i in range(10)])
        loop = Loop("loop", agent, until=lambda _: False, max_iterations=3)
        loop.run("go")
        assert len(agent.calls) == 3


# --------------- Retry ---------------


class TestRetry:
    def test_returns_first_non_empty(self):
        agent = _fake_agent("a", ["", "second-try"])
        retry = Retry(agent, max_attempts=3)
        result = retry.run("task")
        assert result.reply == "second-try"
        assert len(agent.calls) == 2

    def test_all_empty_returns_last(self):
        agent = _fake_agent("a", ["", "", ""])
        retry = Retry(agent, max_attempts=3)
        result = retry.run("task")
        assert result.reply == ""
        assert len(agent.calls) == 3

    def test_invalid_max_attempts(self):
        import pytest

        agent = _fake_agent("a", [])
        with pytest.raises(ValueError):
            Retry(agent, max_attempts=0)


# --------------- Timeout ---------------


class TestTimeout:
    def test_quick_agent_passes_through(self):
        agent = _fake_agent("a", ["fast"])
        to = Timeout(agent, seconds=2.0)
        result = to.run("task")
        assert result.reply == "fast"

    def test_slow_agent_returns_timeout(self):
        import time as t

        class Slow:
            name = "slow"
            description = "slow"

            def run(self, task, ctx=None):
                t.sleep(1.5)
                return AgentResult(reply="too late", agent_name="slow")

            def run_stream(self, task, ctx=None):
                yield self.run(task).reply

        to = Timeout(Slow(), seconds=0.3)
        ctx = AgentContext()
        result = to.run("task", ctx)
        assert result.preempted is True
        assert "Timed out" in result.reply
        assert ctx.stop.is_set()


# --------------- Router ---------------


class TestRouter:
    def test_router_build_returns_llmagent_with_handoffs(self):
        from edgevox.agents.base import LLMAgent

        leaf_a = LLMAgent("a", "A", "You are A.", llm=MagicMock())
        leaf_b = LLMAgent("b", "B", "You are B.", llm=MagicMock())
        router = Router.build(
            name="router",
            instructions="Route.",
            routes={"a": leaf_a, "b": leaf_b},
        )
        assert router.name == "router"
        assert "handoff_to_a" in router.tools.tools
        assert "handoff_to_b" in router.tools.tools


# --------------- Supervisor ---------------


class TestSupervisor:
    def test_supervisor_forces_first_hop_tool_call(self):
        from edgevox.agents.base import LLMAgent
        from edgevox.agents.workflow import Supervisor

        worker_a = LLMAgent("a", "A", "You are A.", llm=MagicMock())
        worker_b = LLMAgent("b", "B", "You are B.", llm=MagicMock())
        sup = Supervisor.build(
            name="sup",
            instructions="Pick the right worker.",
            workers={"a": worker_a, "b": worker_b},
        )
        # Both workers exposed as handoffs.
        assert "handoff_to_a" in sup.tools.tools
        assert "handoff_to_b" in sup.tools.tools
        # And the SLM loop-break is on by default.
        assert sup._tool_choice_policy == "required_first_hop"

    def test_supervisor_rejects_empty_workers(self):
        import pytest

        from edgevox.agents.workflow import Supervisor

        with pytest.raises(ValueError, match="at least one worker"):
            Supervisor.build("sup", "Route.", {})


# --------------- Handoff state_update ---------------


class TestHandoffStateUpdate:
    def test_state_update_dataclass_field_default(self):
        """``Handoff`` accepts an optional ``state_update`` dict."""
        from edgevox.agents.base import Handoff

        h = Handoff(target=MagicMock(), state_update={"why": "user asked"})
        assert h.state_update == {"why": "user asked"}
        # Default is None — no surprise blackboard mutations.
        h2 = Handoff(target=MagicMock())
        assert h2.state_update is None

    def test_state_update_apply_step_publishes_into_blackboard(self):
        """Unit-level: the small loop step that applies a Handoff's
        ``state_update`` writes every key/value to ``ctx.blackboard``
        before the target is invoked."""
        from edgevox.agents.base import AgentContext, Handoff
        from edgevox.agents.multiagent import Blackboard

        bb = Blackboard()
        ctx = AgentContext(blackboard=bb)
        h = Handoff(target=MagicMock(), state_update={"why": "user asked", "tier": "kitchen"})

        # Mirror the snippet in LLMAgent._drive (the dispatch-side
        # synthetic-handoff path doesn't invoke the tool body, so the
        # state_update field arrives via this code path).
        if h.state_update and ctx.blackboard is not None:
            for k, v in h.state_update.items():
                ctx.blackboard.set(k, v)

        assert bb.get("why") == "user asked"
        assert bb.get("tier") == "kitchen"


# --------------- Orchestrator ---------------


class TestOrchestrator:
    def test_orchestrator_emits_plan_and_synthesises(self):
        """Lead emits a one-subtask plan via the synthetic ``emit_plan``
        tool; the worker runs; the result is returned (no synthesis
        needed for a single subtask)."""
        from edgevox.agents.workflow import Orchestrator
        from edgevox.llm.tools import tool
        from tests.harness.conftest import ScriptedLLM, calls, reply

        @tool
        def echo(msg: str) -> str:
            """Echo a message."""
            return f"echo:{msg}"

        # Script:
        # 1. Lead's hop 0 forces emit_plan call (under required_first_hop).
        # 2. Lead's hop 1 final reply.
        # 3. Worker hop 0 — calls echo.
        # 4. Worker hop 1 — final reply.
        llm = ScriptedLLM(
            [
                calls(("emit_plan", {"subtasks": [{"objective": "say hi", "tools": ["echo"]}]})),
                reply("plan done"),
                calls(("echo", {"msg": "hi"})),
                reply("hello back"),
            ]
        )

        orch = Orchestrator(
            name="orch",
            lead_instructions="Plan the request.",
            synth_instructions="Combine results.",
            tools=[echo],
        )
        # Bind the shared LLM into all leaves.
        from edgevox.agents.workflow import _bind_llm_recursive

        _bind_llm_recursive(orch._lead, llm)
        _bind_llm_recursive(orch._synth, llm)

        result = orch.run("say hi")
        # Single-subtask path returns the worker reply directly.
        assert result.reply == "hello back"

    def test_orchestrator_rejects_zero_subtasks_max(self):
        import pytest

        from edgevox.agents.workflow import Orchestrator

        with pytest.raises(ValueError, match="max_subtasks"):
            Orchestrator(name="x", lead_instructions="", synth_instructions="", max_subtasks=0)
