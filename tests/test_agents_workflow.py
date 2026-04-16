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
