"""Tests for ReActAgent's mid-execution periodic verifier hook.

The hook fires inside the inner LLMAgent's AFTER_TOOL fire point
and short-circuits the loop with a TASK COMPLETE marker when the
verifier confirms task completion. The outer _ReActRunner recheck
loop then accepts the result without another LLM round-trip.
"""

from __future__ import annotations

import pytest

from edgevox.agents.hooks import AFTER_TOOL
from edgevox.agents.workflow_recipes import ReActAgent, _PeriodicVerifierHook


class _FakeCtx:
    """Minimal AgentContext stand-in for hook unit tests."""

    def __init__(self):
        self.hook_state: dict = {}


# ---------------------------------------------------------------------------
# Direct hook unit tests (no LLM, no agent loop)
# ---------------------------------------------------------------------------


class TestPeriodicVerifierHook:
    def test_fires_only_on_after_tool(self):
        """Hook should be a no-op for unrelated fire points."""
        h = _PeriodicVerifierHook(every_n=1, verifier=lambda ctx: True)
        ctx = _FakeCtx()
        # Wrong fire point - should return None even with verifier=True.
        assert h("before_tool", ctx, payload=None) is None
        assert h("after_llm", ctx, payload=None) is None
        # Right fire point - should fire.
        result = h(AFTER_TOOL, ctx, payload=None)
        assert result is not None
        assert result.is_end

    def test_counts_calls_and_fires_every_n(self):
        """Counter increments on each AFTER_TOOL; verifier consulted
        only on multiples of N."""
        consult_count = 0

        def verifier(ctx):
            nonlocal consult_count
            consult_count += 1
            return False  # never done

        h = _PeriodicVerifierHook(every_n=3, verifier=verifier)
        ctx = _FakeCtx()
        for _ in range(7):
            h(AFTER_TOOL, ctx, payload=None)
        # 7 calls / every 3 -> verifier consulted at calls 3 and 6 -> 2x.
        assert consult_count == 2

    def test_short_circuits_with_task_complete_marker(self):
        """When verifier returns True, hook ends turn with a reply
        containing the TASK COMPLETE marker so the outer recheck
        loop accepts termination."""
        h = _PeriodicVerifierHook(every_n=2, verifier=lambda ctx: True)
        ctx = _FakeCtx()
        # First call: verifier not consulted (count=1, not multiple of 2).
        assert h(AFTER_TOOL, ctx, payload=None) is None
        # Second call: verifier consulted, returns True, hook ends turn.
        result = h(AFTER_TOOL, ctx, payload=None)
        assert result is not None
        assert result.is_end
        assert "TASK COMPLETE" in result.payload
        assert "2 actions" in result.payload  # count in the reply

    def test_swallows_verifier_exception(self):
        """A broken verifier shouldn't crash the agent loop -- treat
        an exception as 'not done yet'."""

        def boom(ctx):
            raise RuntimeError("intentional")

        h = _PeriodicVerifierHook(every_n=1, verifier=boom)
        ctx = _FakeCtx()
        result = h(AFTER_TOOL, ctx, payload=None)
        assert result is None  # silent fall-through

    def test_state_keyed_by_id_self(self):
        """Two instances of the hook on the same context must keep
        separate counters."""
        h1 = _PeriodicVerifierHook(every_n=2, verifier=lambda ctx: False)
        h2 = _PeriodicVerifierHook(every_n=2, verifier=lambda ctx: False)
        ctx = _FakeCtx()
        # Fire each hook once on the shared ctx; counters must not collide.
        h1(AFTER_TOOL, ctx, payload=None)
        h1(AFTER_TOOL, ctx, payload=None)
        # h1's count is now 2 (multiple of 2 -> verifier consulted).
        # h2 hasn't fired yet -> its count must be 0.
        assert ctx.hook_state[id(h1)]["count"] == 2
        assert id(h2) not in ctx.hook_state

    def test_zero_every_n_is_inert(self):
        """``every_n=0`` should never fire (defensive: avoid div-by-zero
        and surprise activation if a caller passes 0 by accident)."""
        called = False

        def verifier(ctx):
            nonlocal called
            called = True
            return True

        h = _PeriodicVerifierHook(every_n=0, verifier=verifier)
        ctx = _FakeCtx()
        for _ in range(10):
            assert h(AFTER_TOOL, ctx, payload=None) is None
        assert called is False


# ---------------------------------------------------------------------------
# Build-level integration: ReActAgent.build wires the hook
# ---------------------------------------------------------------------------


class TestReActBuildWiring:
    def test_no_hook_when_param_omitted(self):
        runner = ReActAgent.build(verifier=lambda ctx: True)
        # The inner agent's hook list shouldn't include our class.
        inner_hooks = list(runner._inner._hooks)
        assert not any(isinstance(h, _PeriodicVerifierHook) for h in inner_hooks)

    def test_hook_installed_when_both_params_set(self):
        runner = ReActAgent.build(
            verifier=lambda ctx: True,
            verify_every_n_actions=4,
        )
        inner_hooks = list(runner._inner._hooks)
        periodic = [h for h in inner_hooks if isinstance(h, _PeriodicVerifierHook)]
        assert len(periodic) == 1
        assert periodic[0].every_n == 4

    def test_no_hook_without_verifier(self):
        """``verify_every_n_actions`` without a verifier is a no-op
        (nothing to verify against)."""
        runner = ReActAgent.build(verify_every_n_actions=3)
        inner_hooks = list(runner._inner._hooks)
        assert not any(isinstance(h, _PeriodicVerifierHook) for h in inner_hooks)

    def test_legacy_completion_check_keyword_works(self):
        """The legacy ``completion_check`` keyword should also enable
        the periodic hook."""
        runner = ReActAgent.build(
            completion_check=lambda ctx: True,
            verify_every_n_actions=2,
        )
        inner_hooks = list(runner._inner._hooks)
        assert any(isinstance(h, _PeriodicVerifierHook) for h in inner_hooks)

    def test_caller_hooks_are_preserved(self):
        """Caller-supplied hooks must coexist with the periodic
        verifier -- not get overwritten."""
        from edgevox.agents.hooks import AFTER_LLM, hook

        @hook(AFTER_LLM)
        def custom(point, ctx, payload):
            return None

        runner = ReActAgent.build(
            verifier=lambda ctx: True,
            verify_every_n_actions=2,
            hooks=[custom],
        )
        inner_hooks = list(runner._inner._hooks)
        assert custom in inner_hooks
        assert any(isinstance(h, _PeriodicVerifierHook) for h in inner_hooks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
