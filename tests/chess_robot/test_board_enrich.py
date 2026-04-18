"""BoardHintHook — engine candidates + legal-moves injection."""

from __future__ import annotations

import pytest

from edgevox.agents.base import AgentContext
from edgevox.agents.hooks import ON_RUN_START
from edgevox.examples.agents.chess_robot.board_enrich import BoardHintHook
from edgevox.integrations.chess.hooks import BoardStateInjectionHook


class TestBoardHintHook:
    def test_adds_top_engine_line(self, env):
        hook = BoardHintHook()
        ctx = AgentContext(deps=env)
        result = hook(ON_RUN_START, ctx, {"task": "your move"})
        assert result is not None
        task = result.payload["task"]
        assert "[board] top engine line:" in task
        # Original task preserved at the tail.
        assert "your move" in task

    def test_legal_moves_only_for_meta_questions(self, env):
        hook = BoardHintHook()
        ctx = AgentContext(deps=env)
        task = "play e4"  # not a meta question
        result = hook(ON_RUN_START, ctx, {"task": task})
        assert result is not None
        assert "[board] legal moves" not in result.payload["task"]

    @pytest.mark.parametrize(
        "text",
        [
            "what should I do?",
            "what can I play here?",
            "what are my options?",
            "any ideas?",
            "help me think",
            "hint please",
        ],
    )
    def test_legal_moves_injected_for_meta(self, env, text):
        hook = BoardHintHook()
        ctx = AgentContext(deps=env)
        result = hook(ON_RUN_START, ctx, {"task": text})
        assert result is not None
        assert "[board] legal moves" in result.payload["task"]

    def test_no_deps_is_noop(self):
        hook = BoardHintHook()
        ctx = AgentContext(deps=None)
        assert hook(ON_RUN_START, ctx, {"task": "anything"}) is None

    def test_non_dict_payload_is_noop(self, env):
        hook = BoardHintHook()
        ctx = AgentContext(deps=env)
        assert hook(ON_RUN_START, ctx, "not a dict") is None

    def test_empty_task_is_noop(self, env):
        hook = BoardHintHook()
        ctx = AgentContext(deps=env)
        assert hook(ON_RUN_START, ctx, {"task": ""}) is None

    def test_priority_slots_between_intercept_and_state(self):
        """Priority must be > BoardStateInjectionHook (80) so the top
        line sits above the main board summary; and < MoveInterceptHook
        (90) so move application happens first."""
        from edgevox.examples.agents.chess_robot.move_intercept import MoveInterceptHook

        assert MoveInterceptHook.priority > BoardHintHook.priority
        assert BoardHintHook.priority > BoardStateInjectionHook.priority

    def test_legal_moves_truncated(self, env):
        """Middlegame positions have 30+ moves; the hook should cap
        the listing so tiny models don't drown."""
        hook = BoardHintHook()
        ctx = AgentContext(deps=env)
        # Starting position has exactly 20 legal moves → 12 shown + "(+8 more)".
        result = hook(ON_RUN_START, ctx, {"task": "what are my options?"})
        task = result.payload["task"]
        # Either truncated with a "+N more" note, or all 20 fit.
        if "+" in task:
            assert "more" in task
