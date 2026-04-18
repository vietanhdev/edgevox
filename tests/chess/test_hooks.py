"""Hook tests — BoardStateInjectionHook + MoveCommentaryHook + personas."""

from __future__ import annotations

import pytest

from edgevox.agents.base import AgentContext
from edgevox.agents.hooks import AFTER_TOOL, ON_RUN_START, HookAction
from edgevox.integrations.chess.environment import ChessEnvironment
from edgevox.integrations.chess.hooks import BoardStateInjectionHook, MoveCommentaryHook
from edgevox.integrations.chess.personas import PERSONAS, resolve_persona
from edgevox.llm.tools import ToolCallResult


@pytest.fixture
def env(fake_engine):
    e = ChessEnvironment(fake_engine, user_plays="white")
    yield e
    e.close()


@pytest.fixture
def ctx(env):
    return AgentContext(deps=env)


class TestBoardStateInjectionHook:
    def test_fires_only_at_on_run_start(self):
        hook = BoardStateInjectionHook()
        assert hook.points == {ON_RUN_START}

    def test_injects_fen_and_turn(self, ctx):
        hook = BoardStateInjectionHook()
        result = hook(ON_RUN_START, ctx, {"task": "your move"})
        assert result is not None
        assert result.action is HookAction.MODIFY
        new_task = result.payload["task"]
        assert "[board] fen:" in new_task
        assert "side to move: white" in new_task
        assert "your move" in new_task  # original task preserved

    def test_skips_without_chess_deps(self):
        hook = BoardStateInjectionHook()
        plain = AgentContext(deps=object())
        result = hook(ON_RUN_START, plain, {"task": "hi"})
        assert result is None

    def test_includes_recent_moves(self, env, ctx):
        env.play_user_move("e4")
        env.engine_move()  # FakeEngine plays first legal black move
        hook = BoardStateInjectionHook(include_history_plies=4)
        result = hook(ON_RUN_START, ctx, {"task": "what next?"})
        assert "recent moves" in result.payload["task"]

    def test_snapshot_recorded_in_hook_state(self, ctx):
        hook = BoardStateInjectionHook()
        hook(ON_RUN_START, ctx, {"task": "go"})
        assert "last_prefix" in ctx.hook_state[id(hook)]


class TestMoveCommentaryHook:
    def test_fires_only_at_after_tool(self):
        hook = MoveCommentaryHook()
        assert hook.points == {AFTER_TOOL}

    def test_captures_engine_move_state(self, env, ctx):
        hook = MoveCommentaryHook()
        env.play_user_move("e4")
        outcome = ToolCallResult(name="engine_move", arguments={}, result={"move_san": "e5"})
        hook(AFTER_TOOL, ctx, outcome)
        bucket = ctx.hook_state[id(hook)]
        assert bucket["last_tool"] == "engine_move"
        assert bucket["last_state"]["last_move_san"] is not None

    def test_ignores_unrelated_tools(self, ctx):
        hook = MoveCommentaryHook()
        outcome = ToolCallResult(name="get_board_state", arguments={}, result={})
        result = hook(AFTER_TOOL, ctx, outcome)
        assert result is None
        assert id(hook) not in ctx.hook_state

    def test_ignores_failed_tool_calls(self, ctx):
        hook = MoveCommentaryHook()
        outcome = ToolCallResult(name="play_user_move", arguments={}, error="illegal")
        hook(AFTER_TOOL, ctx, outcome)
        assert id(hook) not in ctx.hook_state


class TestPersonas:
    def test_all_three_registered(self):
        assert set(PERSONAS) == {"grandmaster", "casual", "trash_talker"}

    def test_resolve_is_case_insensitive(self):
        assert resolve_persona("Casual").slug == "casual"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown persona"):
            resolve_persona("mittens")

    def test_each_persona_has_system_prompt_and_engine(self):
        for p in PERSONAS.values():
            assert p.system_prompt
            assert p.engine_kind in ("stockfish", "maia")
