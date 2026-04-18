"""MoveInterceptHook — deterministic move application before the LLM sees the task."""

from __future__ import annotations

import pytest

from edgevox.agents.base import AgentContext
from edgevox.agents.hooks import ON_RUN_START, HookAction
from edgevox.examples.agents.chess_robot.move_intercept import (
    MoveInterceptHook,
    extract_move,
)


class TestExtractMove:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("e2e4", "e2e4"),
            ("play e2e4", "e2e4"),
            ("e4", "e4"),
            ("play e4", "e4"),
            ("Nf3", "Nf3"),
            ("I play Nf3", "Nf3"),
            ("Nxd5", "Nxd5"),
            ("O-O", "O-O"),
            ("castle kingside", "O-O"),
            ("castles king", "O-O"),
            ("O-O-O", "O-O-O"),
            ("castle queenside", "O-O-O"),
            ("knight to c3", "Nc3"),
            ("knight takes d5", "Nxd5"),
            ("bishop to f4", "Bf4"),
            ("queen to h5", "Qh5"),
        ],
    )
    def test_parses(self, text, expected):
        assert extract_move(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "what do you think?",
            "how is my position",
            "hey rook",
            "good luck",
            "cool4ever",  # false-positive guard
        ],
    )
    def test_no_match(self, text):
        assert extract_move(text) is None

    def test_case_sensitive_san(self):
        """SAN is case-sensitive: `N` is knight, `n` is a square file noise.
        Our regex requires uppercase piece letters (SAN convention)."""
        assert extract_move("ng1") is None  # not a valid SAN (lowercase N)
        assert extract_move("Ng1") == "Ng1"


@pytest.fixture
def env():
    from edgevox.integrations.chess.environment import ChessEnvironment
    from tests.chess.conftest import FakeEngine

    e = ChessEnvironment(FakeEngine(), user_plays="white")
    try:
        yield e
    finally:
        e.close()


class TestInterceptMove:
    def test_happy_path_applies_and_rewrites_task(self, env):
        hook = MoveInterceptHook()
        ctx = AgentContext(deps=env)
        result = hook(ON_RUN_START, ctx, {"task": "play e4"})
        assert result is not None
        assert result.action is HookAction.MODIFY
        new_task = result.payload["task"]
        assert "e4" in new_task
        # The task is now first-person from the user's POV — "I just played e4".
        assert "i just played" in new_task.lower() or "user played" in new_task.lower()
        # Both moves applied: user's e4 + engine's reply.
        state = env.snapshot()
        assert state.ply == 2

    def test_illegal_move_passes_through_with_note(self, env):
        hook = MoveInterceptHook()
        ctx = AgentContext(deps=env)
        result = hook(ON_RUN_START, ctx, {"task": "play e5"})  # illegal as white opening
        assert result is not None
        new_task = result.payload["task"]
        assert "[intercept]" in new_task or "could not be applied" in new_task
        assert env.snapshot().ply == 0  # board untouched

    def test_no_move_pattern_is_noop(self, env):
        hook = MoveInterceptHook()
        ctx = AgentContext(deps=env)
        result = hook(ON_RUN_START, ctx, {"task": "how do you think I'm doing?"})
        assert result is None
        assert env.snapshot().ply == 0

    def test_new_game_resets_board(self, env):
        hook = MoveInterceptHook()
        ctx = AgentContext(deps=env)
        env.play_user_move("e4")
        result = hook(ON_RUN_START, ctx, {"task": "new game"})
        assert result is not None
        assert env.snapshot().ply == 0
        assert "new game" in result.payload["task"].lower()

    def test_undo_rolls_back(self, env):
        hook = MoveInterceptHook()
        ctx = AgentContext(deps=env)
        env.play_user_move("e4")
        before_ply = env.snapshot().ply
        result = hook(ON_RUN_START, ctx, {"task": "undo"})
        assert result is not None
        assert env.snapshot().ply == before_ply - 1

    def test_undo_on_empty_board_is_graceful(self, env):
        hook = MoveInterceptHook()
        ctx = AgentContext(deps=env)
        result = hook(ON_RUN_START, ctx, {"task": "undo"})
        assert result is not None
        assert "could not" in result.payload["task"].lower() or "no move" in result.payload["task"].lower()

    def test_no_deps_is_noop(self):
        hook = MoveInterceptHook()
        ctx = AgentContext(deps=None)
        assert hook(ON_RUN_START, ctx, {"task": "play e4"}) is None

    def test_non_dict_payload_is_noop(self, env):
        hook = MoveInterceptHook()
        ctx = AgentContext(deps=env)
        assert hook(ON_RUN_START, ctx, "play e4") is None

    def test_runs_before_board_injection(self):
        """Priority must be higher than BoardStateInjectionHook's 80 so
        the board injection sees the post-move state."""
        from edgevox.integrations.chess.hooks import BoardStateInjectionHook

        assert MoveInterceptHook.priority > BoardStateInjectionHook.priority

    def test_fires_only_at_on_run_start(self):
        assert MoveInterceptHook().points == {ON_RUN_START}
