"""ChessEnvironment tests — legal moves, game over, undo, listeners."""

from __future__ import annotations

import pytest

from edgevox.agents.sim import SimEnvironment
from edgevox.integrations.chess.environment import ChessEnvironment, ChessState


@pytest.fixture
def env(fake_engine):
    e = ChessEnvironment(fake_engine, user_plays="white")
    yield e
    e.close()


class TestChessEnvironmentBasics:
    def test_implements_sim_environment_protocol(self, env):
        assert isinstance(env, SimEnvironment)

    def test_starting_position_has_20_legal_moves(self, env):
        assert len(env.list_legal_moves()) == 20

    def test_snapshot_returns_chess_state(self, env):
        state = env.snapshot()
        assert isinstance(state, ChessState)
        assert state.turn == "white"
        assert state.ply == 0
        assert state.is_game_over is False

    def test_get_world_state_is_json_dict(self, env):
        data = env.get_world_state()
        assert isinstance(data, dict)
        assert "fen" in data and "turn" in data


class TestPlayUserMove:
    def test_uci_move_applies(self, env):
        state = env.play_user_move("e2e4")
        assert state.last_move_uci == "e2e4"
        assert state.last_move_san == "e4"
        assert state.turn == "black"

    def test_san_move_applies(self, env):
        state = env.play_user_move("e4")
        assert state.last_move_san == "e4"

    def test_illegal_move_raises_without_mutating(self, env):
        with pytest.raises(ValueError):
            env.play_user_move("e2e5")
        # Board still pristine.
        assert env.snapshot().ply == 0

    def test_wrong_turn_raises(self, fake_engine):
        env = ChessEnvironment(fake_engine, user_plays="black")
        with pytest.raises(ValueError, match="engine's turn"):
            env.play_user_move("e2e4")


class TestEngineMove:
    def test_engine_move_advances_ply(self, env):
        env.play_user_move("e4")
        state, move = env.engine_move()
        assert state.ply == 2
        assert move.uci
        assert state.turn == "white"

    def test_engine_move_wrong_turn_raises(self, env):
        with pytest.raises(ValueError, match="user's turn"):
            env.engine_move()


class TestUndo:
    def test_undo_reverts_last_move(self, env):
        env.play_user_move("e4")
        state = env.undo_last_move()
        assert state.ply == 0
        assert state.last_move_uci is None

    def test_undo_on_empty_board_raises(self, env):
        with pytest.raises(ValueError, match="no moves to undo"):
            env.undo_last_move()


class TestNewGame:
    def test_new_game_resets_board(self, env):
        env.play_user_move("e4")
        state = env.new_game()
        assert state.ply == 0
        assert state.last_move_uci is None

    def test_new_game_switches_sides(self, env):
        env.new_game(user_plays="black")
        assert env.user_plays == "black"
        assert env.engine_plays == "white"


class TestGameOverDetection:
    def test_fools_mate_is_checkmate(self, env):
        # Fool's mate: 1. f3 e5 2. g4 Qh4#
        env.play_user_move("f3")
        env.engine_move()  # black plays whatever first-legal move FakeEngine picks
        # Reset and play a scripted sequence using direct board pushes via
        # play_user_move + mock legal engine response. For a fast test we
        # bypass the engine and push moves through the board.
        env.new_game()
        for move in ("f3", "g4"):
            env.play_user_move(move)
            # Black replies via raw board so we control the sequence.
            env._board.push_san("e5" if move == "f3" else "Qh4#")
            env._last_move = env._board.peek()
            env._san_history.append("e5" if move == "f3" else "Qh4#")
        state = env.snapshot()
        assert state.is_game_over is True
        assert state.game_over_reason == "checkmate"
        assert state.winner == "black"

    def test_cannot_play_after_game_over(self, env):
        env.play_user_move("f3")
        env._board.push_san("e5")
        env.play_user_move("g4")
        env._board.push_san("Qh4#")
        with pytest.raises(ValueError, match="already over"):
            env.play_user_move("a3")


class TestListeners:
    def test_subscribed_listener_fires_on_move(self, env):
        captured: list[ChessState] = []
        env.subscribe(captured.append)
        env.play_user_move("e4")
        # Starting value isn't captured (subscribed after init); one move → one callback.
        assert len(captured) == 1
        assert captured[0].last_move_san == "e4"

    def test_reset_fires_listener_snapshot(self, env):
        captured: list[ChessState] = []
        env.subscribe(captured.append)
        env.reset()
        assert len(captured) == 1
        assert captured[0].ply == 0

    def test_raising_listener_does_not_break_move(self, env):
        def broken(_state):
            raise RuntimeError("nope")

        env.subscribe(broken)
        # Must not propagate the listener's exception.
        state = env.play_user_move("e4")
        assert state.last_move_san == "e4"
