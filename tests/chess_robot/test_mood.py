"""Mood derivation + gaze conversion — pure functions, no framework imports."""

from __future__ import annotations

import pytest

from edgevox.examples.agents.chess_robot.mood import (
    AMUSED_CP,
    DEFEATED_CP,
    TRIUMPHANT_CP,
    WORRIED_CP,
    Mood,
    derive_mood,
    gaze_from_uci,
)


def _state(**overrides) -> dict:
    """Minimal ChessState-shaped dict with sensible defaults."""
    base = {
        "fen": "",
        "ply": 0,
        "turn": "white",
        "last_move_uci": None,
        "last_move_san": None,
        "last_move_classification": None,
        "san_history": [],
        "eval_cp": None,
        "mate_in": None,
        "win_prob_white": 0.5,
        "opening": None,
        "is_game_over": False,
        "game_over_reason": None,
        "winner": None,
    }
    base.update(overrides)
    return base


class TestMoodFromEval:
    def test_zero_eval_is_calm(self):
        assert derive_mood(_state(eval_cp=0), engine_plays="black") is Mood.CALM

    def test_engine_slight_advantage_is_still_calm(self):
        assert derive_mood(_state(eval_cp=AMUSED_CP - 1), engine_plays="white") is Mood.CALM

    def test_engine_decisive_is_amused(self):
        assert derive_mood(_state(eval_cp=AMUSED_CP + 1), engine_plays="white") is Mood.AMUSED

    def test_engine_crushing_is_triumphant(self):
        assert derive_mood(_state(eval_cp=TRIUMPHANT_CP + 50), engine_plays="white") is Mood.TRIUMPHANT

    def test_engine_losing_is_worried(self):
        assert derive_mood(_state(eval_cp=WORRIED_CP - 1), engine_plays="white") is Mood.WORRIED

    def test_engine_losing_badly_is_defeated(self):
        assert derive_mood(_state(eval_cp=DEFEATED_CP - 1), engine_plays="white") is Mood.DEFEATED

    def test_eval_is_flipped_for_engine_playing_black(self):
        """white-POV +500 means black (engine) is *losing*, so face should be defeated."""
        assert derive_mood(_state(eval_cp=500), engine_plays="black") is Mood.DEFEATED

    def test_eval_flipped_favourable_for_black_engine(self):
        assert derive_mood(_state(eval_cp=-500), engine_plays="black") is Mood.TRIUMPHANT


class TestMoodFromMateIn:
    def test_engine_white_mate_positive_is_triumphant(self):
        assert derive_mood(_state(mate_in=3), engine_plays="white") is Mood.TRIUMPHANT

    def test_engine_white_mate_negative_is_defeated(self):
        assert derive_mood(_state(mate_in=-3), engine_plays="white") is Mood.DEFEATED

    def test_engine_black_mate_flipped(self):
        assert derive_mood(_state(mate_in=3), engine_plays="black") is Mood.DEFEATED
        assert derive_mood(_state(mate_in=-3), engine_plays="black") is Mood.TRIUMPHANT


class TestMoodFromGameOver:
    def test_engine_wins_is_triumphant(self):
        s = _state(is_game_over=True, winner="white", game_over_reason="checkmate")
        assert derive_mood(s, engine_plays="white") is Mood.TRIUMPHANT

    def test_user_wins_is_defeated(self):
        s = _state(is_game_over=True, winner="white", game_over_reason="checkmate")
        assert derive_mood(s, engine_plays="black") is Mood.DEFEATED

    def test_draw_is_calm(self):
        s = _state(is_game_over=True, winner=None, game_over_reason="stalemate")
        assert derive_mood(s, engine_plays="white") is Mood.CALM


class TestMoodFromClassification:
    def test_user_blunder_makes_engine_amused(self):
        """Last move was a user blunder, engine's turn now."""
        s = _state(
            turn="white",  # engine's turn after black's blunder
            last_move_classification="blunder",
            eval_cp=50,  # not yet decisive
        )
        assert derive_mood(s, engine_plays="white") is Mood.AMUSED

    def test_engine_own_blunder_does_not_amuse(self):
        """Last move was engine's (user's turn now) — classification
        refers to engine's move, so the face shouldn't gloat."""
        s = _state(
            turn="black",  # user's turn after engine's blunder
            last_move_classification="blunder",
            eval_cp=50,
        )
        assert derive_mood(s, engine_plays="white") is Mood.CALM

    def test_grandmaster_persona_stays_composed(self):
        s = _state(turn="white", last_move_classification="blunder", eval_cp=50)
        assert derive_mood(s, engine_plays="white", persona="grandmaster") is Mood.CURIOUS

    def test_trash_talker_stays_amused_at_triumphant_level(self):
        s = _state(eval_cp=600)
        assert derive_mood(s, engine_plays="white", persona="trash_talker") is Mood.AMUSED


class TestMoodAcceptsBothDictAndObject:
    def test_dataclass_state(self):
        from edgevox.integrations.chess.environment import ChessState

        state = ChessState(fen="", ply=0, turn="white", eval_cp=250)
        assert derive_mood(state, engine_plays="white") is Mood.AMUSED


class TestGazeFromUci:
    def test_none_is_centre(self):
        assert gaze_from_uci(None) == (0.0, 0.0)

    def test_empty_is_centre(self):
        assert gaze_from_uci("") == (0.0, 0.0)

    def test_destination_drives_gaze(self):
        # e4 → file 4 (e), rank 3 (idx) → (0.143, -0.143)
        x, y = gaze_from_uci("e2e4")
        assert -0.2 < x < 0.2
        assert -0.2 < y < 0.2

    def test_corner_a1(self):
        assert gaze_from_uci("a2a1") == (-1.0, -1.0)

    def test_corner_h8(self):
        assert gaze_from_uci("h7h8") == (1.0, 1.0)

    def test_garbage_returns_centre(self):
        assert gaze_from_uci("zzzz") == (0.0, 0.0)

    @pytest.mark.parametrize(
        ("uci", "x_sign", "y_sign"),
        [
            ("e2e4", 1, -1),  # e4 → slight right, lower half
            ("e7e5", 1, 1),  # e5 → slight right, upper half
            ("a1a8", -1, 1),  # a8 → left edge, top
            ("h1a1", -1, -1),  # a1 → left edge, bottom
        ],
    )
    def test_quadrant_signs(self, uci, x_sign, y_sign):
        x, y = gaze_from_uci(uci)
        # Signs encode which side of centre the gaze lands on.
        assert (x == 0) or ((x > 0) == (x_sign > 0))
        assert (y == 0) or ((y > 0) == (y_sign > 0))
