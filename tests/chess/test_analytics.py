"""Analytics tests — classification thresholds + opening recognition + win prob."""

from __future__ import annotations

import chess
import pytest

from edgevox.integrations.chess.analytics import (
    MoveClassification,
    classify_move,
    opening_name,
    win_probability,
)


class TestClassifyMove:
    @pytest.mark.parametrize(
        ("cp_swing", "expected"),
        [
            (0, MoveClassification.BEST),
            (10, MoveClassification.BEST),
            (11, MoveClassification.GOOD),
            (50, MoveClassification.GOOD),
            (51, MoveClassification.INACCURACY),
            (150, MoveClassification.INACCURACY),
            (151, MoveClassification.MISTAKE),
            (300, MoveClassification.MISTAKE),
            (301, MoveClassification.BLUNDER),
            (1000, MoveClassification.BLUNDER),
        ],
    )
    def test_thresholds(self, cp_swing, expected):
        assert classify_move(cp_swing) is expected

    def test_negative_swing_is_absolute_value(self):
        assert classify_move(-300) is MoveClassification.MISTAKE
        assert classify_move(-301) is MoveClassification.BLUNDER

    def test_none_becomes_blunder(self):
        assert classify_move(None) is MoveClassification.BLUNDER


class TestWinProbability:
    def test_zero_cp_is_fifty_percent(self):
        assert win_probability(0) == pytest.approx(0.5)

    def test_large_positive_eval_is_high_win_prob(self):
        assert win_probability(500) > 0.85

    def test_large_negative_eval_is_low_win_prob(self):
        assert win_probability(-500) < 0.15

    def test_mate_in_positive_is_one(self):
        assert win_probability(None, mate_in=3) == 1.0

    def test_mate_in_negative_is_zero(self):
        assert win_probability(None, mate_in=-3) == 0.0

    def test_none_eval_without_mate_is_draw(self):
        assert win_probability(None) == 0.5


class TestOpeningName:
    def test_starting_position_is_none(self):
        assert opening_name(chess.Board()) is None

    def test_sicilian_detected(self):
        board = chess.Board()
        board.push_san("e4")
        board.push_san("c5")
        assert opening_name(board) == "Sicilian Defence"

    def test_french_defence_detected(self):
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e6")
        assert opening_name(board) == "French Defence"

    def test_unknown_position_returns_none(self):
        board = chess.Board()
        board.push_san("a3")
        board.push_san("h6")
        assert opening_name(board) is None

    def test_kings_pawn_detected_after_one_move(self):
        board = chess.Board()
        board.push_san("e4")
        assert opening_name(board) == "King's Pawn"
