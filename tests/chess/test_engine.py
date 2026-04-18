"""Engine-layer tests — Protocol conformance + Stockfish smoke test."""

from __future__ import annotations

import shutil

import chess
import pytest

from edgevox.integrations.chess.engine import (
    ChessEngine,
    EngineUnavailable,
    StockfishEngine,
    build_engine,
)


class TestFakeEngineImplementsProtocol:
    def test_fake_is_chess_engine(self, fake_engine):
        assert isinstance(fake_engine, ChessEngine)

    def test_bestmove_returns_engine_move(self, fake_engine):
        board = chess.Board()
        move = fake_engine.bestmove(board)
        assert move.uci and move.san
        # First legal move in starting position is either a knight or pawn — pick any.
        assert chess.Move.from_uci(move.uci) in board.legal_moves

    def test_analyse_reports_depth(self, fake_engine):
        board = chess.Board()
        move = fake_engine.analyse(board, depth=8)
        assert move.depth == 8


class TestEngineMoveScoreNormalisation:
    def test_score_from_white_returns_eval_cp_by_default(self):
        from edgevox.integrations.chess.engine import EngineMove

        move = EngineMove(uci="e2e4", san="e4", eval_cp=35)
        assert move.score_from_white == 35

    def test_mate_in_positive_clamps_to_large_white_advantage(self):
        from edgevox.integrations.chess.engine import EngineMove

        move = EngineMove(uci="e2e4", san="e4", mate_in=2)
        assert move.score_from_white == 10_000

    def test_mate_in_negative_clamps_to_large_black_advantage(self):
        from edgevox.integrations.chess.engine import EngineMove

        move = EngineMove(uci="e2e4", san="e4", mate_in=-3)
        assert move.score_from_white == -10_000


class TestBuildEngineFactory:
    def test_unknown_kind_raises(self):
        with pytest.raises(EngineUnavailable, match="unknown chess engine"):
            build_engine("alphazero")

    def test_maia_without_weights_raises(self):
        with pytest.raises((EngineUnavailable, TypeError)):
            build_engine("maia")


@pytest.mark.skipif(shutil.which("stockfish") is None, reason="stockfish binary not installed")
class TestStockfishIntegration:
    """Runs only when stockfish is available — CI-friendly skip otherwise."""

    def test_bestmove_from_starting_position(self):
        with StockfishEngine(skill=5) as engine:
            move = engine.bestmove(chess.Board(), time_limit=0.1)
            assert move.uci
            assert chess.Move.from_uci(move.uci) in chess.Board().legal_moves

    def test_analyse_returns_score_and_pv(self):
        with StockfishEngine(skill=5) as engine:
            move = engine.analyse(chess.Board(), depth=6)
            assert move.pv
            # Starting position evaluation should be near 0 (within engine noise).
            assert move.eval_cp is None or abs(move.eval_cp) < 200

    def test_stockfish_unavailable_raises_when_binary_path_bogus(self, tmp_path):
        with pytest.raises(EngineUnavailable):
            StockfishEngine(binary=tmp_path / "nonexistent_stockfish")
