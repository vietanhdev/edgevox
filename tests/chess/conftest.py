"""Shared fixtures for the chess integration tests.

A ``FakeEngine`` implements the :class:`ChessEngine` Protocol without
launching any subprocess. It mirrors how the harness SLM tests use
``LLM`` stubs — fast, deterministic, offline. Tests that specifically
need a real Stockfish / LC0 binary are marked ``integration`` and get
``skip`` when the binary isn't present on ``$PATH``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field

import chess
import pytest

from edgevox.integrations.chess.engine import EngineMove


@dataclass
class FakeEngine:
    """Deterministic test double for :class:`ChessEngine`.

    ``scores`` is a queue of centipawn scores the engine returns in
    order — one per ``bestmove`` / ``analyse`` call. When exhausted,
    the engine returns 0 cp so tests don't have to over-specify.

    The "best" move is always the first legal move in python-chess's
    iteration order; that's enough for environment / classification
    tests which care about *state transitions* not playing strength.
    """

    name: str = "fake"
    scores: list[int] = field(default_factory=list)

    def _next_score(self) -> int:
        if self.scores:
            return self.scores.pop(0)
        return 0

    def _first_legal(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("no legal moves")
        return legal[0]

    def bestmove(self, board: chess.Board, *, time_limit: float = 1.0) -> EngineMove:
        move = self._first_legal(board)
        san = board.san(move)
        return EngineMove(uci=move.uci(), san=san, eval_cp=self._next_score(), pv=[san])

    def analyse(self, board: chess.Board, *, depth: int = 12) -> EngineMove:
        move = self._first_legal(board)
        san = board.san(move)
        return EngineMove(uci=move.uci(), san=san, eval_cp=self._next_score(), pv=[san], depth=depth)

    def close(self) -> None:
        return None


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def stockfish_available() -> bool:
    return shutil.which("stockfish") is not None
