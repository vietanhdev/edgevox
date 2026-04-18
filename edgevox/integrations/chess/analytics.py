"""Analytics helpers — eval thresholds, move classification, opening names.

Thin utilities the agent tools call after each move so the reply can
carry commentary ("that was a mistake — the engine rates this -180
centipawns") and the web UI can render badges on the move list.
"""

from __future__ import annotations

import math
from enum import Enum

import chess


class MoveClassification(str, Enum):
    """How a move compares to the engine's best line before it was played.

    Thresholds are in centipawns of eval swing against the mover. The
    bands mirror what Lichess / chess.com show next to each move so the
    voice commentary ("that was a blunder") stays consistent with what
    users see on the board.
    """

    BEST = "best"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"


_BAND_EDGES_CP: tuple[tuple[int, MoveClassification], ...] = (
    (10, MoveClassification.BEST),
    (50, MoveClassification.GOOD),
    (150, MoveClassification.INACCURACY),
    (300, MoveClassification.MISTAKE),
)


def classify_move(cp_swing: int | None) -> MoveClassification:
    """Bucket a centipawn eval swing into a human-readable label.

    Args:
        cp_swing: how many centipawns the mover lost by playing this
            move instead of the engine's top choice. ``None`` (e.g. the
            position went straight to mate) is treated as a blunder.
    """
    if cp_swing is None:
        return MoveClassification.BLUNDER
    magnitude = abs(int(cp_swing))
    for edge, label in _BAND_EDGES_CP:
        if magnitude <= edge:
            return label
    return MoveClassification.BLUNDER


def win_probability(eval_cp: int | None, *, mate_in: int | None = None) -> float:
    """Estimate white's win probability from a centipawn eval.

    Uses the Lichess-style logistic with ``k = 0.004`` — not rigorous
    but widely accepted and intuitively calibrated (±200 cp ≈ 69/31).
    ``mate_in`` short-circuits to 0 or 1.

    Returns a float in ``[0.0, 1.0]``.
    """
    if mate_in is not None:
        return 1.0 if mate_in > 0 else 0.0
    if eval_cp is None:
        return 0.5
    return 1.0 / (1.0 + math.exp(-0.004 * eval_cp))


# ---------------------------------------------------------------------------
# Opening lookup
# ---------------------------------------------------------------------------

# Curated short list — enough for the agent to name the most common
# openings without shipping a polyglot book. Keys are the FEN of the
# position *after* the move sequence; values are the ECO-style name.
# The agent tool falls back to "unknown opening" for anything off-book.
_OPENING_TABLE: dict[str, str] = {
    # After 1.e4
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1": "King's Pawn",
    # Sicilian
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": "Sicilian Defence",
    # French
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": "French Defence",
    # Caro-Kann
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": "Caro-Kann Defence",
    # After 1.d4
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1": "Queen's Pawn",
    # Queen's Gambit
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2": "Queen's Gambit",
    # Indian Defence (1.d4 Nf6)
    "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2": "Indian Defence",
    # Ruy Lopez (after 3.Bb5)
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": "Ruy Lopez",
    # Italian Game (after 3.Bc4)
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": "Italian Game",
    # English (1.c4)
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1": "English Opening",
    # Reti (1.Nf3)
    "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKBNR b KQkq - 1 1": "Réti Opening",
}


def opening_name(board: chess.Board) -> str | None:
    """Return a human-readable opening name if the current position is in
    the curated table, else ``None``.

    Uses the current position's FEN so transpositions (the same
    resulting position reached by different move orders) match too.
    The halfmove and fullmove counters are stripped so the lookup is
    order-independent.
    """
    fen = board.fen()
    if fen in _OPENING_TABLE:
        return _OPENING_TABLE[fen]
    # Also match ignoring move counters (last two FEN fields).
    fen_stem = " ".join(fen.split(" ")[:4])
    for key, name in _OPENING_TABLE.items():
        if " ".join(key.split(" ")[:4]) == fen_stem:
            return name
    return None


__all__ = [
    "MoveClassification",
    "classify_move",
    "opening_name",
    "win_probability",
]
