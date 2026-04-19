"""Legality check for the ``eval_llm_commentary.scenarios()`` corpus.

Every scenario we feed through the LLM benchmark must replay to a
legal chess position — otherwise the directive's description
helpers (``_describe_move`` replays the pre-move board to name the
captured piece) silently return ``None`` and the LLM sees a
truncated briefing that no longer tests what the scenario claims.

This test runs on every CI invocation so a future contributor who
adds a fresh scenario with a typo'd SAN fails fast locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import pytest

# The eval harness lives under ``scripts/`` which isn't on the package
# path in the default pytest config — prepend the repo root so the
# import resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.eval_llm_commentary import scenarios


@pytest.mark.parametrize("scn", scenarios(), ids=lambda s: s.name)
def test_scenario_replays_legally(scn):
    """Replay every SAN in ``scn.san_history`` on a fresh
    :class:`chess.Board`. Any ``ValueError`` from ``push_san`` means
    the sequence isn't a real game — either a typo or a stale edit
    against a moved-on position."""
    board = chess.Board()
    for i, san in enumerate(scn.san_history):
        try:
            board.push_san(san)
        except ValueError as e:
            pytest.fail(f"scenario {scn.name!r}: ply {i + 1} SAN {san!r} is illegal on position {board.fen()!r}: {e}")


def test_terminal_scenarios_have_winner_or_draw_flag():
    """If ``is_game_over`` is set, the scenario must either name a
    winner or mark itself a draw. The gate's game-over branch picks
    the tone cue off this — silently missing both fields dropped the
    model into neutral ground with the wrong prompt."""
    for scn in scenarios():
        if not scn.is_game_over:
            continue
        reason = (scn.game_over_reason or "").lower()
        has_winner = bool(scn.winner)
        is_draw = reason in ("stalemate", "insufficient material", "threefold repetition", "fifty-move rule", "draw")
        assert has_winner or is_draw, (
            f"scenario {scn.name!r} claims is_game_over=True but has no winner and no draw reason"
        )
