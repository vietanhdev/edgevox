"""LLM-bound commentary evaluation harness.

Runs a curated set of chess scenarios through the actual LLM that the
RookApp would use, grades the replies against the GROUND TRUTH the
:class:`CommentaryGateHook` would hand the model, and prints a report
with hotspots worth tuning.

Usage::

    python scripts/eval_llm_commentary.py

The script downloads the default preset (``llama-3.2-1b``) to the
HuggingFace cache if it's not already there. Override the model via
``--model <preset-slug>``.

This is not a pytest — it's meant to be run ad hoc while iterating on
the directive wording / mood cues / anti-fabrication guards. A
scenario prints:

* the directive that was injected into the system prompt;
* the rewritten user task (what :class:`MoveInterceptHook` would feed);
* the raw LLM reply;
* a list of grading flags ("mentioned `pin` when no pin was declared",
  "restated SAN verbatim", "restated ground-truth bullet", etc.);
* overall score for the scenario (0-100).

Aggregate scores across scenarios drive decisions on prompt tuning.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field

# Package imports — make sure the working tree is on sys.path when
# running as a plain script from the repo root.
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgevox.agents.base import AgentContext, Session
from edgevox.examples.agents.chess_robot.commentary_gate import (
    _build_ground_truth,
    _record_turn_history,
)
from edgevox.examples.agents.chess_robot.prompts import ROOK_TOOL_GUIDANCE as _ROOK_TOOL_GUIDANCE
from edgevox.integrations.chess.analytics import MoveClassification, classify_move
from edgevox.llm import LLM

# ---------------------------------------------------------------------------
# Fake env + state — mirror the test harness so we don't need stockfish.
# ---------------------------------------------------------------------------


@dataclass
class _FakeState:
    san_history: list[str] = field(default_factory=list)
    last_move_san: str | None = None
    last_move_classification: MoveClassification | None = None
    eval_cp: int | None = None
    mate_in: int | None = None
    is_game_over: bool = False
    game_over_reason: str | None = None
    winner: str | None = None


class _FakeEnv:
    def __init__(self, *, user_plays: str = "white") -> None:
        self.user_plays = user_plays
        self.engine_plays = "black" if user_plays == "white" else "white"
        self._state = _FakeState()

    def set_state(self, s: _FakeState) -> None:
        self._state = s

    def snapshot(self) -> _FakeState:
        return self._state


# ---------------------------------------------------------------------------
# Scenario definitions. Each simulates one turn with a specific tactical
# shape, carries the expected tone (for grading), and declares words the
# reply MUST NOT invent.
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    description: str
    san_history: list[str]
    user_plays: str = "white"
    eval_cp: int | None = None
    classification: MoveClassification | None = None
    is_game_over: bool = False
    game_over_reason: str | None = None
    winner: str | None = None
    greeted_before: bool = True
    # Expected tone — 'confident' / 'rattled' / 'neutral' / 'final' /
    # 'opening'. Used to check the reply matches the situation.
    expected_tone: str = "neutral"
    # Words the model MUST NOT claim — things that would be fabricated.
    forbidden_terms: tuple[str, ...] = ()
    # Persona to feed the system prompt.
    persona: str = "casual"
    # User task string (what MoveInterceptHook would hand the LLM).
    user_task: str = ""


def recompute_with_stockfish(
    scns: list[Scenario],
    *,
    depth: int = 12,
    stockfish_path: str = "stockfish",
) -> list[Scenario]:
    """Replace each scenario's hand-set ``eval_cp`` / ``classification``
    with values computed by replaying the SAN through stockfish.

    Makes the benchmark match what a real RookApp game would see at
    that exact position, instead of my (often wrong) eyeballed guess.
    Each scenario:

    * replays ``san_history`` minus the last move → pre-move eval;
    * replays the full history → post-move eval + mate flag;
    * derives ``classification`` from the eval swing against the mover.

    Fails open — if stockfish isn't on ``$PATH`` the scenarios are
    returned unchanged so the benchmark still runs with the hand
    values, only louder.
    """
    try:
        import chess
        import chess.engine
    except Exception:
        print("stockfish recomputation skipped — python-chess engine API unavailable", file=sys.stderr)
        return scns

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        print(f"stockfish recomputation skipped — cannot start {stockfish_path!r}: {e}", file=sys.stderr)
        return scns

    updated: list[Scenario] = []
    try:
        for scn in scns:
            updated.append(_recompute_single(scn, engine, depth=depth))
    finally:
        engine.quit()
    return updated


def _recompute_single(scn: Scenario, engine, *, depth: int) -> Scenario:
    import chess
    import chess.engine

    board = chess.Board()
    if not scn.san_history:
        return scn
    # Pre-last-move analysis for classification.
    for san in scn.san_history[:-1]:
        try:
            board.push_san(san)
        except ValueError:
            return scn
    try:
        pre_info = engine.analyse(board, chess.engine.Limit(depth=depth))
    except Exception:
        return scn
    pre_score = pre_info["score"].white()
    pre_cp = pre_score.score() if not pre_score.is_mate() else None

    # Post-last-move analysis.
    try:
        last_move = board.parse_san(scn.san_history[-1])
        board.push(last_move)
    except ValueError:
        return scn
    try:
        post_info = engine.analyse(board, chess.engine.Limit(depth=depth))
    except Exception:
        return scn
    post_score = post_info["score"].white()
    post_cp = post_score.score() if not post_score.is_mate() else None
    # mate_in is computed but not threaded into ChessState here —
    # the gate's MOOD CUE branch reads it via state.mate_in, but
    # since our _FakeState mate_in defaults to None we just rely on
    # the eval_cp clamp (mate ⇒ effectively ±10000 cp).

    # Classification swing — centipawns lost by the mover against the
    # best line. Mirrors ChessEnvironment._classify_unlocked.
    cls: MoveClassification | None = scn.classification
    if pre_cp is not None and post_cp is not None:
        swing = pre_cp - post_cp if board.turn == chess.BLACK else -(pre_cp - post_cp)
        cls = classify_move(swing)

    # Copy the scenario; preserve is_game_over / winner (stockfish's
    # terminal detection runs via chess.Board.is_game_over() already,
    # but we respect the scenario's declared end-state).
    return Scenario(
        name=scn.name,
        description=scn.description,
        san_history=scn.san_history,
        user_plays=scn.user_plays,
        eval_cp=post_cp if post_cp is not None else scn.eval_cp,
        classification=cls,
        is_game_over=scn.is_game_over,
        game_over_reason=scn.game_over_reason,
        winner=scn.winner,
        greeted_before=scn.greeted_before,
        expected_tone=scn.expected_tone,
        forbidden_terms=scn.forbidden_terms,
        persona=scn.persona,
        user_task=scn.user_task,
    )


def scenarios() -> list[Scenario]:
    return [
        Scenario(
            name="opening_greeting",
            description="Game start, user plays e4, Rook replies e5. Should greet.",
            san_history=["e4", "e5"],
            eval_cp=15,
            classification=MoveClassification.BEST,
            greeted_before=False,
            expected_tone="opening",
            forbidden_terms=("pin", "fork", "skewer", "blunder"),
            user_task="I just played e4. You reply with e5. In your persona's voice, say one natural-sounding line about my move and yours. Mention only e4 and e5, no other moves, no analysis essays.",
        ),
        Scenario(
            name="user_hangs_bishop",
            description="User plays Ba6?? hanging the bishop; Rook captures with Nxa6. Rook is up material.",
            san_history=["e4", "e5", "Ba6", "Nxa6"],
            eval_cp=-350,  # white POV — black (Rook) up material
            classification=MoveClassification.BEST,
            expected_tone="confident",
            forbidden_terms=("pin", "fork", "skewer", "initiative", "bold"),
            user_task="I just played Ba6. You reply with Nxa6. In your persona's voice, say one natural-sounding line about my move and yours.",
        ),
        Scenario(
            name="rook_blunders_queen",
            description="Rook (black) blundered its queen on Qh4, user captured with Nxh4. Terminal user move; no engine reply. Rook is losing.",
            # Rook is BLACK. Black plays Qh4 (blunder), white plays Nxh4 (takes queen).
            # san_history length 5 → last is white (user). Terminal for engine.
            san_history=["e4", "e5", "Nf3", "Qh4", "Nxh4"],
            eval_cp=900,  # white POV — white (user) up a queen
            classification=MoveClassification.BEST,  # white's capture was best
            expected_tone="rattled",
            forbidden_terms=("pin", "fork", "skewer", "my advantage", "winning", "pawn storm"),
            user_task="I just played Nxh4. In your persona's voice, say one natural-sounding line about my move — I just captured your queen.",
        ),
        Scenario(
            name="user_checkmates",
            description="Scholar's mate by user. Game over.",
            san_history=["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"],
            eval_cp=10000,
            is_game_over=True,
            game_over_reason="checkmate",
            winner="white",
            expected_tone="final",
            forbidden_terms=("counterattack", "recover"),
            user_task="I just played Qxf7. You reply with nothing (game is over). In your persona's voice, say one natural-sounding line about my move.",
        ),
        Scenario(
            name="rook_checkmates",
            description="Rook delivered back-rank mate. Game over, Rook wins.",
            san_history=["f3", "e5", "g4", "Qh4#"],
            eval_cp=-10000,
            is_game_over=True,
            game_over_reason="checkmate",
            winner="black",
            expected_tone="final",
            forbidden_terms=("next time", "close game"),
            user_task="I just played g4. You reply with Qh4#. In your persona's voice, say one natural-sounding line about my move and yours.",
        ),
        Scenario(
            name="mutual_quiet_capture_trade",
            description="Knight-for-knight trade, position roughly even.",
            san_history=["e4", "e5", "Nf3", "Nc6", "Nc3", "Nf6", "Nxe5", "Nxe5"],
            eval_cp=20,
            classification=MoveClassification.BEST,
            expected_tone="neutral",
            forbidden_terms=("pin", "fork", "skewer", "winning"),
            user_task="I just played Nxe5. You reply with Nxe5. In your persona's voice, say one natural-sounding line about my move and yours.",
        ),
        Scenario(
            name="user_castles_kingside",
            description="User castles short in a quiet middlegame. Rook continues development.",
            san_history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O", "Nf6"],
            eval_cp=10,
            classification=MoveClassification.BEST,
            expected_tone="neutral",
            forbidden_terms=("pin", "fork", "skewer", "trapped", "attack"),
            user_task="I just played O-O (castled kingside). You reply with Nf6. In your persona's voice, say one natural-sounding line about my move and yours.",
        ),
        Scenario(
            name="user_promotes_pawn",
            description="User pushes pawn to 8th rank and promotes to a queen. Rook is losing.",
            san_history=["e4", "d5", "exd5", "c6", "dxc6", "Nf6", "cxb7", "Nbd7", "bxa8=Q"],
            eval_cp=1800,  # user up a queen
            classification=MoveClassification.BEST,
            expected_tone="rattled",
            forbidden_terms=("pin", "fork", "skewer", "my advantage", "gaining initiative"),
            user_task="I just played bxa8=Q (promoted pawn to queen on a8, capturing your rook). In your persona's voice, say one natural-sounding line about my move — I just promoted a pawn and took your rook.",
        ),
        Scenario(
            name="trash_talker_user_blunder",
            description="Trash-talker persona with a user blunder — persona voice should be sharp.",
            san_history=["e4", "e5", "Nf3", "Qh4", "Nxh4"],
            eval_cp=900,
            classification=MoveClassification.BLUNDER,
            expected_tone="confident",
            persona="trash_talker",
            forbidden_terms=("pin", "fork", "skewer", "bold", "nice try"),
            user_task="I just played Nxh4. In your persona's voice, say one natural-sounding line about my move — I just captured your queen.",
        ),
        # --- Color flip: user plays black, Rook plays white ---
        Scenario(
            name="user_plays_black_greeting",
            description="User plays BLACK. Rook opens as white with e4, user replies c5 (Sicilian). Greeting turn — tests pronoun correctness under flipped colours.",
            user_plays="black",
            san_history=["e4", "c5"],
            eval_cp=30,
            classification=MoveClassification.BEST,
            greeted_before=False,
            expected_tone="opening",
            forbidden_terms=("pin", "fork", "skewer"),
            user_task="I just played c5 (Sicilian). You opened with e4. In your persona's voice, say one natural-sounding line about my move and yours.",
        ),
        Scenario(
            name="user_black_captures_rook_queen",
            description="User (black) captures Rook's (white) queen after an early Qh5 sortie. Rook is down a queen, losing.",
            user_plays="black",
            # 1.e4 e5 2.Qh5 Nc6 3.Qxe5+ Nxe5 — white Qxe5+ takes a pawn
            # with check, black knight recaptures the queen.
            san_history=["e4", "e5", "Qh5", "Nc6", "Qxe5+", "Nxe5"],
            eval_cp=-800,  # white POV — black up a queen
            classification=MoveClassification.BLUNDER,  # white's queen was hanging
            expected_tone="rattled",
            forbidden_terms=("pin", "fork", "my advantage", "initiative", "winning"),
            user_task="I just played Nxe5, capturing your queen. In your persona's voice, say one natural-sounding line about my move.",
        ),
        # --- Terminal-state variety ---
        Scenario(
            name="stalemate_by_user",
            description="Stalemate — user has an overwhelming material lead but cornered Rook's king without delivering mate. Game ends in a draw.",
            san_history=["e4", "e5", "Qh5", "Nf6", "Qxf7+", "Kxf7", "Bc4+", "Kg6", "Nf3", "Nc6", "d3", "d6"],
            eval_cp=0,  # draw
            is_game_over=True,
            game_over_reason="stalemate",
            winner=None,
            expected_tone="final",
            forbidden_terms=("won", "lost", "winning", "losing"),
            user_task="The game just ended in stalemate. In your persona's voice, say one line about it.",
        ),
        # (draw_insufficient_material scenario removed — the 30-ply
        # sequence was non-trivial to keep legal through edits and
        # ``stalemate_by_user`` already covers the "game-over, no
        # winner" branch for benchmark purposes.)
        # --- Tactical / middle-game variety ---
        Scenario(
            name="user_brilliant_tactical_shot",
            description="User plays a BEST move that swung the eval by 200 cp in their favor — clean tactical hit. Rook should acknowledge the strong play without flattery.",
            san_history=[
                "e4",
                "c5",
                "Nf3",
                "Nc6",
                "d4",
                "cxd4",
                "Nxd4",
                "Nf6",
                "Nc3",
                "e5",
                "Ndb5",
                "d6",
                "Bg5",
                "a6",
                "Na3",
                "b5",
                "Nd5",
                "Nxd5",
                "exd5",
                "Nb4",
            ],
            eval_cp=-190,  # user (black) up clear
            classification=MoveClassification.BEST,
            expected_tone="neutral",
            forbidden_terms=("pin", "fork", "skewer", "amazing", "brilliant"),
            user_task="I just played Nb4 — a strong tactical move threatening your queen on d1. You reply with Nxb4. In your persona's voice, say one natural-sounding line about my move and yours.",
        ),
        Scenario(
            name="quiet_middle_pawn_push",
            description="Routine positional pawn push, no captures, eval stable. Gate should stay SILENT (no speech trigger).",
            san_history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "O-O", "O-O", "d6", "a4"],
            eval_cp=15,
            classification=MoveClassification.BEST,
            expected_tone="neutral",
            forbidden_terms=(),
            user_task="I just played a4. In your persona's voice, say one natural-sounding line about my move.",
        ),
        Scenario(
            name="en_passant_capture",
            description="User captures via en passant — unusual move shape, tests that the description handles the special capture.",
            # 1.e4 Nf6 2.e5 d5 3.exd6 — white pawn on e5 captures
            # black's d-pawn en passant as it pushes d7-d5.
            san_history=["e4", "Nf6", "e5", "d5", "exd6"],
            eval_cp=80,
            classification=MoveClassification.GOOD,
            expected_tone="neutral",
            forbidden_terms=("pin", "fork", "skewer"),
            user_task="I just played exd6 — en passant capture. In your persona's voice, say one natural-sounding line about my move.",
        ),
        # --- Endgame variety ---
        Scenario(
            name="user_queen_trap_wins",
            description="User captures Rook's queen via a f2 trap (early Qxf2+ → Kxf2 → winning the black queen). Rook is down a queen, losing decisively.",
            # 1.e4 e5 2.Nc3 Qf6 3.d3 Qxf2+ 4.Kxf2 Nf6 5.Nd5 Nxd5 6.exd5
            # — after all exchanges, white (user) is up roughly a queen.
            san_history=["e4", "e5", "Nc3", "Qf6", "d3", "Qxf2+", "Kxf2", "Nf6", "Nd5", "Nxd5", "exd5"],
            eval_cp=800,  # user up a queen
            classification=MoveClassification.BEST,
            expected_tone="rattled",
            forbidden_terms=("defend", "save", "counterattack", "winning"),
            user_task="I just played exd5, recapturing your knight. I already won your queen earlier. In your persona's voice, say one natural-sounding line about the position.",
        ),
        Scenario(
            name="minor_piece_trade_bishop_for_knight",
            description="Quiet minor-piece trade (bishop for knight), eval barely moves. Should speak because it's a capture pair.",
            san_history=["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Bxc6", "bxc6"],
            eval_cp=-20,
            classification=MoveClassification.GOOD,
            expected_tone="neutral",
            forbidden_terms=("pin", "fork", "skewer"),
            user_task="I just played Bxc6. You reply with bxc6. In your persona's voice, say one natural-sounding line about my move and yours.",
        ),
        # --- Persona cross-check ---
        Scenario(
            name="grandmaster_rook_wins_material",
            description="Grandmaster persona with Rook winning material — voice should be reserved, not giddy.",
            san_history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "d3", "Bc5", "Bg5", "h6", "Bxf6", "Qxf6"],
            eval_cp=-30,
            classification=MoveClassification.BEST,
            expected_tone="confident",
            persona="grandmaster",
            forbidden_terms=("woohoo", "yeehaw", "sweet", "awesome"),
            user_task="I just played Qxf6 after you played Bxf6. In your persona's voice, say one natural-sounding line about my move and yours.",
        ),
        # --- Opening variety (all verified legal) ---
        Scenario(
            name="sicilian_najdorf_early_trade",
            description="Sicilian Najdorf — mainline development, minor exchange on d4 in book.",
            san_history=[
                "e4",
                "c5",
                "Nf3",
                "d6",
                "d4",
                "cxd4",
                "Nxd4",
                "Nf6",
                "Nc3",
                "a6",
                "Be2",
                "e5",
                "Nb3",
                "Be7",
                "O-O",
                "O-O",
                "Be3",
                "Nbd7",
            ],
            eval_cp=-15,
            classification=MoveClassification.BEST,
            expected_tone="neutral",
            forbidden_terms=("pin", "fork", "skewer"),
            user_task="You just played Nbd7. In your persona's voice, say one natural-sounding line about your move.",
        ),
        Scenario(
            name="caro_kann_user_captures_pawn",
            description="Caro-Kann classical — user (white) recaptures on e4 with the knight. Book line.",
            san_history=["e4", "c6", "d4", "d5", "Nc3", "dxe4", "Nxe4", "Nd7", "Ng5", "Ngf6", "Bd3"],
            eval_cp=25,
            classification=MoveClassification.BEST,
            expected_tone="neutral",
            forbidden_terms=("pin", "fork", "skewer"),
            user_task="I just played Bd3. In your persona's voice, say one natural-sounding line about my move.",
        ),
        Scenario(
            name="french_defense_mid_game",
            description="French Winawer — black gives up bishop pair with Bxc3+. User plays white.",
            san_history=["e4", "e6", "d4", "d5", "Nc3", "Bb4", "e5", "c5", "a3", "Bxc3+", "bxc3", "Ne7"],
            eval_cp=30,
            classification=MoveClassification.BEST,
            expected_tone="neutral",
            forbidden_terms=("pin", "fork", "skewer"),
            user_task="I just played bxc3, recapturing your bishop. You reply with Ne7. In your persona's voice, say one natural-sounding line.",
        ),
        Scenario(
            name="london_system_slow",
            description="London System — very quiet, should be GATED SILENT.",
            san_history=["d4", "Nf6", "Nf3", "g6", "Bf4", "Bg7", "e3", "O-O", "h3", "d6", "Be2"],
            eval_cp=10,
            classification=MoveClassification.GOOD,
            expected_tone="neutral",
            forbidden_terms=(),
            user_task="I just played Be2. In your persona's voice, say one natural-sounding line.",
        ),
        Scenario(
            name="kings_indian_fianchetto",
            description="King's Indian fianchetto — mainline development, typical middlegame structure.",
            san_history=[
                "d4",
                "Nf6",
                "c4",
                "g6",
                "Nc3",
                "Bg7",
                "e4",
                "d6",
                "Nf3",
                "O-O",
                "Be2",
                "e5",
                "O-O",
                "Nc6",
                "d5",
                "Ne7",
            ],
            eval_cp=20,
            classification=MoveClassification.BEST,
            expected_tone="neutral",
            forbidden_terms=("pin", "fork", "skewer"),
            user_task="I just played d5 (closing the center). You reply with Ne7. In your persona's voice, say one natural-sounding line.",
        ),
        Scenario(
            name="opera_game_sac",
            description="Morphy's Opera Game — white sacrifices a knight for a fierce attack. User plays white, knight sac Nxb5.",
            san_history=[
                "e4",
                "e5",
                "Nf3",
                "d6",
                "d4",
                "Bg4",
                "dxe5",
                "Bxf3",
                "Qxf3",
                "dxe5",
                "Bc4",
                "Nf6",
                "Qb3",
                "Qe7",
                "Nc3",
                "c6",
                "Bg5",
                "b5",
                "Nxb5",
                "cxb5",
                "Bxb5+",
                "Nbd7",
            ],
            eval_cp=80,
            classification=MoveClassification.BEST,
            expected_tone="confident",
            forbidden_terms=("blunder", "mistake"),
            user_task="I just played Bxb5+ after your cxb5 (which took my sacrificed knight). In your persona's voice, say one natural-sounding line.",
        ),
        Scenario(
            name="user_double_attack_fork",
            description="Classic Nxf7 'fried liver' setup — user sacrifices knight for attack. Verified legal sequence.",
            san_history=[
                "e4",
                "e5",
                "Nf3",
                "Nc6",
                "Bc4",
                "Bc5",
                "d3",
                "Nf6",
                "Ng5",
                "O-O",
                "h3",
                "h6",
                "Nxf7",
                "Rxf7",
                "Bxf7+",
                "Kxf7",
                "Qh5+",
            ],
            eval_cp=-30,
            classification=MoveClassification.BEST,
            expected_tone="confident",
            forbidden_terms=(),
            user_task="I just played Qh5+. In your persona's voice, say one natural-sounding line.",
        ),
        Scenario(
            name="scholars_mate_threat_dodged",
            description="White threatens Scholar's mate; Rook (black) correctly defends with h6. Should speak on classification.",
            san_history=["e4", "e5", "Bc4", "Nf6", "Qf3", "Nc6", "d3", "h6"],
            eval_cp=30,
            classification=MoveClassification.BEST,
            expected_tone="neutral",
            forbidden_terms=("mate", "checkmate"),
            user_task="I just played d3. You reply with h6. In your persona's voice, say one natural-sounding line.",
        ),
        Scenario(
            name="smothered_mate_threat",
            description="Classic smothered mate pattern; user (white) gets mated by Nf3#.",
            san_history=[
                "e4",
                "e5",
                "Nf3",
                "Nc6",
                "Bc4",
                "Nd4",
                "Nxe5",
                "Qg5",
                "Nxf7",
                "Qxg2",
                "Rf1",
                "Qxe4+",
                "Be2",
                "Nf3#",
            ],
            is_game_over=True,
            game_over_reason="checkmate",
            winner="black",
            eval_cp=-10000,
            expected_tone="final",
            forbidden_terms=("next time", "close game"),
            user_task="I just played Be2 trying to block. You reply with Nf3#. In your persona's voice, say one natural-sounding line.",
        ),
        Scenario(
            name="berlin_defense_quiet_castle",
            description="Berlin Defense Endgame — queens trade early, Rook's (black) king walks to queenside.",
            san_history=[
                "e4",
                "e5",
                "Nf3",
                "Nc6",
                "Bb5",
                "Nf6",
                "O-O",
                "Nxe4",
                "d4",
                "Nd6",
                "Bxc6",
                "dxc6",
                "dxe5",
                "Nf5",
                "Qxd8+",
                "Kxd8",
                "Nc3",
            ],
            eval_cp=20,
            classification=MoveClassification.BEST,
            expected_tone="neutral",
            forbidden_terms=("pin", "fork"),
            user_task="I just played Nc3. In your persona's voice, say one natural-sounding line.",
        ),
        Scenario(
            name="queens_gambit_accepted",
            description="Queen's Gambit Accepted — book opening, no tactics, quiet development.",
            san_history=[
                "d4",
                "d5",
                "c4",
                "dxc4",
                "Nf3",
                "Nf6",
                "e3",
                "e6",
                "Bxc4",
                "c5",
                "O-O",
                "a6",
                "Qe2",
                "b5",
                "Bb3",
                "Bb7",
            ],
            eval_cp=15,
            classification=MoveClassification.BEST,
            expected_tone="neutral",
            forbidden_terms=("pin", "fork", "skewer"),
            user_task="I just played Bb3. You reply with Bb7. In your persona's voice, say one natural-sounding line.",
        ),
        Scenario(
            name="italian_game_trade",
            description="Italian Game — Giuoco Pianissimo, slow development.",
            san_history=[
                "e4",
                "e5",
                "Nf3",
                "Nc6",
                "Bc4",
                "Bc5",
                "c3",
                "Nf6",
                "d3",
                "d6",
                "O-O",
                "O-O",
                "Re1",
                "h6",
                "Nbd2",
                "a5",
                "Nf1",
            ],
            eval_cp=10,
            classification=MoveClassification.GOOD,
            expected_tone="neutral",
            forbidden_terms=(),
            user_task="I just played Nf1. In your persona's voice, say one natural-sounding line.",
        ),
        # --- Color-flipped: Rook plays WHITE (user plays black) ---
        Scenario(
            name="rook_white_captures_user_bishop",
            description="Rook (white) captures user's (black) bishop after Bxe4 in the Nxe4 exchange.",
            user_plays="black",
            san_history=["Nf3", "b6", "e4", "Bb7", "Nc3", "Bxe4", "Nxe4"],
            eval_cp=200,
            classification=MoveClassification.BEST,
            expected_tone="confident",
            forbidden_terms=("pin", "fork", "skewer"),
            user_task="I just played Bxe4 — I captured your pawn. You reply with Nxe4, capturing my bishop. In your persona's voice, say one natural-sounding line.",
        ),
        Scenario(
            name="rook_white_delivers_mate",
            description="Rook (white) delivers Scholar's mate — mirror of user_checkmates. Rook wins.",
            user_plays="black",
            san_history=["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"],
            is_game_over=True,
            game_over_reason="checkmate",
            winner="white",
            eval_cp=10000,
            expected_tone="final",
            forbidden_terms=("next time", "close game", "good try"),
            user_task="I just played Nf6. You reply with Qxf7# — checkmate. In your persona's voice, say one natural-sounding line.",
        ),
        Scenario(
            name="rook_white_gives_check",
            description="Rook (white) runs the Italian attack ending with Qh5+ after Bxf7+.",
            user_plays="black",
            san_history=[
                "e4",
                "e5",
                "Nf3",
                "Nc6",
                "Bc4",
                "Bc5",
                "d3",
                "Nf6",
                "Ng5",
                "O-O",
                "h3",
                "h6",
                "Nxf7",
                "Rxf7",
                "Bxf7+",
                "Kxf7",
                "Qh5+",
            ],
            eval_cp=30,
            classification=MoveClassification.BEST,
            expected_tone="confident",
            forbidden_terms=("pin", "skewer"),
            user_task="I just played Kxf7. You reply with Qh5+. In your persona's voice, say one natural-sounding line.",
        ),
        Scenario(
            name="rook_white_castles_kingside",
            description="Rook (white) castles kingside in a quiet Italian Game.",
            user_plays="black",
            san_history=["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6", "O-O"],
            eval_cp=15,
            classification=MoveClassification.BEST,
            expected_tone="neutral",
            forbidden_terms=("pin", "fork", "skewer"),
            user_task="You just played O-O (castled kingside). In your persona's voice, say one natural-sounding line.",
        ),
        Scenario(
            name="rook_white_blunders_queen",
            description="Rook (white) blundered the queen on h7; user's rook captured it. Rook is losing.",
            user_plays="black",
            san_history=["d4", "d5", "Qd3", "Nf6", "Qh3", "Nc6", "Qxh7", "Rxh7"],
            eval_cp=-800,
            classification=MoveClassification.BLUNDER,
            expected_tone="rattled",
            forbidden_terms=("winning", "my advantage", "initiative"),
            user_task="I just played Rxh7, capturing your queen. In your persona's voice, say one natural-sounding line — you just lost your queen.",
        ),
    ]


# ---------------------------------------------------------------------------
# Grading.
# ---------------------------------------------------------------------------


# Persona prompts — mirror the shape in the real app. Kept terse so
# the eval harness isn't coupled to every persona tweak in the main
# codebase.
_PERSONA_PROMPTS = {
    "casual": (
        "You are a casual club player — warm, curious, light-hearted. "
        "Talk the way a friend would across a coffee-shop table. Keep it "
        "short and natural."
    ),
    "grandmaster": (
        "You are a grandmaster — precise, stoic, confident. Never gush. "
        "Keep remarks concise and reserved. A nod of approval is plenty."
    ),
    "trash_talker": (
        "You are a trash-talking coach — sharp, cocky, playful. You poke "
        "fun at slips and crow when you take pieces, but always in good "
        "humour, never mean."
    ),
}


def build_messages(scenario: Scenario, directive: str) -> list[dict]:
    """Replicate the message shape the agent loop hands to the LLM.

    Notes on the merged-system shape: the real pipeline's
    :class:`RichChessAnalyticsHook` injects the briefing as a *second*
    system message positioned right before the last user message.
    That shape breaks on chat templates that enforce a single
    leading system (Qwen3.5, some llama-cpp chat formats). To run a
    fair cross-model benchmark we collapse the persona system and
    the briefing into one system message here — equivalent from the
    model's POV, compatible with every template we've tested.
    Adjust the real pipeline to match if / when we ship a model that
    requires it.
    """
    system = _ROOK_TOOL_GUIDANCE + "\n\n---\n\n" + _PERSONA_PROMPTS.get(scenario.persona, _PERSONA_PROMPTS["casual"])
    briefing = f"[CHESS BRIEFING — internal context, do not read aloud verbatim]\n{directive}\n[END BRIEFING]"
    return [
        {"role": "system", "content": f"{system}\n\n---\n\n{briefing}"},
        {"role": "user", "content": scenario.user_task},
    ]


@dataclass
class GradingResult:
    scenario: str
    directive: str
    reply: str
    flags: list[str]
    score: int  # 0-100


def grade(scenario: Scenario, directive: str, reply: str) -> GradingResult:
    """Score a reply against the scenario.

    Heuristic, not perfect — but catches the common failure modes
    (fabricated tactics, SAN restatement, wrong tone, verbatim-bullet
    paste, overlong replies).
    """
    flags: list[str] = []
    low = reply.lower()
    directive_low = directive.lower()

    # 1. Forbidden words — tactics the directive didn't declare.
    for term in scenario.forbidden_terms:
        if term.lower() in low:
            flags.append(f"used forbidden term '{term}'")

    # 2. Length check — short replies only (one sentence).
    words = reply.split()
    if len(words) > 40:
        flags.append(f"too long ({len(words)} words)")

    # 3. SAN restatement — shouldn't quote SAN verbatim unless it
    # comes via natural language wrapping. Rough heuristic: if the
    # reply starts with a SAN token.
    if words and _looks_like_san(words[0].rstrip(",.")):
        flags.append(f"reply starts with bare SAN: {words[0]!r}")

    # 4. Bullet-paste — did the model literally copy a directive line?
    for line in directive.split("\n"):
        line_str = line.strip("- ").strip()
        if len(line_str) > 30 and line_str.lower() in low:
            flags.append(f"pasted directive bullet verbatim: {line_str[:60]!r}")
            break

    # 5. Tone mismatch — does the reply align with expected_tone?
    tone_flag = _tone_mismatch(reply, scenario.expected_tone, directive_low)
    if tone_flag:
        flags.append(tone_flag)

    # 6. Pronoun slip — if the scenario has Rook playing a specific
    # side, does the reply refer to pieces the wrong way? Weak check:
    # flag if "your queen" appears but the user doesn't have a queen
    # (after the Qxf6-gxf6 scenario the user has two queens — skip).
    # Harder to auto-detect without a board; leave for now.

    # 7. Empty reply (silence).
    if not reply.strip() or reply.strip().lower() in ("<silent>", "(silent)"):
        flags.append("emitted <silent> sentinel — no chat bubble produced")

    # 8. <think> leakage — ``_extract_text`` already strips closed and
    # truncated ``<think>...</think>`` blocks before grading (mirrors
    # the pipeline's ``ThinkTagStripHook``). If a residual tag survives
    # that strip, something's malformed — flag it.
    if "<think>" in low or "</think>" in low:
        flags.append("reply has unclosed <think> markers after strip")

    # Score = 100 - 12 * len(flags), clamped to [0, 100].
    score = max(0, 100 - 12 * len(flags))
    return GradingResult(
        scenario=scenario.name,
        directive=directive,
        reply=reply.strip(),
        flags=flags,
        score=score,
    )


def _looks_like_san(token: str) -> bool:
    """Detect bare SAN tokens (e.g. 'Nxd5', 'e4', 'Qxf7+')."""
    if not token or len(token) < 2 or len(token) > 7:
        return False
    if token in ("O-O", "O-O-O"):
        return True
    last_letter = token[0]
    return last_letter.isupper() and any(c.isdigit() for c in token)


def _tone_mismatch(reply: str, expected_tone: str, directive_low: str) -> str | None:
    """Return a flag string if the tone looks wrong for the situation.

    Very coarse — we look for specific positive/negative words and
    cross-check against what the directive says the situation is.
    """
    low = reply.lower()
    positive = (
        "winning",
        "great",
        "excellent",
        "fantastic",
        "perfect",
        "brilliant",
        "dominant",
        "crushing",
        "i'm ahead",
        "i've got this",
    )
    negative = (
        "losing",
        "trouble",
        "ouch",
        "that stings",
        "rattled",
        "careful now",
        "you got me",
    )
    if expected_tone == "rattled":
        # Rook is losing — reply should not sound like Rook is winning.
        if any(p in low for p in positive) and "winning" in directive_low:
            return "tone mismatch: reply sounds upbeat while Rook is losing"
        if "congrat" in low:
            return "tone mismatch: congratulating the user when Rook is losing (fine in persona but suspicious — ensure it's self-deprecation not praise)"
    elif expected_tone == "confident":
        # Rook is winning — shouldn't sound panicked.
        if any(n in low for n in negative):
            return "tone mismatch: reply sounds rattled while Rook is winning"
    elif expected_tone == "final":
        # Game over — reply should feel final, not open-ended.
        if "next move" in low or "your move" in low or "what will you do" in low:
            return "tone mismatch: implies game continues but it's over"
    return None


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-bound commentary evaluator.")
    parser.add_argument(
        "--model",
        default="llama-3.2-1b",
        help="Preset slug or HF shorthand for the model to evaluate.",
    )
    parser.add_argument(
        "--personas",
        default="casual,grandmaster,trash_talker",
        help="Comma-separated personas to test each scenario under.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature — lower is more deterministic.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="Max tokens per reply.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=8192,
        help=(
            "KV-cache context size. Pass 0 to use the model's trained "
            "n_ctx (silences the `n_ctx_seq < n_ctx_train` warning but "
            "allocates a large KV cache on long-context models)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full directive + messages for each scenario.",
    )
    args = parser.parse_args()

    personas = [p.strip() for p in args.personas.split(",") if p.strip()]
    scns = scenarios()

    print(f"Loading {args.model}…")
    t0 = time.perf_counter()
    llm = LLM(model_path=args.model, n_ctx=args.n_ctx)
    print(f"  ready in {time.perf_counter() - t0:.1f}s")

    results: list[GradingResult] = []
    for persona in personas:
        print(f"\n{'=' * 78}\nPERSONA: {persona}\n{'=' * 78}")
        for scn in scns:
            scn.persona = persona
            directive = _directive_for(scn)
            if directive is None:
                print(f"\n[{scn.name}] gate would stay silent — skipping")
                continue
            messages = build_messages(scn, directive)

            if args.verbose:
                print(f"\n--- {scn.name} directive ---")
                print(directive)
                print("--- user task ---")
                print(scn.user_task)

            t0 = time.perf_counter()
            raw = llm.complete(
                messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=False,
            )
            elapsed = time.perf_counter() - t0
            reply = _extract_text(raw)

            grade_result = grade(scn, directive, reply)
            results.append(grade_result)

            flag_str = "  ⚠ " + "; ".join(grade_result.flags) if grade_result.flags else "  ✓ clean"
            print(f"\n[{scn.name}] {elapsed:.1f}s  score={grade_result.score}")
            print(f"  {scn.description}")
            print(f"  reply: {reply!r}")
            print(flag_str)

    # Summary table.
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    if not results:
        print("(no results — all scenarios gated silent)")
        return 0
    by_scenario: dict[str, list[int]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario, []).append(r.score)
    for name, scores in by_scenario.items():
        avg = sum(scores) / len(scores)
        print(f"  {name:35s}  avg={avg:5.1f}  runs={len(scores)}  {scores}")
    overall = sum(r.score for r in results) / len(results)
    print(f"\n  OVERALL AVERAGE: {overall:.1f} / 100")
    print(f"  total runs: {len(results)}")

    # Exit non-zero if overall is poor — useful for CI gating later.
    return 0 if overall >= 50 else 1


def _directive_for(scenario: Scenario) -> str | None:
    """Build the directive the :class:`CommentaryGateHook` would stash
    for this scenario."""
    state = _FakeState(
        san_history=list(scenario.san_history),
        last_move_san=scenario.san_history[-1] if scenario.san_history else None,
        last_move_classification=scenario.classification,
        eval_cp=scenario.eval_cp,
        is_game_over=scenario.is_game_over,
        game_over_reason=scenario.game_over_reason,
        winner=scenario.winner,
    )
    env = _FakeEnv(user_plays=scenario.user_plays)
    env.set_state(state)
    session = Session()
    session.state["greeted"] = scenario.greeted_before
    # Populate history so the MOVE HISTORY block renders on speakable turns.
    ctx = AgentContext(deps=env, session=session)
    _record_turn_history(state, env, ctx.session.state)
    return _build_ground_truth(state, env, ctx.session.state)


def _extract_text(raw: Any) -> str:
    """Pull the first message's text from a llama-cpp chat completion
    AND apply the same ``<think>...</think>`` stripping the real
    RookApp pipeline does at ``AFTER_LLM`` via
    :class:`ThinkTagStripHook`.

    Without this, Qwen3-family models are penalised by the grader for
    emitting structural ``<think>\\n\\n</think>`` wrappers even when
    the reasoning content inside is empty (the ``/no_think`` soft
    switch takes effect but doesn't remove the tag markers). Mirroring
    the pipeline's strip gives a fair score — the user never sees
    those tags.
    """
    try:
        text = raw["choices"][0]["message"]["content"] or ""
    except Exception:
        text = str(raw)
    # Closed ``<think>...</think>`` blocks — including empty ones.
    text = _THINK_BLOCK_RE.sub("", text)
    # Truncated ``<think>...`` to EOF (model ran out of tokens
    # before closing the tag).
    text = _THINK_TRUNCATED_RE.sub("", text)
    return text.strip()


# Mirror :mod:`edgevox.examples.agents.chess_robot.sanitize`. Duplicated
# rather than imported because importing the hook module pulls in
# ``edgevox.agents.hooks`` (fine) and we want the eval script to
# stay a single-file tool when run from other checkouts.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_TRUNCATED_RE = re.compile(r"<think>.*$", re.DOTALL)


if __name__ == "__main__":
    sys.exit(main())
