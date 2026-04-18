"""MoveInterceptHook — deterministic move application at ON_RUN_START.

Small models are unreliable at emitting tool calls on every turn; they
may narrate a move in prose and never invoke ``play_user_move``, which
leaves the board frozen. For a voice chess app, the board *must* move
every time the user says one.

This hook runs at ``on_run_start`` — it looks at the raw user task for
a chess-move pattern and, when it finds one, applies it directly to
:class:`ChessEnvironment` (and fires the engine's reply) *before* the
LLM sees anything. The LLM then gets a rewritten task like "I just
played e4 and you replied Nf3. Comment briefly." The LLM only has to
talk, which small models do reliably.

The hook fails open — if it can't parse a move confidently, it
returns ``None`` and the normal tool-calling path runs. So ambiguous
utterances ("what do you think?", "knight somewhere on the queenside")
still go through the LLM as before.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from edgevox.agents.hooks import ON_RUN_START, HookResult

if TYPE_CHECKING:
    from edgevox.agents.base import AgentContext
    from edgevox.integrations.chess.environment import ChessEnvironment

log = logging.getLogger(__name__)


# UCI: e.g. ``e2e4``, ``e7e8q`` (promotion). 4-5 chars, files a-h, ranks 1-8.
_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")

# SAN: anchored at word boundaries so we don't match ``a4`` inside
# ``casual4ever`` etc. Covers pawn moves, piece moves, captures, castling,
# promotion, and ``+``/``#`` suffixes.
_SAN_RE = re.compile(
    r"\b("
    r"O-O-O|O-O|"  # castling
    r"[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|"  # piece move
    r"[a-h]x?[a-h]?[1-8](?:=[QRBN])?[+#]?"  # pawn move (incl. captures / promotions)
    r")\b"
)

# Special commands for in-game flow. These bypass LLM entirely.
_NEW_GAME_RE = re.compile(r"\b(new game|reset|restart)\b", re.IGNORECASE)
_UNDO_RE = re.compile(r"\b(undo|take back|wait,? i meant)\b", re.IGNORECASE)

# Natural-language moves: "knight to c3", "bishop takes d5", "castle kingside".
# Kept intentionally narrow — if the phrase doesn't map cleanly, we let the
# LLM handle it. Better to miss a few than to fabricate wrong moves.
_NATURAL_PIECE = {
    "king": "K",
    "queen": "Q",
    "rook": "R",
    "bishop": "B",
    "knight": "N",
}
_NATURAL_RE = re.compile(
    r"\b(king|queen|rook|bishop|knight)\s+(?:to\s+|takes\s+|x\s*)?([a-h][1-8])\b",
    re.IGNORECASE,
)
_CASTLE_SHORT_RE = re.compile(r"\bcastle(?:s)?\s+(king|kingside|short)\b", re.IGNORECASE)
_CASTLE_LONG_RE = re.compile(r"\bcastle(?:s)?\s+(queen|queenside|long)\b", re.IGNORECASE)


def extract_move(text: str) -> str | None:
    """Return a move string parseable by :meth:`ChessEnvironment.play_user_move`.

    Priority: explicit UCI → SAN → natural language. Returns ``None``
    if no confident match. The move is returned in whatever notation
    matches first; the environment accepts both UCI and SAN.
    """
    if not text:
        return None
    stripped = text.strip()

    # "play e4", "let me play Nf3" — strip common pre-move filler.
    stripped = re.sub(
        r"^(?:(?:i'?ll?|let me|i|let\s+me)\s+)?(?:play|move|go)\s+",
        "",
        stripped,
        flags=re.IGNORECASE,
    ).strip()

    # Natural-language phrases come BEFORE bare SAN because "knight to c3"
    # contains a square name (``c3``) that the SAN regex would pick up first,
    # giving us a pawn-move interpretation.
    if _CASTLE_SHORT_RE.search(stripped):
        return "O-O"
    if _CASTLE_LONG_RE.search(stripped):
        return "O-O-O"

    if m := _NATURAL_RE.search(stripped):
        piece = _NATURAL_PIECE[m.group(1).lower()]
        square = m.group(2).lower()
        # Detect capture phrasing ("takes", " x "); emit SAN with the ``x``
        # marker so the env's SAN parser resolves disambiguation right.
        captures = bool(re.search(r"\btakes\b|\bx\b", stripped, re.IGNORECASE))
        return f"{piece}{'x' if captures else ''}{square}"

    if m := _UCI_RE.search(stripped):
        return m.group(1).lower()

    if m := _SAN_RE.search(stripped):
        return m.group(1)

    return None


class MoveInterceptHook:
    """Deterministically apply moves the user speaks, before the LLM runs.

    Fires at :data:`ON_RUN_START`. Reads ``payload['task']`` for a move
    pattern. On match:

    1. Apply the user's move via :meth:`ChessEnvironment.play_user_move`.
       On failure (illegal move), pass through to the LLM with an
       ``[intercept]`` note so the model can clarify with the user.
    2. Immediately apply the engine's reply via
       :meth:`ChessEnvironment.engine_move`.
    3. Rewrite the task to a short "narrate what just happened" prompt
       so the LLM just has to talk.

    Handles ``new game`` and ``undo`` commands the same way —
    deterministic apply, then LLM narration.

    Priority 90: runs *before* :class:`BoardStateInjectionHook` (80)
    so the task injection sees the post-move state, not the pre-move
    one.
    """

    points = frozenset({ON_RUN_START})
    priority = 90

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        if not isinstance(payload, dict):
            return None
        env = _chess_env(ctx)
        if env is None:
            return None
        task = payload.get("task", "")
        if not task:
            return None

        # --- flow commands: new game, undo ---
        if _NEW_GAME_RE.search(task):
            try:
                env.new_game()
            except Exception:
                log.exception("MoveInterceptHook: new_game failed")
                return None
            new_task = "A new game just started. The board is reset. Say one short welcoming line to the user — in persona — and invite their first move."
            return HookResult.replace({**payload, "task": new_task}, reason="new game applied")

        if _UNDO_RE.search(task):
            try:
                env.undo_last_move()
            except Exception as e:
                # Nothing to undo — tell the LLM to gracefully decline.
                new_task = f"The user asked to undo, but the board could not be rolled back ({e}). Respond briefly — tell them there is no move to undo."
                return HookResult.replace({**payload, "task": new_task}, reason="undo declined")
            new_task = "The last move was just undone. Say one short line — in persona — acknowledging the take-back and asking for their new move."
            return HookResult.replace({**payload, "task": new_task}, reason="undo applied")

        # --- move intercept ---
        move = extract_move(task)
        if move is None:
            return None

        try:
            env.play_user_move(move)
        except ValueError as e:
            # Not legal (or wrong turn) — hand off to the LLM with a
            # clarifying note so it can ask the user to rephrase.
            note = f'[intercept] tried to apply "{move}" but it failed: {e}'
            new_task = f"{task}\n\n{note}\n\nThe user's move could not be applied. Respond in persona, explain briefly, and ask them to try again."
            return HookResult.replace({**payload, "task": new_task}, reason="illegal move")

        # Engine replies — best effort. If the board is terminal (user
        # delivered checkmate/stalemate), skip.
        engine_san: str | None = None
        try:
            state_after_user = env.snapshot()
            if not state_after_user.is_game_over:
                _, engine_move = env.engine_move()
                engine_san = engine_move.san
        except Exception:
            log.exception("MoveInterceptHook: engine_move failed")

        # Keep the rewrite short and open-ended. The rich briefing
        # injected at BEFORE_LLM already gives the LLM full context
        # (perspective, eval, opening, top lines, threats). A verbose
        # instruction here just makes the model write formulaic replies.
        state = env.snapshot()
        if state.is_game_over:
            winner = state.winner or "draw"
            new_task = (
                f"I just played {move}. Game's over — {state.game_over_reason}, {winner} wins. "
                "Say one short line in your persona's voice."
            )
        elif engine_san:
            new_task = (
                f"I just played {move}. You reply with {engine_san}. "
                "In your persona's voice, say one natural-sounding line about my move and yours. "
                f"Mention only {move} and {engine_san}, no other moves, no analysis essays."
            )
        else:
            new_task = f"I just played {move}. Say one natural line in your persona's voice."

        return HookResult.replace(
            {**payload, "task": new_task},
            reason=f"intercepted move {move}",
        )


def _chess_env(ctx: AgentContext) -> ChessEnvironment | None:
    """Duck-typed :class:`ChessEnvironment` lookup. Same pattern as the
    other chess hooks — check for the minimal surface we need."""
    deps = getattr(ctx, "deps", None)
    if deps is None:
        return None
    if hasattr(deps, "play_user_move") and hasattr(deps, "engine_move") and hasattr(deps, "snapshot"):
        return deps  # type: ignore[return-value]
    return None


__all__ = ["MoveInterceptHook", "extract_move"]
