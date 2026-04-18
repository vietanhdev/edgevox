"""Terminal rendering for :class:`ChessState` — used by text-mode example + TUI.

Exposes two surfaces:

- :func:`render_board_rich` — builds a ``rich.panel.Panel`` you can
  ``Console.print`` or stash in a Textual ``Static`` via ``update()``.
  Pure text, unicode pieces, last-move squares highlighted.
- :class:`ChessPanel` — a thin Textual widget that re-renders on every
  :class:`ChessState` update. Subscribe with
  ``env.subscribe(panel.post_state)`` after mounting.

Both share the same renderer so the board looks identical between
plain-console output and a future full TUI integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chess
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from edgevox.integrations.chess.environment import ChessState


_UNICODE_PIECES = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",
    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


def _highlight_squares(last_move_uci: str | None) -> set[str]:
    """Return the 'from' + 'to' squares as algebraic names.

    Returns an empty set if there's no last move; the renderer just
    skips the highlight in that case.
    """
    if not last_move_uci or len(last_move_uci) < 4:
        return set()
    return {last_move_uci[0:2], last_move_uci[2:4]}


def render_board_rich(state: ChessState) -> Panel:
    """Render a :class:`ChessState` as a rich :class:`Panel`.

    Layout mirrors what lichess shows: white ranks at the bottom for
    white-to-move, flipped when black is to move AND the user plays
    black. We intentionally don't hide anything — the full board is
    visible regardless of side. Last-move squares get a yellow
    background so it's easy to see what just happened.
    """
    board = chess.Board(state.fen)
    highlight = _highlight_squares(state.last_move_uci)

    text = Text()
    # Ranks 8→1 top-to-bottom (standard POV).
    for rank_idx in range(7, -1, -1):
        text.append(f" {rank_idx + 1} ", style="dim")
        for file_idx in range(8):
            sq = chess.square(file_idx, rank_idx)
            sq_name = chess.square_name(sq)
            piece = board.piece_at(sq)
            glyph = _UNICODE_PIECES.get(piece.symbol(), " ") if piece else "·"
            style = _square_style(file_idx, rank_idx, sq_name in highlight)
            text.append(f" {glyph} ", style=style)
        text.append("\n")
    text.append("    a  b  c  d  e  f  g  h\n", style="dim")

    # Status line under the board.
    title = f"{state.turn.capitalize()} to move · ply {state.ply}"
    if state.is_game_over:
        title = f"GAME OVER — {state.game_over_reason} · winner: {state.winner or 'draw'}"
    subtitle_parts: list[str] = []
    if state.last_move_san:
        classification = state.last_move_classification.value if state.last_move_classification else "?"
        subtitle_parts.append(f"last: {state.last_move_san} ({classification})")
    if state.eval_cp is not None:
        subtitle_parts.append(f"eval: {state.eval_cp:+d} cp")
    elif state.mate_in is not None:
        subtitle_parts.append(f"mate in {state.mate_in}")
    if state.opening:
        subtitle_parts.append(state.opening)
    subtitle = " · ".join(subtitle_parts) if subtitle_parts else None

    return Panel(text, title=title, subtitle=subtitle, border_style="cyan")


def _square_style(file_idx: int, rank_idx: int, highlighted: bool) -> str:
    """Pick a rich style for one board square.

    Dark / light squares + a yellow overlay for the last-move
    highlight. Keeping this out of the hot render loop below so a
    future colour-scheme knob has one place to edit.
    """
    dark = (file_idx + rank_idx) % 2 == 0
    if highlighted:
        return "black on yellow" if dark else "black on bright_yellow"
    return "white on grey30" if dark else "black on grey70"


# ---------------------------------------------------------------------------
# Textual widget — thin wrapper that exposes an update() the env can call
# ---------------------------------------------------------------------------


try:
    from textual.widgets import Static

    class ChessPanel(Static):
        """Textual widget rendering the live board.

        Usage::

            panel = ChessPanel()
            env.subscribe(panel.post_state)

        ``post_state`` is thread-safe: it calls Textual's ``call_from_thread``
        so the env's listener thread doesn't touch the UI loop directly.
        """

        DEFAULT_CSS = "ChessPanel { height: auto; min-width: 38; }"

        def post_state(self, state: ChessState) -> None:
            """Schedule a re-render from any thread.

            ``call_from_thread`` refuses to run on Textual's own main
            thread — so when the listener fires from the UI thread
            (e.g. right after a user-initiated move inside the app) we
            update synchronously; only cross-thread calls go through
            ``call_from_thread``.
            """
            import threading

            app = getattr(self, "app", None)
            if app is None or not app._thread_id:
                # Widget hasn't mounted yet — store for the first mount.
                self._pending_state = state
                return
            if threading.get_ident() == app._thread_id:
                self._apply_state(state)
            else:
                app.call_from_thread(self._apply_state, state)

        def _apply_state(self, state: ChessState) -> None:
            self.update(render_board_rich(state))

        def on_mount(self) -> None:
            pending = getattr(self, "_pending_state", None)
            if pending is not None:
                self._apply_state(pending)

except ImportError:  # pragma: no cover — textual always installed in EdgeVox
    ChessPanel = None  # type: ignore[assignment,misc]


__all__ = ["ChessPanel", "render_board_rich"]
