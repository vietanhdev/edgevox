// Chess board rendered from a server-sent ``chess_state`` message.
//
// Wraps react-chessboard (MIT) so the drag-and-drop / animation work is
// handled for us. The board becomes interactive when the parent passes
// an ``onMove`` callback — drag-to-move and click-to-move both fire
// it with a UCI string ("e2e4" or "e7e8q" for promotions). The parent
// is responsible for forwarding the UCI to the server; this component
// never mutates state on its own.

import { useCallback, useMemo, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import type { ChessStateMessage } from "./chess-types";

const CLASSIFICATION_COLOR: Record<string, string> = {
  best: "rgba(72, 221, 149, 0.45)",
  good: "rgba(72, 221, 149, 0.25)",
  inaccuracy: "rgba(250, 204, 21, 0.35)",
  mistake: "rgba(248, 113, 113, 0.35)",
  blunder: "rgba(248, 113, 113, 0.55)",
};

// Highlight applied to a piece the user just clicked (selection state
// for click-to-move). Keeping it a soft yellow so it reads as "picked up"
// not "danger".
const SELECTED_BG = "rgba(255, 215, 0, 0.45)";
const LEGAL_MOVE_DOT = "radial-gradient(circle, rgba(52, 211, 153, 0.55) 18%, transparent 20%)";

interface ChessBoardProps {
  state: ChessStateMessage;
  boardWidth?: number;
  orientation?: "white" | "black";
  /**
   * Called when the user attempts a move by dragging or clicking.
   * ``uci`` is "e2e4" (or "e7e8q" for promotions). The parent should
   * forward to the server — we don't animate the move locally;
   * instead we wait for the next ``chess_state`` event.
   *
   * Return ``false`` to reject the move (react-chessboard snaps the
   * piece back). Return ``true`` to accept it optimistically.
   */
  onMove?: (uci: string) => boolean;
  /**
   * Whose pieces the user is allowed to grab. If the server tells us
   * it's the engine's turn, we disable move input so the user doesn't
   * try to play twice.
   */
  canMove?: boolean;
}

function lastMoveSquares(uci: string | null | undefined): [string, string] | null {
  if (!uci || uci.length < 4) return null;
  return [uci.slice(0, 2), uci.slice(2, 4)];
}

export function ChessBoardPanel({
  state,
  boardWidth = 360,
  orientation = "white",
  onMove,
  canMove = true,
}: ChessBoardProps) {
  const [selected, setSelected] = useState<string | null>(null);
  const interactive = Boolean(onMove) && canMove && !state.is_game_over;

  // Compute legal moves for the selected piece via chess.js. Kept in
  // a memo keyed on the FEN + selected so it doesn't recompute on
  // every render. chess.js is forgiving — a malformed FEN yields an
  // empty set, which just means "no hints for this position" rather
  // than an error.
  const legalTargets = useMemo<Set<string>>(() => {
    if (!selected || !interactive) return new Set();
    try {
      const chess = new Chess(state.fen);
      const moves = chess.moves({ square: selected as any, verbose: true });
      return new Set(moves.map((m: any) => m.to as string));
    } catch {
      return new Set();
    }
  }, [selected, state.fen, interactive]);

  const last = lastMoveSquares(state.last_move_uci);
  const classColor =
    (state.last_move_classification && CLASSIFICATION_COLOR[state.last_move_classification]) ||
    "rgba(56, 189, 248, 0.35)";

  const customSquareStyles: Record<string, React.CSSProperties> = {};
  if (last) {
    customSquareStyles[last[0]] = { background: classColor };
    customSquareStyles[last[1]] = { background: classColor };
  }
  if (selected) {
    customSquareStyles[selected] = {
      ...(customSquareStyles[selected] ?? {}),
      background: SELECTED_BG,
    };
  }
  // Hint-dot on every legal destination for the selected piece.
  for (const sq of legalTargets) {
    // Different visual for capture vs move: captures get a ring
    // around the square, quiet moves get a small centred dot.
    const isCapture = (() => {
      try {
        const chess = new Chess(state.fen);
        const moves = chess.moves({ square: selected as any, verbose: true });
        const m = moves.find((mv: any) => mv.to === sq);
        return Boolean(m && (m.flags.includes("c") || m.flags.includes("e")));
      } catch {
        return false;
      }
    })();
    customSquareStyles[sq] = {
      ...(customSquareStyles[sq] ?? {}),
      background: isCapture
        ? "radial-gradient(circle, transparent 55%, rgba(239, 68, 68, 0.35) 57%, rgba(239, 68, 68, 0.35) 72%, transparent 74%)"
        : LEGAL_MOVE_DOT,
    };
  }

  // Always-available promotion: pieces auto-promote to queen. For voice
  // chess at small-model scale, nobody needs underpromotion in the hot
  // path — if they really want it, they'll say "promote to knight".
  const makeUci = (from: string, to: string): string => {
    const isPawn = state.fen.includes("/") && pieceAt(state.fen, from) === "P" && to[1] === "8";
    const isBlackPawn = pieceAt(state.fen, from) === "p" && to[1] === "1";
    if (isPawn || isBlackPawn) return `${from}${to}q`;
    return `${from}${to}`;
  };

  const onPieceDrop = useCallback(
    (from: string, to: string): boolean => {
      if (!interactive || !onMove) return false;
      setSelected(null);
      return onMove(makeUci(from, to));
    },
    [interactive, onMove, state.fen],
  );

  const onSquareClick = useCallback(
    (square: string) => {
      if (!interactive || !onMove) return;
      if (!selected) {
        // First click: only select if it's a piece of the side to move.
        const piece = pieceAt(state.fen, square);
        if (!piece) return;
        const isWhiteTurn = state.turn === "white";
        const pieceIsWhite = piece === piece.toUpperCase();
        if (isWhiteTurn !== pieceIsWhite) return;
        setSelected(square);
        return;
      }
      // Second click: attempt the move or re-select.
      if (square === selected) {
        setSelected(null);
        return;
      }
      const ok = onMove(makeUci(selected, square));
      if (ok) {
        setSelected(null);
      } else {
        // If the destination is another own-piece, swap selection.
        const piece = pieceAt(state.fen, square);
        const isWhiteTurn = state.turn === "white";
        if (piece && (piece === piece.toUpperCase()) === isWhiteTurn) {
          setSelected(square);
        } else {
          setSelected(null);
        }
      }
    },
    [interactive, onMove, selected, state.fen, state.turn],
  );

  return (
    <div className="flex flex-col items-center gap-1">
      <Chessboard
        position={state.fen}
        boardWidth={boardWidth}
        boardOrientation={orientation}
        arePiecesDraggable={interactive}
        customBoardStyle={{ borderRadius: 6, boxShadow: "0 0 20px rgba(56, 189, 248, 0.15)" }}
        customSquareStyles={customSquareStyles}
        onPieceDrop={onPieceDrop}
        onSquareClick={onSquareClick}
      />
      <div className="font-mono text-xs text-muted-foreground">
        {state.is_game_over ? (
          <span className="text-neon-red">
            {state.game_over_reason?.toUpperCase() || "GAME OVER"} · winner {state.winner ?? "draw"}
          </span>
        ) : (
          <span>
            <span className="text-neon-green">{state.turn}</span> to move · ply {state.ply}
            {state.opening ? <span className="ml-2 text-neon-purple">{state.opening}</span> : null}
          </span>
        )}
      </div>
    </div>
  );
}

// Derive the piece character at a FEN square ("e2", "h8"). Returns
// undefined when the square is empty or parsing fails.
function pieceAt(fen: string, square: string): string | undefined {
  const placement = fen.split(" ")[0];
  const file = square.charCodeAt(0) - "a".charCodeAt(0);
  const rank = parseInt(square[1], 10) - 1;
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return undefined;
  const rows = placement.split("/");
  if (rows.length !== 8) return undefined;
  const rowStr = rows[7 - rank];
  let col = 0;
  for (const ch of rowStr) {
    if (/[1-8]/.test(ch)) {
      col += parseInt(ch, 10);
      if (col > file) return undefined;
    } else {
      if (col === file) return ch;
      col += 1;
    }
  }
  return undefined;
}

export default ChessBoardPanel;
