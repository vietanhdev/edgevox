// Chess board rendered from a server-sent ``chess_state`` message.
//
// Wraps react-chessboard (MIT) so the drag-and-drop / animation work is
// handled for us. The board is read-only by default — the voice agent
// is the move producer; this panel reflects the state the environment
// emits. When the user finishes voice or text input, the agent calls
// play_user_move / engine_move and the next chess_state event snaps
// the board to its new position.

import { Chessboard } from "react-chessboard";
import type { ChessStateMessage } from "./chess-types";

const CLASSIFICATION_COLOR: Record<string, string> = {
  best: "rgba(72, 221, 149, 0.45)",
  good: "rgba(72, 221, 149, 0.25)",
  inaccuracy: "rgba(250, 204, 21, 0.35)",
  mistake: "rgba(248, 113, 113, 0.35)",
  blunder: "rgba(248, 113, 113, 0.55)",
};

interface ChessBoardProps {
  state: ChessStateMessage;
  boardWidth?: number;
  orientation?: "white" | "black";
}

// Convert a UCI move like "e2e4" into the two squares the last move
// touched. Returns null when there's no last move so callers can skip
// the highlight overlay.
function lastMoveSquares(uci: string | null | undefined): [string, string] | null {
  if (!uci || uci.length < 4) return null;
  return [uci.slice(0, 2), uci.slice(2, 4)];
}

export function ChessBoardPanel({ state, boardWidth = 360, orientation = "white" }: ChessBoardProps) {
  const last = lastMoveSquares(state.last_move_uci);
  const classColor =
    (state.last_move_classification && CLASSIFICATION_COLOR[state.last_move_classification]) ||
    "rgba(56, 189, 248, 0.35)";

  const customSquareStyles: Record<string, React.CSSProperties> = {};
  if (last) {
    customSquareStyles[last[0]] = { background: classColor };
    customSquareStyles[last[1]] = { background: classColor };
  }

  return (
    <div className="flex flex-col items-center gap-1">
      <Chessboard
        position={state.fen}
        boardWidth={boardWidth}
        boardOrientation={orientation}
        arePiecesDraggable={false}
        customBoardStyle={{ borderRadius: 6, boxShadow: "0 0 20px rgba(56, 189, 248, 0.15)" }}
        customSquareStyles={customSquareStyles}
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

export default ChessBoardPanel;
