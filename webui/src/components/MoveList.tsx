// Scrolling SAN move list with classification badges for the last move.
//
// Only the *most recent* move carries a classification badge — earlier
// moves keep their SAN but drop the colour. That matches what Lichess
// does while the game is live and avoids forcing the server to send a
// per-move classification history.

import type { ChessStateMessage } from "./chess-types";

interface MoveListProps {
  state: ChessStateMessage;
  maxHeight?: number;
}

const BADGE_COLORS: Record<string, string> = {
  best: "text-neon-green",
  good: "text-neon-green",
  inaccuracy: "text-neon-orange",
  mistake: "text-neon-red",
  blunder: "text-neon-red font-bold",
};

function pairMoves(sans: string[] | undefined): Array<[number, string, string?]> {
  if (!sans) return [];
  const rows: Array<[number, string, string?]> = [];
  for (let i = 0; i < sans.length; i += 2) {
    rows.push([i / 2 + 1, sans[i], sans[i + 1]]);
  }
  return rows;
}

export function MoveList({ state, maxHeight = 360 }: MoveListProps) {
  const rows = pairMoves(state.san_history);
  const lastIndex = (state.san_history?.length ?? 0) - 1;
  const lastClass = state.last_move_classification ?? null;

  return (
    <div
      className="font-mono text-xs bg-[#0d1117] border border-[#1e3a2e] rounded p-2 overflow-y-auto"
      style={{ maxHeight }}
    >
      <div className="text-neon-green font-bold mb-1">■ Moves</div>
      {rows.length === 0 ? (
        <div className="text-muted-foreground">no moves yet</div>
      ) : (
        <table className="w-full text-left">
          <tbody>
            {rows.map(([ply, whiteSan, blackSan]) => {
              const whiteIdx = (ply - 1) * 2;
              const blackIdx = whiteIdx + 1;
              return (
                <tr key={ply}>
                  <td className="text-muted-foreground pr-2 w-6">{ply}.</td>
                  <td className={whiteIdx === lastIndex && lastClass ? BADGE_COLORS[lastClass] : "text-foreground"}>
                    {whiteSan}
                  </td>
                  <td className={blackIdx === lastIndex && lastClass ? BADGE_COLORS[lastClass] : "text-foreground"}>
                    {blackSan ?? ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {lastClass ? (
        <div className="mt-2 text-xs">
          <span className="text-muted-foreground">last: </span>
          <span className={BADGE_COLORS[lastClass]}>{lastClass}</span>
        </div>
      ) : null}
    </div>
  );
}

export default MoveList;
