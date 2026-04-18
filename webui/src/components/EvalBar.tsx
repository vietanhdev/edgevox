// Vertical evaluation bar — white from the bottom, black from the top.
//
// Renders a monotonic mapping from centipawn eval to bar fill, capped
// to keep the bar usable once one side is clearly winning. The
// conversion to percent uses the same logistic as
// ``integrations.chess.analytics.win_probability`` so the bar agrees
// with what the commentary is saying.

import type { ChessStateMessage } from "./chess-types";

interface EvalBarProps {
  state: ChessStateMessage;
  height?: number;
}

// Identical math to the Python-side ``win_probability`` so the bar
// matches what the agent's tool commentary reports.
function winProbability(evalCp: number | null | undefined, mateIn: number | null | undefined): number {
  if (mateIn != null) return mateIn > 0 ? 1 : 0;
  if (evalCp == null) return 0.5;
  return 1 / (1 + Math.exp(-0.004 * evalCp));
}

function formatEval(state: ChessStateMessage): string {
  if (state.mate_in != null) return `M${Math.abs(state.mate_in)}`;
  if (state.eval_cp == null) return "—";
  const pawns = state.eval_cp / 100;
  const sign = pawns > 0 ? "+" : "";
  return `${sign}${pawns.toFixed(1)}`;
}

export function EvalBar({ state, height = 360 }: EvalBarProps) {
  const p = state.win_prob_white ?? winProbability(state.eval_cp, state.mate_in);
  const whitePct = Math.round(p * 100);

  return (
    <div className="flex flex-col items-center gap-1" style={{ height }}>
      <div className="text-xs font-mono text-neon-cyan">{formatEval(state)}</div>
      <div className="relative flex flex-col w-4 flex-1 rounded overflow-hidden border border-[#1e3a2e] bg-black">
        <div className="w-full bg-[#1a1a1a]" style={{ height: `${100 - whitePct}%` }} />
        <div className="w-full bg-[#f5f5f5]" style={{ height: `${whitePct}%` }} />
        <div
          className="absolute left-0 right-0 h-px bg-neon-red"
          style={{ top: `${100 - whitePct}%` }}
        />
      </div>
    </div>
  );
}

export default EvalBar;
