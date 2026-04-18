// Derive captured pieces + material from a FEN string.
//
// Runs on the client so we don't need to expand the server protocol.
// Iterates once over the piece-placement field of the FEN, counts what
// survives, and subtracts from the starting army. No python-chess,
// no regex, just a quick byte walk.
//
// Piece values follow the traditional Reinfeld heuristic: P=1, N/B=3,
// R=5, Q=9, K=0 (never captured). The material diff powers the
// "you're +3" indicator; signed positive = user ahead.

const START_ARMY: Record<string, number> = {
  P: 8, N: 2, B: 2, R: 2, Q: 1, K: 1,
  p: 8, n: 2, b: 2, r: 2, q: 1, k: 1,
};

const PIECE_VALUE: Record<string, number> = {
  P: 1, N: 3, B: 3, R: 5, Q: 9, K: 0,
  p: 1, n: 3, b: 3, r: 5, q: 9, k: 0,
};

export interface MaterialBreakdown {
  /** Captured white pieces (stored as uppercase letters). */
  capturedByBlack: string[];
  /** Captured black pieces (stored as lowercase letters). */
  capturedByWhite: string[];
  /** Signed diff from WHITE's perspective: +N = white ahead by N points. */
  whiteMaterialEdge: number;
}

export function deriveMaterial(fen: string): MaterialBreakdown {
  const placement = fen.split(" ")[0] ?? "";
  const counts: Record<string, number> = {};
  for (const ch of placement) {
    if (ch in START_ARMY) counts[ch] = (counts[ch] ?? 0) + 1;
  }
  const capturedByBlack: string[] = [];
  const capturedByWhite: string[] = [];
  let whiteMaterial = 0;
  let blackMaterial = 0;
  for (const piece of Object.keys(START_ARMY)) {
    const start = START_ARMY[piece];
    const alive = counts[piece] ?? 0;
    const captured = Math.max(0, start - alive);
    for (let i = 0; i < captured; i++) {
      // Uppercase = white piece captured (went to black's tray); lowercase inverse.
      if (piece === piece.toUpperCase()) capturedByBlack.push(piece);
      else capturedByWhite.push(piece);
    }
    if (piece === piece.toUpperCase()) whiteMaterial += alive * PIECE_VALUE[piece];
    else blackMaterial += alive * PIECE_VALUE[piece];
  }
  return {
    capturedByBlack,
    capturedByWhite,
    whiteMaterialEdge: whiteMaterial - blackMaterial,
  };
}

// Unicode pieces for the tray — renders on every platform's default font.
export const PIECE_GLYPH: Record<string, string> = {
  K: "\u2654", Q: "\u2655", R: "\u2656", B: "\u2657", N: "\u2658", P: "\u2659",
  k: "\u265A", q: "\u265B", r: "\u265C", b: "\u265D", n: "\u265E", p: "\u265F",
};

// Quick test: starting FEN (no captures)
export function isStartingPosition(fen: string): boolean {
  return fen.startsWith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR");
}

// Parse out whether the side-to-move is in check. FEN doesn't flag this
// directly, but SAN move suffixes ("+") do. We infer from the last_move_san
// rather than replaying the position client-side.
export function isInCheckFromSan(lastSan: string | null | undefined): boolean {
  if (!lastSan) return false;
  return lastSan.endsWith("+") || lastSan.endsWith("#");
}

export function isCheckmateFromSan(lastSan: string | null | undefined): boolean {
  if (!lastSan) return false;
  return lastSan.endsWith("#");
}
