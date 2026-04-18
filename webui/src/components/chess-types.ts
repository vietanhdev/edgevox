// Shared chess message type — mirrors the `chess_state` variant in
// ws-client.ts so chess components don't depend on the giant
// ``ServerMessage`` union directly.

export interface ChessStateMessage {
  type: "chess_state";
  fen: string;
  ply: number;
  turn: "white" | "black";
  last_move_uci?: string | null;
  last_move_san?: string | null;
  last_move_classification?: "best" | "good" | "inaccuracy" | "mistake" | "blunder" | null;
  san_history?: string[];
  eval_cp?: number | null;
  mate_in?: number | null;
  win_prob_white?: number;
  opening?: string | null;
  is_game_over?: boolean;
  game_over_reason?: string | null;
  winner?: "white" | "black" | null;
}
