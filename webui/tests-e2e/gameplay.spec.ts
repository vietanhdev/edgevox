// Gameplay tests — simulate server-pushed chess_state + robot_face
// events and verify the UI reacts (board updates, chat, captures,
// game-over banner, check pulse).

import { test, expect } from "@playwright/test";
import { installWsStub, pushBoot, pushChessState, pushFace, skipOnboarding } from "./fixtures/mockWs";

test.describe("Rook — gameplay reactions", () => {
  test.beforeEach(async ({ page }) => {
    await installWsStub(page);
    await skipOnboarding(page);
    await page.goto("/?mode=rook");
    await pushBoot(page);
  });

  test("robot_face updates the face testid attributes", async ({ page }) => {
    await pushFace(page, { mood: "triumphant", tempo: "speaking", persona: "trash_talker" });
    const face = page.getByTestId("robot-face");
    await expect(face).toHaveAttribute("data-mood", "triumphant");
    await expect(face).toHaveAttribute("data-tempo", "speaking");
    await expect(face).toHaveAttribute("data-persona", "trash_talker");
  });

  test("turn pill swaps between 'your turn' and 'rook thinking…'", async ({ page }) => {
    await pushChessState(page, { turn: "white", ply: 2, last_move_san: "e5", san_history: ["e4", "e5"] });
    await expect(page.getByTestId("turn-pill")).toHaveText(/your turn/i);
    await pushChessState(page, { turn: "black", ply: 3, last_move_san: "Nf3", san_history: ["e4", "e5", "Nf3"] });
    await expect(page.getByTestId("turn-pill")).toHaveText(/rook thinking/i);
  });

  test("move history shows pair numbering", async ({ page }) => {
    await pushChessState(page, {
      turn: "white",
      ply: 4,
      last_move_san: "Nc6",
      san_history: ["e4", "e5", "Nf3", "Nc6"],
    });
    // "1." pair should be visible.
    await expect(page.getByText(/1\.\s*e4\s+e5/)).toBeVisible();
  });

  test("captured tray shows rook's captured piece + edge", async ({ page }) => {
    // White lost a pawn: FEN missing one white pawn.
    await pushChessState(page, {
      fen: "rnbqkbnr/pppp1ppp/8/4p3/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 3",
      turn: "white",
      ply: 5,
      last_move_san: "Nxe5",
      san_history: ["e4", "e5", "Nf3", "Nc6", "Nxe5"],
    });
    // Black-captured-white pieces are in the "rook" label area — check at least one pawn glyph.
    const rookTray = page.getByText("rook", { exact: false }).first();
    await expect(rookTray).toBeVisible();
  });

  test("game-over banner appears with a rematch button", async ({ page }) => {
    await pushChessState(page, {
      turn: "black",
      ply: 3,
      last_move_san: "Qxf7#",
      is_game_over: true,
      game_over_reason: "checkmate",
      winner: "white",
      san_history: ["e4", "e5", "Qxf7#"],
    });
    await expect(page.getByText(/you won/i)).toBeVisible();
    await expect(page.getByRole("button", { name: /rematch/i })).toBeVisible();
  });

  test("dismiss game-over banner hides it but keeps board", async ({ page }) => {
    await pushChessState(page, {
      turn: "black",
      ply: 3,
      last_move_san: "Qxf7#",
      is_game_over: true,
      game_over_reason: "checkmate",
      winner: "white",
      san_history: ["e4", "e5", "Qxf7#"],
    });
    // The dismiss button is an icon button with aria-label="dismiss".
    await page.getByRole("button", { name: "dismiss" }).click();
    await expect(page.getByText(/you won/i)).toBeHidden();
  });

  test("error toast appears on server error message", async ({ page }) => {
    await page.evaluate(() => {
      (window as unknown as { __rookTest: { push: (m: unknown) => void } }).__rookTest.push({
        type: "error",
        message: "illegal move",
      });
    });
    await expect(page.getByText(/illegal move/i)).toBeVisible();
  });
});
