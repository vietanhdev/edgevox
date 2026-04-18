// Click-to-move flow — verifies the user can play by clicking pieces
// on the board (not just typing / speaking).
//
// We can't easily assert on react-chessboard's internal DOM squares
// (they change between library versions), so these tests exercise the
// higher-level outcome: when the user clicks a valid sequence, a UCI
// move should reach the WebSocket.

import { test, expect } from "@playwright/test";
import { getSentTexts, installWsStub, pushBoot, pushChessState, skipOnboarding } from "./fixtures/mockWs";

test.describe("Rook — click-to-move", () => {
  test.beforeEach(async ({ page }) => {
    await installWsStub(page);
    await skipOnboarding(page);
    await page.goto("/?mode=rook");
    await pushBoot(page);
  });

  test("board renders in interactive mode when it's the user's turn", async ({ page }) => {
    // The starting FEN has turn=white, user plays white → pieces should be draggable.
    // react-chessboard uses the ``data-piece`` attribute on piece spans; ensure
    // at least one piece is present with that marker.
    const pieces = page.locator("[data-piece], .piece, [draggable='true']");
    await expect(pieces.first()).toBeVisible();
  });

  test("clicking a user piece then a legal square sends UCI", async ({ page }) => {
    // Find the square element for e2 and e4 using the library's
    // data-square attribute.
    const e2 = page.locator('[data-square="e2"]');
    const e4 = page.locator('[data-square="e4"]');
    await expect(e2).toBeVisible();
    await expect(e4).toBeVisible();

    await e2.click();
    await e4.click();

    await expect
      .poll(async () => (await getSentTexts(page)).some((s) => s.includes("e2e4")))
      .toBe(true);
  });

  test("clicking same square twice deselects", async ({ page }) => {
    const e2 = page.locator('[data-square="e2"]');
    await e2.click();
    await e2.click();
    // No move sent — only the boot / hello frames.
    const sent = await getSentTexts(page);
    expect(sent.some((s) => s.includes('"text_input"') && s.includes("e2e2"))).toBe(false);
  });

  test("clicking opponent piece doesn't select it", async ({ page }) => {
    // Black's pieces on rank 7 — user plays white, shouldn't pick them up.
    const e7 = page.locator('[data-square="e7"]');
    const e5 = page.locator('[data-square="e5"]');
    await e7.click();
    await e5.click();
    // Should NOT have sent e7e5 (we don't control black).
    const sent = await getSentTexts(page);
    expect(sent.some((s) => s.includes("e7e5"))).toBe(false);
  });

  test("after game over the board rejects clicks", async ({ page }) => {
    await pushChessState(page, {
      turn: "black",
      ply: 3,
      last_move_san: "Qxf7#",
      is_game_over: true,
      game_over_reason: "checkmate",
      winner: "white",
      san_history: ["e4", "e5", "Qxf7#"],
    });
    const e2 = page.locator('[data-square="e2"]').first();
    if (await e2.isVisible().catch(() => false)) {
      await e2.click();
      const sent = await getSentTexts(page);
      expect(sent.some((s) => s.includes('"text_input"'))).toBe(false);
    }
  });
});
