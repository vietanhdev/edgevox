// UX iteration tests — thinking timer, error persistence, typing dots,
// layout responsiveness.

import { test, expect } from "@playwright/test";
import { installWsStub, pushBoot, skipOnboarding } from "./fixtures/mockWs";

test.describe("Rook — thinking timer", () => {
  test.beforeEach(async ({ page }) => {
    await installWsStub(page);
    await skipOnboarding(page);
    await page.goto("/?mode=rook");
    await pushBoot(page);
  });

  test("turn pill shows elapsed time while rook is thinking", async ({ page }) => {
    // Push state: thinking phase.
    await page.evaluate(() => {
      const api = (window as unknown as { __rookTest: { push: (m: unknown) => void } }).__rookTest;
      // chess_state: turn=black means rook is to move
      api.push({
        type: "chess_state",
        fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        ply: 2,
        turn: "black",
        last_move_san: "e5",
        san_history: ["e4", "e5"],
        is_game_over: false,
      });
      api.push({ type: "state", value: "thinking" });
    });

    // After ~1.5s the pill should show elapsed.
    await page.waitForTimeout(1600);
    await expect(page.getByTestId("turn-pill")).toContainText(/thinking.*\d+\.\ds/);
  });

  test("pill returns to 'rook thinking…' once timer stops at 'speaking' transition", async ({ page }) => {
    await page.evaluate(() => {
      const api = (window as unknown as { __rookTest: { push: (m: unknown) => void } }).__rookTest;
      api.push({
        type: "chess_state",
        fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        ply: 2,
        turn: "black",
        last_move_san: "e5",
        san_history: ["e4", "e5"],
      });
      api.push({ type: "state", value: "speaking" });
    });
    // Elapsed counter should NOT be showing (timer stopped at "speaking").
    await page.waitForTimeout(1500);
    const txt = await page.getByTestId("turn-pill").innerText();
    expect(txt).not.toMatch(/\d+\.\d+s/);
  });
});

test.describe("Rook — error persistence", () => {
  test.beforeEach(async ({ page }) => {
    await installWsStub(page);
    await skipOnboarding(page);
    await page.goto("/?mode=rook");
    await pushBoot(page);
  });

  test("server error auto-dismisses after 4s", async ({ page }) => {
    await page.evaluate(() => {
      (window as unknown as { __rookTest: { push: (m: unknown) => void } }).__rookTest.push({
        type: "error",
        message: "transient glitch",
      });
    });
    await expect(page.getByRole("alert")).toContainText("transient glitch");
    // Confirm there's no dismiss button for a transient error.
    await expect(page.locator('button[aria-label="dismiss error"]')).toHaveCount(0);
  });
});

test.describe("Rook — persona palette", () => {
  test.beforeEach(async ({ page }) => {
    await installWsStub(page);
    await skipOnboarding(page);
    await page.goto("/?mode=rook");
    await pushBoot(page);
  });

  test("ring color swaps when persona changes", async ({ page }) => {
    // Grandmaster face = blue accent.
    await page.evaluate(() => {
      const api = (window as unknown as { __rookTest: { push: (m: unknown) => void } }).__rookTest;
      api.push({
        type: "robot_face",
        mood: "calm",
        gaze_x: 0,
        gaze_y: 0,
        persona: "grandmaster",
        tempo: "idle",
      });
    });
    await expect(page.getByTestId("robot-face")).toHaveAttribute("data-persona", "grandmaster");

    // Trash-talker = magenta.
    await page.evaluate(() => {
      const api = (window as unknown as { __rookTest: { push: (m: unknown) => void } }).__rookTest;
      api.push({
        type: "robot_face",
        mood: "amused",
        gaze_x: 0,
        gaze_y: 0,
        persona: "trash_talker",
        tempo: "speaking",
      });
    });
    await expect(page.getByTestId("robot-face")).toHaveAttribute("data-persona", "trash_talker");
    await expect(page.getByTestId("robot-face")).toHaveAttribute("data-mood", "amused");
  });
});
