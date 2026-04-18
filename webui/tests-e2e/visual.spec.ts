// Visual-tuning pass — captures screenshots of key UI states so the
// dev can review without launching a browser. Not a pass/fail suite;
// each test saves a PNG to ``test-results/visual/*``.
//
// Run with: npm run e2e -- --grep visual --workers 1

import fs from "node:fs";
import path from "node:path";
import { test } from "@playwright/test";
import { installWsStub, pushBoot, pushChessState, pushFace, skipOnboarding } from "./fixtures/mockWs";

const SHOTS_DIR = "test-results/visual";

test.beforeAll(async () => {
  fs.mkdirSync(SHOTS_DIR, { recursive: true });
});

async function shot(page: import("@playwright/test").Page, name: string) {
  await page.screenshot({ path: path.join(SHOTS_DIR, `${name}.png`), fullPage: true });
}

test.describe("visual — Rook states", () => {
  test.beforeEach(async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 });
    await installWsStub(page);
    await skipOnboarding(page);
    await page.goto("/?mode=rook");
    await pushBoot(page);
    // Give Lottie a tick to load and paint.
    await page.waitForTimeout(800);
  });

  test("initial state (white to move, calm)", async ({ page }) => {
    await shot(page, "01-initial");
  });

  test("mid-game thinking — grandmaster", async ({ page }) => {
    await pushFace(page, { mood: "curious", tempo: "thinking", persona: "grandmaster" });
    await pushChessState(page, {
      fen: "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
      ply: 2,
      turn: "white",
      last_move_san: "Nf6",
      san_history: ["d4", "Nf6"],
      eval_cp: 20,
    });
    await page.evaluate(() => {
      (window as unknown as { __rookTest: { push: (m: unknown) => void } }).__rookTest.push({
        type: "state",
        value: "thinking",
      });
    });
    await page.waitForTimeout(800);
    await shot(page, "02-thinking-grandmaster");
  });

  test("amused — trash talker speaking", async ({ page }) => {
    await pushFace(page, { mood: "amused", tempo: "speaking", persona: "trash_talker" });
    await pushChessState(page, {
      fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
      ply: 3,
      turn: "black",
      last_move_san: "Nf3",
      san_history: ["e4", "e5", "Nf3"],
      eval_cp: 35,
    });
    await page.evaluate(() => {
      (window as unknown as { __rookTest: { push: (m: unknown) => void } }).__rookTest.push({
        type: "bot_text",
        text: "Nice try. I'm developing pieces for a solid attack.",
        latency: 0.8,
      });
    });
    await page.waitForTimeout(800);
    await shot(page, "03-amused-trashtalker");
  });

  test("game-over victory banner", async ({ page }) => {
    await pushChessState(page, {
      fen: "rnbqkb1r/pppp1Qpp/5n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
      ply: 7,
      turn: "black",
      last_move_san: "Qxf7#",
      is_game_over: true,
      game_over_reason: "checkmate",
      winner: "white",
      san_history: ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"],
    });
    await page.waitForTimeout(500);
    await shot(page, "04-game-over");
  });

  test("in-check pulse", async ({ page }) => {
    await pushChessState(page, {
      fen: "rnb1kbnr/pppp1ppp/8/4p3/4P1Pq/8/PPPP1P1P/RNBQKBNR w KQkq - 1 3",
      ply: 4,
      turn: "white",
      last_move_san: "Qh4+",
      san_history: ["f3", "e5", "g4", "Qh4+"],
    });
    await page.waitForTimeout(500);
    await shot(page, "05-in-check");
  });

  test("narrow viewport (mobile)", async ({ page }) => {
    await page.setViewportSize({ width: 420, height: 860 });
    await page.waitForTimeout(400);
    await pushFace(page, { mood: "calm", tempo: "idle", persona: "casual" });
    await shot(page, "06-mobile");
  });

  test("busy — rook thinking indicator over time", async ({ page }) => {
    await pushChessState(page, {
      fen: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
      ply: 2,
      turn: "black",
      last_move_san: "e5",
      san_history: ["e4", "e5"],
    });
    await page.evaluate(() => {
      (window as unknown as { __rookTest: { push: (m: unknown) => void } }).__rookTest.push({
        type: "state",
        value: "thinking",
      });
    });
    await page.waitForTimeout(2200);
    await shot(page, "07-thinking-timer");
  });
});
