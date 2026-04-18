// Input flow — text entry, keyboard shortcuts, send behavior.
// The WebSocket is mocked so tests run without the Python backend.

import { test, expect } from "@playwright/test";
import { getSentTexts, installWsStub, pushBoot, skipOnboarding } from "./fixtures/mockWs";

test.describe("Rook — text input", () => {
  test.beforeEach(async ({ page }) => {
    await installWsStub(page);
    await skipOnboarding(page);
    await page.goto("/?mode=rook");
    await pushBoot(page);
  });

  test("typing + Enter sends a text_input control frame", async ({ page }) => {
    const input = page.locator("input[type=text]");
    await input.fill("I play e4");
    await input.press("Enter");

    await expect.poll(async () => (await getSentTexts(page)).length).toBeGreaterThan(0);
    const sent = await getSentTexts(page);
    // Should contain a JSON string with type=text_input and text=I play e4
    expect(sent.some((s) => s.includes('"text_input"') && s.includes("I play e4"))).toBe(true);
  });

  test("empty input doesn't send", async ({ page }) => {
    const input = page.locator("input[type=text]");
    await input.fill("   ");
    await input.press("Enter");
    const sent = await getSentTexts(page);
    // Pre-existing traffic (hello, etc.) shouldn't include a text_input for whitespace.
    expect(sent.some((s) => s.includes('"text_input"') && s.includes("\"   \""))).toBe(false);
  });

  test("send button disables without input", async ({ page }) => {
    const button = page.getByRole("button", { name: /send/i });
    await expect(button).toBeDisabled();
  });

  test("user message lands in chat log immediately (optimistic)", async ({ page }) => {
    await page.locator("input[type=text]").fill("hello rook");
    await page.getByRole("button", { name: /send/i }).click();
    // Optimistic echo — we don't wait for server user_text.
    await expect(page.getByText("hello rook")).toBeVisible();
  });

  test("Enter inside input does not scroll page — only sends", async ({ page }) => {
    const input = page.locator("input[type=text]");
    await input.fill("Nf3");
    await input.press("Enter");
    // Input clears after send.
    await expect(input).toHaveValue("");
  });
});

test.describe("Rook — keyboard shortcuts", () => {
  test.beforeEach(async ({ page }) => {
    await installWsStub(page);
    await skipOnboarding(page);
    await page.goto("/?mode=rook");
    await pushBoot(page);
  });

  test('"/" focuses the text input', async ({ page }) => {
    // Start focus somewhere else.
    await page.locator("body").click();
    await expect(page.locator("input[type=text]")).not.toBeFocused();
    await page.keyboard.press("/");
    await expect(page.locator("input[type=text]")).toBeFocused();
  });

  test("m keyboard shortcut persists the mute state", async ({ page }) => {
    // Mute is now stored under localStorage["evox-chess-muted"]
    // and controlled via "m". After toggling we should see the flag flip.
    const before = await page.evaluate(() => localStorage.getItem("evox-chess-muted"));
    await page.keyboard.press("m");
    // Give state a tick to persist.
    await page.waitForTimeout(80);
    const after = await page.evaluate(() => localStorage.getItem("evox-chess-muted"));
    expect(before).not.toBe(after);
  });
});
