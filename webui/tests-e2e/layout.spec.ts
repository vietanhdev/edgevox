// Basic page-level smoke tests — verifies the Rook layout renders,
// onboarding + connection handshake work, and the major structural
// regions exist. Uses a mocked WebSocket so no Python server needed.

import { test, expect } from "@playwright/test";
import { installWsStub, pushBoot, skipOnboarding } from "./fixtures/mockWs";

test.describe("Rook layout — page load", () => {
  test("renders with default layout (App) when no mode param", async ({ page }) => {
    await page.goto("/");
    // Default App layout still shows the EdgeVox brand somewhere.
    await expect(page).toHaveTitle(/EdgeVox|EVox/);
  });

  test("mounts Rook layout on ?mode=rook", async ({ page }) => {
    await installWsStub(page);
    await skipOnboarding(page);
    await page.goto("/?mode=rook");
    // Title swaps to RookApp when RookApp mounts.
    await expect(page).toHaveTitle(/RookApp/);
    await pushBoot(page);
    // Header shows the brand.
    await expect(page.getByText("RookApp")).toBeVisible();
    // Robot face is rendered.
    await expect(page.getByTestId("robot-face")).toBeVisible();
    // Turn pill exists.
    await expect(page.getByTestId("turn-pill")).toBeVisible();
  });

  test("onboarding overlay shows on first load, dismisses on click", async ({ page }) => {
    await installWsStub(page);
    await page.goto("/?mode=rook");
    await pushBoot(page);
    // Overlay visible.
    await expect(page.getByText(/click anywhere to start/i)).toBeVisible();
    // Click anywhere to dismiss.
    await page.locator("body").click();
    await expect(page.getByText(/click anywhere to start/i)).toBeHidden();
  });

  test("empty chat log shows the prompt hint", async ({ page }) => {
    await installWsStub(page);
    await skipOnboarding(page);
    await page.goto("/?mode=rook");
    await pushBoot(page);
    await expect(page.getByText(/talk or type a move to start/i)).toBeVisible();
  });
});
