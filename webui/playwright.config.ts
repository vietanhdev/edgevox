// Playwright config for Rook UI tests.
//
// Tests hit the Vite dev server directly — no Python backend needed —
// so they stay fast (<5s per file) and deterministic. The WebSocket
// layer is mocked inside each test via ``page.routeWebSocket`` or
// ``page.addInitScript`` that injects a fake ``EdgeVoxWs``.

import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests-e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? [["github"], ["list"]] : "list",
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "retain-on-failure",
    // We run against system-installed Chrome because this host's
    // Ubuntu 26.04 isn't in Playwright's browser matrix. No download,
    // just reuse what's already there.
    channel: "chrome",
  },
  webServer: {
    command: "npm run dev -- --port 5173",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 30_000,
    stdout: "pipe",
    stderr: "pipe",
  },
  projects: [
    {
      name: "chrome",
      use: { ...devices["Desktop Chrome"], channel: "chrome" },
    },
  ],
});
