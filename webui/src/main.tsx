import { useEffect, useState } from "react";
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import RookApp from "./components/RookApp";
import { SetupScreen, type SetupState } from "./components/SetupScreen";
import { TitleBar } from "./components/TitleBar";
import "./index.css";

// Layout selector:
//   1. ``?mode=rook`` query param
//   2. Tauri ``invoke("get_mode")``  — desktop build with ROOK_MODE=1
//   3. ``window.__EDGEVOX_MODE__``
//   4. Fallback: generic App
async function pickMode(): Promise<string> {
  const params = new URLSearchParams(window.location.search);
  const query = params.get("mode");
  if (query) return query;

  const tauriInternals = (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  if (tauriInternals) {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      return await invoke<string>("get_mode");
    } catch (e) {
      console.warn("Tauri get_mode failed; using default layout", e);
    }
  }

  const injected = (window as unknown as { __EDGEVOX_MODE__?: string }).__EDGEVOX_MODE__;
  return injected ?? "default";
}

function isTauri(): boolean {
  return Boolean((window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__);
}

/** Root shell for the Rook layout in a Tauri desktop app — waits for
 * the sidecar to finish provisioning before mounting the main UI.
 * In plain-browser mode, we assume the server is already running and
 * skip the setup gate entirely. */
function RookRoot() {
  const [state, setState] = useState<SetupState>(isTauri() ? "booting" : "ready");
  const [error, setError] = useState<string | undefined>(undefined);

  useEffect(() => {
    if (!isTauri()) return;
    let cancelled = false;
    (async () => {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke<string>("get_ws_url");
        if (!cancelled) setState("ready");
      } catch (e) {
        if (!cancelled) {
          setError((e as Error).message ?? String(e));
          setState("error");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const inner =
    state === "ready" ? (
      <RookApp />
    ) : (
      <SetupScreen
        state={state}
        errorMessage={error}
        onRetry={() => {
          setError(undefined);
          setState("booting");
          window.location.reload();
        }}
      />
    );
  return (
    <div style={{ height: "100vh", width: "100vw", display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <TitleBar />
      <div style={{ flex: 1, minHeight: 0 }}>{inner}</div>
    </div>
  );
}

async function bootstrap() {
  const mode = await pickMode();
  const root = mode === "rook" ? <RookRoot /> : <App />;
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>{root}</React.StrictMode>
  );
}

void bootstrap();
