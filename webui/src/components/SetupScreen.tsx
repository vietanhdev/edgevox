// First-run setup screen.
//
// Shows while the Tauri sidecar provisions its venv + installs
// edgevox + launches the server. The steps below roughly track what
// ``sidecar.rs::ensure_venv`` does. We don't have per-step progress
// events (yet) — instead we drive the checklist by elapsed time,
// promoting each step after a representative duration so the user
// sees forward motion. Once ``get_ws_url`` resolves, we flip to
// READY and hand off to RookApp.
//
// Making the first-run experience feel good matters: installing
// Python deps + downloading models on first launch can take 30-120
// seconds depending on network. A static "Loading…" spinner for that
// long feels broken; a checklist that moves feels intentional.

import { useEffect, useMemo, useState } from "react";
import { Check, Download, Loader, Sparkles } from "lucide-react";

export type SetupState = "booting" | "ready" | "error";

interface StepDef {
  id: string;
  label: string;
  detail: string;
  /** How many seconds after boot this step's "in progress" animation
   *  should begin. Rough cadence of the real sidecar. */
  startAt: number;
  /** How many seconds after boot this step should show as complete. */
  doneAt: number;
}

const STEPS: StepDef[] = [
  {
    id: "runtime",
    label: "Checking the Python runtime",
    detail: "Finding uv + a compatible Python interpreter.",
    startAt: 0,
    doneAt: 1.5,
  },
  {
    id: "venv",
    label: "Preparing a virtual environment",
    detail: "One-time setup. Cached for future launches.",
    startAt: 1.5,
    doneAt: 4.0,
  },
  {
    id: "deps",
    label: "Installing chess + voice dependencies",
    detail: "Small models, voice pipeline, stockfish wiring…",
    startAt: 4.0,
    doneAt: 45.0,
  },
  {
    id: "launch",
    label: "Starting the local server",
    detail: "Connecting to the built-in voice pipeline.",
    startAt: 45.0,
    doneAt: 55.0,
  },
  {
    id: "models",
    label: "Warming up the AI",
    detail: "Loading speech + language models into memory.",
    startAt: 55.0,
    doneAt: 999.0, // Advances only when get_ws_url resolves.
  },
];

interface SetupScreenProps {
  state: SetupState;
  errorMessage?: string;
  onRetry?: () => void;
  accent?: string;
}

export function SetupScreen({
  state,
  errorMessage,
  onRetry,
  accent = "#34d399",
}: SetupScreenProps) {
  const [startMs] = useState(() => Date.now());
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (state !== "booting") return;
    const id = setInterval(() => setTick((t) => t + 1), 500);
    return () => clearInterval(id);
  }, [state]);

  const elapsed = (Date.now() - startMs) / 1000;
  const steps = useMemo(() => STEPS, []);

  // Derive per-step status from elapsed time (when booting) or force
  // all-complete (when ready).
  const status = (step: StepDef): "pending" | "active" | "done" => {
    if (state === "ready") return "done";
    if (elapsed >= step.doneAt) return "done";
    if (elapsed >= step.startAt) return "active";
    return "pending";
  };

  const currentStep = steps.find((s) => status(s) === "active") ?? steps[steps.length - 1];

  return (
    <div
      data-testid="setup-screen"
      style={{
        height: "100vh",
        width: "100vw",
        background: "radial-gradient(circle at 50% 30%, #0f1622 0%, #060a12 75%)",
        color: "#dbe4ef",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "32px",
        fontFamily: "ui-sans-serif, system-ui, -apple-system, 'Segoe UI'",
      }}
    >
      <div
        style={{
          fontSize: 40,
          fontWeight: 700,
          color: accent,
          marginBottom: 6,
          letterSpacing: "0.02em",
        }}
      >
        RookApp
      </div>
      <div style={{ fontSize: 14, color: "#8796a8", marginBottom: 36, textAlign: "center" }}>
        {state === "error"
          ? "Setup hit a snag."
          : state === "ready"
          ? "All set — here we go!"
          : "Setting things up for your first match"}
      </div>

      {state === "error" ? (
        <div
          style={{
            maxWidth: 520,
            background: "rgba(239, 68, 68, 0.12)",
            border: "1px solid rgba(239, 68, 68, 0.5)",
            borderRadius: 10,
            padding: "18px 22px",
            marginBottom: 20,
            fontSize: 13,
            lineHeight: 1.5,
          }}
        >
          <div style={{ color: "#ef8a8a", fontWeight: 600, marginBottom: 6 }}>
            {errorMessage ?? "Unknown error"}
          </div>
          <div style={{ color: "#c7b2b2" }}>
            Common causes: <code>uv</code> not on PATH, no network, or a
            Python install conflict. Try again after confirming{" "}
            <code>uv --version</code> works in a terminal.
          </div>
        </div>
      ) : (
        <>
          <ul
            style={{
              listStyle: "none",
              margin: 0,
              padding: 0,
              display: "flex",
              flexDirection: "column",
              gap: 10,
              minWidth: 420,
              maxWidth: 560,
              width: "100%",
            }}
          >
            {steps.map((s) => {
              const st = status(s);
              return (
                <li
                  key={s.id}
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 12,
                    padding: "12px 14px",
                    background:
                      st === "active"
                        ? `${accent}14`
                        : st === "done"
                        ? "rgba(52, 211, 153, 0.06)"
                        : "rgba(20, 29, 42, 0.4)",
                    border: `1px solid ${
                      st === "active"
                        ? accent
                        : st === "done"
                        ? "rgba(52, 211, 153, 0.25)"
                        : "#1e2b3a"
                    }`,
                    borderRadius: 8,
                    transition: "background 200ms, border-color 200ms",
                    opacity: st === "pending" ? 0.55 : 1,
                  }}
                >
                  <div
                    style={{
                      width: 24,
                      height: 24,
                      borderRadius: 12,
                      background: st === "done" ? accent : st === "active" ? `${accent}33` : "#1a2230",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: st === "done" ? "#0a0e14" : accent,
                      flexShrink: 0,
                      marginTop: 1,
                    }}
                  >
                    {st === "done" ? (
                      <Check size={14} strokeWidth={3} />
                    ) : st === "active" ? (
                      <Loader
                        size={14}
                        style={{ animation: "setup-spin 1.1s linear infinite" }}
                      />
                    ) : (
                      <span style={{ fontSize: 11, opacity: 0.6 }}>·</span>
                    )}
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div
                      style={{
                        fontSize: 14,
                        fontWeight: st === "active" ? 600 : 500,
                        color: st === "pending" ? "#7a8899" : "#dbe4ef",
                      }}
                    >
                      {s.label}
                    </div>
                    <div
                      style={{
                        fontSize: 12,
                        color: "#8796a8",
                        marginTop: 2,
                        lineHeight: 1.4,
                      }}
                    >
                      {s.detail}
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>

          <div
            style={{
              marginTop: 28,
              fontSize: 11,
              fontFamily: "monospace",
              color: "#6a7a8d",
              letterSpacing: "0.05em",
            }}
          >
            {state === "ready" ? (
              <span style={{ display: "inline-flex", alignItems: "center", gap: 6, color: accent }}>
                <Sparkles size={12} /> READY
              </span>
            ) : (
              <span>
                <Download size={11} style={{ display: "inline", verticalAlign: "middle", marginRight: 4 }} />
                elapsed {Math.round(elapsed)}s · {currentStep.label.toLowerCase()}
              </span>
            )}
          </div>
        </>
      )}

      {state === "error" && onRetry && (
        <button
          onClick={onRetry}
          style={{
            marginTop: 10,
            padding: "10px 24px",
            background: accent,
            color: "#0a0e14",
            border: "none",
            borderRadius: 6,
            fontFamily: "ui-sans-serif, system-ui",
            fontSize: 14,
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Try again
        </button>
      )}

      <style>
        {`@keyframes setup-spin { to { transform: rotate(360deg); } }`}
      </style>
    </div>
  );
}

export default SetupScreen;
