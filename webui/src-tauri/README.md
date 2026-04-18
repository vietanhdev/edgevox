# EdgeVox Desktop (Tauri shell)

Native desktop app that wraps the existing React web UI and launches
`edgevox-serve` as a Python sidecar managed by [`uv`](https://docs.astral.sh/uv/).

## Architecture

```
┌──────────────────────────────────────────────┐
│ Tauri window (Rust)                          │
│  ┌────────────────────────────────────────┐  │
│  │ Webview (React, from webui/dist)       │  │
│  │  WebSocket → ws://127.0.0.1:PORT/ws    │  │
│  └────────────────────────────────────────┘  │
│                                              │
│  Rust sidecar manager:                       │
│   • `uv venv` in <data_dir>/venv             │
│   • `uv pip install edgevox`  (first run)    │
│   • spawn `edgevox-serve --port <free>`      │
│   • expose URL via `get_ws_url` command      │
└──────────────────────────────────────────────┘
```

Models (STT / LLM / TTS ~3-5 GB total) still download to the Hugging
Face cache on first use — they are **not** bundled in the installer.

## Prereqs

Pick-your-poison install guides linked; none are bundled yet.

- Rust ≥ 1.75 — https://rustup.rs/
- Node.js ≥ 18 — for the Vite dev server and `tauri-cli`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) on `$PATH`
- OS-specific Tauri deps: https://tauri.app/start/prerequisites/

## Dev loop

From `webui/`:

```bash
# One-time
npm install
cargo install tauri-cli --version '^2.0'

# Point the sidecar at your working-copy edgevox (optional):
export EDGEVOX_DESKTOP_DEV_SRC=$(git rev-parse --show-toplevel)

# Launches Vite on :5173 + the native window; auto-spawns the sidecar
# on first frontend `get_ws_url` call.
cargo tauri dev
```

On first launch the sidecar provisions a venv and runs `uv pip install`
into it — takes a minute or two. Subsequent launches are ~1 second.

## Shipping an installer

```bash
npm run build           # builds React → webui/dist
cargo tauri build       # bundles the desktop binary + installer
```

Output lives under `src-tauri/target/release/bundle/`:

- Linux — `.deb`, `.AppImage`, `.rpm`
- macOS — `.app`, `.dmg`
- Windows — `.msi`, `.exe`

Installer size is ~10-20 MB: it doesn't include the venv; the sidecar
provisions one on first run.

## Limitations (prototype)

- **`uv` not bundled** — users install it themselves. Bundling it as an
  app resource and pointing `uv_binary()` at the resource path is the
  next step. See `sidecar.rs::uv_binary()`.
- **`get_ws_url` is blocking** — first call waits through `uv pip install`.
  Should emit progress events via `app_handle.emit(...)` so the frontend
  can show a splash.
- **No icon set yet** — run `cargo tauri icon <source.png>` from
  `webui/` with a 1024×1024 logo to generate the files listed in
  `tauri.conf.json`. Until then `cargo tauri build` will fail on the
  bundle step (dev mode works without icons).
- **Python backend runs through `LLMAgent`** — for chess specifically,
  the existing server path (`edgevox/server/ws.py`) bypasses
  `LLMAgent`, so `chess_state` events don't yet reach the web UI. A
  chess-aware server mode is a follow-up; the React components
  (`ChessBoard.tsx` etc.) already consume the message shape.
