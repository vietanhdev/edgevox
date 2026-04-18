// Prevent an extra console window from appearing on Windows release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod sidecar;

use std::sync::Arc;

use tauri::{Manager, RunEvent, State};
use tokio::sync::OnceCell;

use crate::sidecar::SidecarHandle;

// One sidecar per app run. ``OnceCell`` lets the Tauri command and the
// window-ready handler share the same handle without racing each other
// to spawn duplicate processes.
struct AppState {
    sidecar: Arc<OnceCell<Arc<SidecarHandle>>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            sidecar: Arc::new(OnceCell::new()),
        }
    }
}

/// Frontend calls this once on startup to learn where the sidecar is
/// listening. Returns a ``ws://127.0.0.1:<port>/ws`` URL. Blocks until
/// the sidecar is fully spawned; the first call can take several
/// seconds on a cold venv (uv pip install).
#[tauri::command]
async fn get_ws_url(state: State<'_, AppState>) -> Result<String, String> {
    let sidecar = state
        .sidecar
        .get_or_try_init(|| async {
            SidecarHandle::spawn()
                .await
                .map(Arc::new)
                .map_err(|e| e.to_string())
        })
        .await?;
    Ok(sidecar.ws_url.clone())
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let state = AppState::new();
    let sidecar_for_exit = state.sidecar.clone();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![get_ws_url])
        .build(tauri::generate_context!())
        .expect("failed to build Tauri app")
        .run(move |_app, event| {
            // Cleanly terminate the sidecar on app exit so a stale
            // uvicorn doesn't linger on the port between runs.
            if let RunEvent::Exit = event {
                if let Some(handle) = sidecar_for_exit.get() {
                    let handle = handle.clone();
                    tauri::async_runtime::block_on(async move {
                        handle.shutdown().await;
                    });
                }
            }
        });
}
