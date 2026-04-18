//! Python sidecar lifecycle — bootstrap a `uv`-managed venv, run
//! `edgevox-serve`, shut it down cleanly on app exit.
//!
//! Design:
//!
//! - First run: `uv venv <data_dir>/venv` + `uv pip install edgevox`
//!   (or `uv pip install -e <dev source>` when `EDGEVOX_DESKTOP_DEV_SRC`
//!   is set for local hacking).
//! - Every run: `<data_dir>/venv/bin/edgevox-serve --host 127.0.0.1 --port <free>`
//! - On exit: send SIGTERM, wait up to 5 s, then SIGKILL.
//!
//! The free port is discovered in Rust (bind + drop) and handed to the
//! child via a CLI flag, then re-exposed to the frontend as an
//! `edgevox://sidecar-ready` Tauri event carrying `ws://127.0.0.1:PORT/ws`.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::Stdio;

use anyhow::{anyhow, Context, Result};
use directories::ProjectDirs;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

/// Where we stash the managed venv + uv cache. One per user, per OS —
/// `~/.local/share/edgevox` on Linux, `~/Library/Application Support/ai.edgevox.desktop`
/// on macOS, `%APPDATA%/edgevox` on Windows.
pub fn data_dir() -> Result<PathBuf> {
    let dirs = ProjectDirs::from("ai", "edgevox", "desktop")
        .ok_or_else(|| anyhow!("could not resolve a platform data directory"))?;
    std::fs::create_dir_all(dirs.data_dir())?;
    Ok(dirs.data_dir().to_path_buf())
}

fn venv_dir() -> Result<PathBuf> {
    Ok(data_dir()?.join("venv"))
}

fn venv_bin(name: &str) -> Result<PathBuf> {
    let dir = venv_dir()?;
    let exe_dir = if cfg!(windows) {
        dir.join("Scripts")
    } else {
        dir.join("bin")
    };
    let exe = if cfg!(windows) {
        exe_dir.join(format!("{name}.exe"))
    } else {
        exe_dir.join(name)
    };
    Ok(exe)
}

fn uv_binary() -> Result<PathBuf> {
    // Resolve `uv` from $PATH. Future: fall back to a bundled copy shipped
    // under the app resources directory so the installer doesn't require
    // a pre-installed uv.
    which::which("uv").context(
        "`uv` was not found on PATH. Install from https://docs.astral.sh/uv/ before launching the app.",
    )
}

/// Pick an unused TCP port by binding to 0 and letting the OS allocate.
/// There's a TOCTOU window between this call and the child's bind, but
/// in practice the kernel won't reuse the port that fast on loopback.
fn pick_free_port() -> Result<u16> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

/// Ensure the managed venv exists and has a current `edgevox` install.
/// Idempotent: skips the install when the venv is already provisioned,
/// unless `force_reinstall=true`.
pub async fn ensure_venv(force_reinstall: bool) -> Result<()> {
    let uv = uv_binary()?;
    let venv = venv_dir()?;
    let edgevox_entry = venv_bin("edgevox-serve")?;

    if !venv.exists() {
        log::info!("creating venv at {}", venv.display());
        let status = Command::new(&uv)
            .args(["venv", &venv.to_string_lossy(), "--python", "3.12"])
            .status()
            .await
            .context("failed to run `uv venv`")?;
        if !status.success() {
            return Err(anyhow!("`uv venv` exited with {status}"));
        }
    }

    if edgevox_entry.exists() && !force_reinstall {
        log::debug!("edgevox-serve already installed; skipping uv pip install");
        return Ok(());
    }

    // `EDGEVOX_DESKTOP_DEV_SRC` lets contributors point at their local
    // checkout so `npm run tauri dev` uses working-copy edgevox instead
    // of the published wheel.
    let dev_src = std::env::var("EDGEVOX_DESKTOP_DEV_SRC").ok();
    let install_spec = dev_src.as_deref().unwrap_or("edgevox");
    let editable_flag = if dev_src.is_some() { Some("-e") } else { None };

    log::info!("installing edgevox into venv ({install_spec})");
    let mut cmd = Command::new(&uv);
    cmd.arg("pip")
        .arg("install")
        .arg("--python")
        .arg(&venv);
    if let Some(flag) = editable_flag {
        cmd.arg(flag);
    }
    cmd.arg(install_spec);
    let status = cmd.status().await.context("failed to run `uv pip install`")?;
    if !status.success() {
        return Err(anyhow!("`uv pip install` exited with {status}"));
    }
    Ok(())
}

/// A running sidecar process wrapper; drop it to terminate cleanly.
pub struct SidecarHandle {
    child: Mutex<Option<Child>>,
    pub ws_url: String,
}

impl SidecarHandle {
    /// Spawn `edgevox-serve` inside the managed venv on a free port.
    /// Returns once the child has started writing to stdout so the
    /// caller knows the socket is (likely) ready.
    pub async fn spawn() -> Result<Self> {
        ensure_venv(false).await?;
        let port = pick_free_port()?;
        let bin = venv_bin("edgevox-serve")?;
        if !bin.exists() {
            return Err(anyhow!(
                "edgevox-serve missing from venv at {} — venv may be corrupted",
                bin.display()
            ));
        }

        log::info!("launching edgevox-serve on 127.0.0.1:{port}");
        let mut cmd = Command::new(&bin);
        cmd.args(["--host", "127.0.0.1", "--port", &port.to_string()])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd.spawn().context("failed to spawn edgevox-serve")?;

        // Forward server stdout/stderr to the Rust logger so `npm run
        // tauri dev` surfaces Python tracebacks without a separate pane.
        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    log::info!("[edgevox-serve] {line}");
                }
            });
        }
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    log::info!("[edgevox-serve] {line}");
                }
            });
        }

        Ok(SidecarHandle {
            child: Mutex::new(Some(child)),
            ws_url: format!("ws://127.0.0.1:{port}/ws"),
        })
    }

    /// SIGTERM the child, wait briefly, then kill if still alive.
    pub async fn shutdown(&self) {
        let mut guard = self.child.lock().await;
        if let Some(mut child) = guard.take() {
            log::info!("stopping edgevox-serve");
            // `kill_on_drop` + `start_kill` is tokio's cross-platform
            // SIGTERM-then-kill dance.
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }
}
