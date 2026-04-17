"""Subprocess probe for a working MuJoCo viewer context.

``mujoco.viewer.launch_passive`` segfaults at the C level on some Linux
GL stacks — Wayland + proprietary NVIDIA drivers, broken GLFW installs,
WSLg without viewer support, remote X without direct rendering. A
C-level segfault kills the whole Python process, so catching it with
``try/except`` is not possible.

Workaround: before the main process tries to open the viewer, spawn a
short-lived child that attempts the same GLFW context creation. If the
child exits 0 in time, the viewer is safe to launch; any other outcome
(segfault, non-zero exit, timeout, missing DISPLAY) means we should
fall back to a headless sim with a friendly warning.

The probe result is cached per-process so the 1-2 s startup cost is
paid at most once.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from functools import lru_cache

logger = logging.getLogger(__name__)

# Exercise the ACTUAL ``mujoco.viewer.launch_passive`` call path, not
# just GLFW — some stacks (Wayland/NVIDIA, WSLg) pass a bare GLFW probe
# but still segfault inside launch_passive. A ``MJ_OK`` sentinel on
# stdout is the success signal because Wayland cleanup can hang the
# child at exit even after the viewer opened successfully.
_PROBE_SCRIPT = """
import sys, time
try:
    import mujoco, mujoco.viewer
    # Minimal in-memory scene — no file IO, no scene-specific bugs.
    xml = '<mujoco><worldbody><body><geom type="box" size="0.1 0.1 0.1"/></body></worldbody></mujoco>'
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    v = mujoco.viewer.launch_passive(m, d)
    # One sync ensures the GL context is actually rendering without
    # crashing. If we reach the print, the viewer is safe to launch
    # in the parent. Flush before the Wayland cleanup hang.
    v.sync()
    print('MJ_OK', flush=True)
    v.close()
    sys.exit(0)
except Exception as e:
    print(f'MJ_ERR {type(e).__name__}: {e}', file=sys.stderr, flush=True)
    sys.exit(2)
"""


@lru_cache(maxsize=1)
def viewer_available(timeout_s: float = 6.0) -> tuple[bool, str]:
    """Return ``(ok, reason)`` describing whether a viewer window can
    safely be opened. The result is cached so repeat calls are free.

    The probe watches for a ``MJ_OK`` sentinel on the child's stdout
    rather than its exit code — Wayland cleanup sometimes hangs a
    perfectly-working viewer at shutdown, and we don't want to
    misreport that as a failure.
    """
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False, "no $DISPLAY / $WAYLAND_DISPLAY set"

    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", _PROBE_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as e:
        return False, f"could not launch viewer probe ({e})"

    # Read stdout line-by-line until we see MJ_OK or the child exits.
    # This avoids waiting for Wayland cleanup hangs on valid sessions.
    import time as _time

    deadline = _time.monotonic() + timeout_s
    seen_ok = False
    try:
        assert proc.stdout is not None
        while _time.monotonic() < deadline:
            rc = proc.poll()
            if rc is not None:
                break
            line = proc.stdout.readline()
            if not line:
                _time.sleep(0.05)
                continue
            if "MJ_OK" in line:
                seen_ok = True
                break
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                proc.kill()

    if seen_ok:
        return True, "ok"

    rc = proc.poll() or 0
    if rc < 0:
        sig = -rc
        return False, f"viewer probe crashed with signal {sig} (likely segfault in MuJoCo/GL)"
    stderr = (proc.stderr.read() if proc.stderr else "").strip()
    if stderr:
        logger.debug("viewer probe stderr: %s", stderr)
        if "MJ_ERR" in stderr:
            first = stderr.splitlines()[-1]
            return False, f"viewer probe failed: {first.removeprefix('MJ_ERR ').strip()}"
    if rc == 0:
        return False, "viewer probe finished without MJ_OK sentinel"
    return False, f"viewer probe exited {rc}"


__all__ = ["viewer_available"]
