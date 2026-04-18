"""MuJoCo humanoid adapter — EdgeVox Tier 2b (Unitree G1 / H1).

Loads a Unitree Menagerie humanoid and drives it from voice commands.
The scene + meshes are fetched from the EdgeVox HuggingFace model repo
(``nrl-ai/edgevox-models``) on first use, with a git sparse-clone
fallback to upstream ``google-deepmind/mujoco_menagerie``.

**Honest scope note.** Real humanoid locomotion is an RL problem — a
robust walking policy needs to be trained against the specific robot's
URDF, mass distribution, and actuator model. Shipping a policy inline
is out of scope. Instead the adapter:

1. Loads the model's ``home`` keyframe (Menagerie convention) so the
   robot starts in a balanced standing pose.
2. Runs a **procedural gait** — the freejoint's translation + yaw
   advance at the commanded speed while leg + arm actuators follow a
   sinusoidal walk cycle so the humanoid *visibly* steps instead of
   gliding. Joints are matched by name pattern
   (``{side}_hip_pitch``, ``{side}_knee``, etc.), so this works for
   any Menagerie humanoid that follows the convention.
3. Exposes :meth:`set_walking_policy` so callers can plug in an ONNX
   RL policy that replaces the gait generator entirely.

Threading model mirrors the arm adapter: physics on a daemon thread,
rendering on main via :meth:`pump_render`, skills driven by
``GoalHandle`` with cancellation checked every tick.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from edgevox.agents.skills import GoalHandle, GoalStatus

try:
    import mujoco
    import mujoco.viewer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "mujoco is not installed. Run `pip install 'edgevox[sim-mujoco]'` to enable the "
        "MujocoHumanoidEnvironment adapter."
    ) from e

log = logging.getLogger(__name__)

# Supported Unitree humanoids. Each maps to the relative path inside
# the EdgeVox HuggingFace model repo (mirrors of the corresponding
# google-deepmind/mujoco_menagerie subdirectories). The HF repo is the
# primary source — fast, cached under the standard HF cache — with a
# git-clone fallback to the upstream Menagerie if HF is unreachable.
_HF_REPO = "nrl-ai/edgevox-models"
_HF_SUBDIR_TMPL = "mujoco_scenes/{name}"

_MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
_MENAGERIE_CACHE_DIR = Path.home() / ".cache" / "edgevox" / "menagerie"
_MENAGERIE_MODELS: dict[str, tuple[str, str]] = {
    "unitree_g1": ("unitree_g1", "scene.xml"),
    "unitree_h1": ("unitree_h1", "scene.xml"),
}
_DEFAULT_MODEL = "unitree_g1"


def _fetch_from_hf(name: str) -> Path | None:
    """Download the named humanoid scene bundle from the EdgeVox HF
    model repo. Returns the local path to ``scene.xml``, or ``None``
    if the fetch fails for any reason (offline, repo not yet populated)
    so the caller can fall back to git."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None

    subdir = _HF_SUBDIR_TMPL.format(name=name)
    try:
        local_dir = snapshot_download(
            repo_id=_HF_REPO,
            repo_type="model",
            allow_patterns=f"{subdir}/**",
        )
    except Exception as e:
        log.info("HF fetch for %s failed (%s); will try git fallback", name, e)
        return None

    scene = Path(local_dir) / subdir / "scene.xml"
    if not scene.exists():
        log.info("HF fetch succeeded but scene.xml missing for %s", name)
        return None
    return scene


def _fetch_from_menagerie_git(name: str) -> Path:
    """Fallback: sparse-clone the upstream MuJoCo Menagerie repo and
    extract just the named humanoid subdirectory."""
    import shutil
    import subprocess

    subdir, scene_file = _MENAGERIE_MODELS[name]
    root = _MENAGERIE_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    clone_dir = root / "mujoco_menagerie"
    scene_path = clone_dir / subdir / scene_file
    if scene_path.exists():
        return scene_path

    if shutil.which("git") is None:
        raise RuntimeError(
            f"{name} isn't cached locally, HuggingFace fetch failed, and git is "
            "not on PATH for the Menagerie fallback — install git or pre-populate "
            f"{scene_path} manually."
        )

    log.info("fetching menagerie/%s via git into %s (one-time)", subdir, clone_dir)
    if not clone_dir.exists():
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth=1",
                "--filter=blob:none",
                "--sparse",
                _MENAGERIE_REPO,
                str(clone_dir),
            ]
        )
    subprocess.check_call(
        ["git", "-C", str(clone_dir), "sparse-checkout", "set", subdir],
    )
    if not scene_path.exists():
        raise RuntimeError(f"expected scene at {scene_path} after clone but nothing landed")
    return scene_path


def _fetch_menagerie_model(name: str) -> Path:
    """Resolve the local ``scene.xml`` path for a supported Unitree
    humanoid. Prefers the EdgeVox HF repo (fast, cached); falls back
    to a git sparse-clone of upstream ``mujoco_menagerie``."""
    if name not in _MENAGERIE_MODELS:
        raise FileNotFoundError(f"unknown humanoid {name!r}. Supported: {sorted(_MENAGERIE_MODELS)}")
    hf_path = _fetch_from_hf(name)
    if hf_path is not None:
        return hf_path
    return _fetch_from_menagerie_git(name)


# Locomotion speed limits tuned for the demo. The agent passes
# ``distance`` + ``speed`` to walk skills; higher values just mean a
# longer-running goal, they don't change these caps.
_WALK_SPEED_MPS = 0.6
_TURN_SPEED_RAD = 0.8
_POSITION_TOL = 0.05  # metres
_HEADING_TOL = 0.05  # rad


class WalkingPolicy(Protocol):
    """Optional pluggable policy hook. Any object matching this
    protocol can be registered via :meth:`set_walking_policy`.

    ``reset`` is called when the env resets. ``step`` is called every
    physics tick with the current observation; the returned action is
    written to ``data.ctrl`` and replaces the mocap-style stub.
    """

    def reset(self) -> None: ...
    def step(self, obs: np.ndarray, command: np.ndarray) -> np.ndarray: ...


class MujocoHumanoidEnvironment:
    """MuJoCo bipedal humanoid environment.

    Args:
        model_path: override the bundled humanoid MJCF.
        render: open the passive viewer window (main-thread only).
        tick_interval: wall-clock seconds between physics batches.
        steps_per_tick: ``mj_step`` calls per batch.
        walking_policy: optional ``WalkingPolicy``. If ``None``, the
            adapter uses the mocap-style root-override stub.
    """

    def __init__(
        self,
        *,
        model_source: str = _DEFAULT_MODEL,
        model_path: str | Path | None = None,
        render: bool = True,
        tick_interval: float = 0.02,
        steps_per_tick: int = 10,
        walking_policy: WalkingPolicy | None = None,
        pelvis_body: str | None = None,
        head_body: str | None = None,
    ) -> None:
        scene_path = self._resolve_scene(model_source, model_path)
        self._scene_path = scene_path
        self._render = render
        self._tick_interval = tick_interval
        self._steps_per_tick = max(1, steps_per_tick)

        self._model = mujoco.MjModel.from_xml_path(str(scene_path))
        self._data = mujoco.MjData(self._model)

        # --- root joint (freejoint) -- required for locomotion stub.
        self._root_jid = self._first_freejoint_id()
        self._root_qpos_adr = int(self._model.jnt_qposadr[self._root_jid]) if self._root_jid >= 0 else -1

        # --- home pose — Menagerie humanoids ship a "home" keyframe
        # that encodes a stable standing pose. Load it so the robot
        # starts upright.
        self._home_qpos = self._resolve_home_qpos()
        self._home_ctrl = self._resolve_home_ctrl()
        self._set_home_state()

        # --- body IDs — auto-detected with Menagerie conventions first.
        self._pelvis_bid = self._resolve_body_id(pelvis_body, candidates=("pelvis", "base_link", "torso_link", "torso"))
        self._head_bid = self._resolve_body_id(head_body, candidates=("head", "head_link", "neck"))

        # --- gait actuators — procedural walk cycle targets during
        # locomotion so legs visibly step instead of gliding. Matched
        # by substring on actuator names, so this works for any
        # Menagerie humanoid that follows the `{side}_{joint}_pitch`
        # convention (Unitree G1 / H1, Berkeley, OP3, ...).
        self._gait_actuators = self._build_gait_actuator_table()

        # Brief warmup so the freejoint rests on the floor contacts —
        # 100 steps at dt=2 ms is ~0.2 s, enough to let the feet settle
        # without giving the PD controller time to destabilise the
        # keyframe pose on high-DoF humanoids like the Unitree H1.
        for _ in range(100):
            mujoco.mj_step(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

        self._initial_qpos = self._data.qpos.copy()
        self._initial_qvel = self._data.qvel.copy()
        self._initial_ctrl = self._data.ctrl.copy()

        # Lock + goal state
        self._lock = threading.RLock()
        self._active_goal: GoalHandle | None = None
        self._active_kind: str | None = None
        self._active_target: np.ndarray | None = None
        self._active_target_heading: float | None = None
        self._active_command: str | None = None

        # Mocap-style root override for the no-policy locomotion stub.
        # When a locomotion goal is running we cache the previous root
        # qvel and write a commanded twist into it each tick.
        self._cmd_vel = np.zeros(2)  # (linear, angular)
        self._manual_deadline = 0.0
        self._gait_phase = 0.0

        # Walking policy hook (optional).
        self._walking_policy: WalkingPolicy | None = None
        if walking_policy is not None:
            self.set_walking_policy(walking_policy)

        # Offscreen camera state (lazy).
        self._offscreen_renderer: Any | None = None
        self._offscreen_size: tuple[int, int] | None = None

        # Physics thread
        self._phys_stop = threading.Event()
        self._phys_thread = threading.Thread(target=self._physics_loop, name="mujoco-humanoid-physics", daemon=True)
        self._phys_thread.start()

        # Viewer — ``mujoco.viewer.launch_passive`` opens a GLFW window.
        # On some Linux GL stacks (Wayland + proprietary NVIDIA, WSLg,
        # broken GLFW installs) the call segfaults at the C level,
        # which kills this process. A subprocess probe tests GLFW
        # context creation first so we can fall back to headless
        # without crashing.
        self._viewer: Any | None = None
        if render:
            from edgevox.integrations.sim._viewer_probe import viewer_available

            ok, reason = viewer_available()
            if not ok:
                log.warning(
                    "MuJoCo viewer unavailable (%s) — continuing headless. Pass --no-render to silence this warning.",
                    reason,
                )
                self._render = False
            else:
                try:
                    self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
                except Exception:
                    log.exception("mujoco.viewer.launch_passive failed; continuing headless")
                    self._viewer = None
                    self._render = False

    # ----- scene / keyframe resolution ----------------------------------

    @staticmethod
    def _resolve_scene(model_source: str, model_path: str | Path | None) -> Path:
        if model_path is not None:
            p = Path(model_path).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"humanoid MJCF not found: {p}")
            return p
        if model_source in _MENAGERIE_MODELS:
            return _fetch_menagerie_model(model_source)
        raise FileNotFoundError(f"unknown model_source {model_source!r}. Supported: {sorted(_MENAGERIE_MODELS)}")

    def _first_freejoint_id(self) -> int:
        for jid in range(self._model.njnt):
            if self._model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                return jid
        return -1

    def _resolve_home_qpos(self) -> Any:
        """Use a "home" keyframe if present (Menagerie convention)."""
        kid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid >= 0:
            return np.asarray(self._model.key_qpos[kid]).copy()
        return None

    def _resolve_home_ctrl(self) -> Any:
        kid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid >= 0 and self._model.key_ctrl is not None and self._model.key_ctrl.size:
            return np.asarray(self._model.key_ctrl[kid]).copy()
        return None

    def _set_home_state(self) -> None:
        if self._home_qpos is not None and self._home_qpos.size == self._data.qpos.size:
            self._data.qpos[:] = self._home_qpos
        if self._home_ctrl is not None and self._home_ctrl.size == self._data.ctrl.size:
            self._data.ctrl[:] = self._home_ctrl
        mujoco.mj_forward(self._model, self._data)

    def _build_gait_actuator_table(self) -> dict[str, int]:
        """Match left/right hip / knee / ankle / shoulder / elbow
        actuators so the procedural gait can drive them. Returns a
        mapping ``"left_hip" -> actuator_id`` keyed by role."""
        roles = {
            "left_hip": ("left", "hip_pitch"),
            "right_hip": ("right", "hip_pitch"),
            "left_knee": ("left", "knee"),
            "right_knee": ("right", "knee"),
            "left_ankle": ("left", "ankle_pitch"),
            "right_ankle": ("right", "ankle_pitch"),
            "left_shoulder": ("left", "shoulder_pitch"),
            "right_shoulder": ("right", "shoulder_pitch"),
            "left_elbow": ("left", "elbow"),
            "right_elbow": ("right", "elbow"),
        }
        table: dict[str, int] = {}
        for aid in range(self._model.nu):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            if name is None:
                continue
            lname = name.lower()
            for role, (side, keyword) in roles.items():
                if role in table:
                    continue
                if side in lname and keyword in lname:
                    table[role] = aid
        return table

    def _resolve_body_id(self, preferred: str | None, *, candidates: tuple[str, ...]) -> int:
        if preferred is not None:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, preferred)
            if bid >= 0:
                return bid
            log.warning("body %r not found in model; trying defaults", preferred)
        for name in candidates:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return bid
        # Last resort: pick the body with the highest initial z that
        # isn't the world body — usually the torso or head.
        best = 1 if self._model.nbody > 1 else 0
        best_z = -1e9
        for bid in range(1, self._model.nbody):
            z = float(self._data.xpos[bid][2])
            if z > best_z:
                best_z = z
                best = bid
        return best

    def _id(self, obj_type: int, name: str) -> int:
        bid = mujoco.mj_name2id(self._model, obj_type, name)
        if bid < 0:
            raise ValueError(f"{name!r} not in humanoid model")
        return bid

    def _root_xy_unlocked(self) -> np.ndarray:
        return self._data.qpos[self._root_qpos_adr : self._root_qpos_adr + 2].copy()

    def _root_yaw_unlocked(self) -> float:
        quat = self._data.qpos[self._root_qpos_adr + 3 : self._root_qpos_adr + 7]
        # z-axis yaw from quaternion (w, x, y, z)
        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _set_root_yaw_unlocked(self, yaw: float) -> None:
        half = yaw * 0.5
        self._data.qpos[self._root_qpos_adr + 3] = math.cos(half)
        self._data.qpos[self._root_qpos_adr + 4] = 0.0
        self._data.qpos[self._root_qpos_adr + 5] = 0.0
        self._data.qpos[self._root_qpos_adr + 6] = math.sin(half)

    # ----- physics loop --------------------------------------------------

    def _physics_loop(self) -> None:
        next_tick = time.perf_counter()
        while not self._phys_stop.is_set():
            try:
                self.tick_physics()
            except Exception:
                log.exception("humanoid physics tick failed")
            next_tick += self._tick_interval
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_tick = time.perf_counter()

    def tick_physics(self) -> None:
        with self._lock:
            # Apply policy or mocap override BEFORE stepping physics so
            # ctrl is fresh for this tick.
            if self._walking_policy is not None:
                self._apply_walking_policy_unlocked()
            else:
                self._apply_locomotion_stub_unlocked()

            for _ in range(self._steps_per_tick):
                mujoco.mj_step(self._model, self._data)
            self._update_active_goal_unlocked()

    # ----- mocap-style locomotion stub ----------------------------------

    # Gait cycle timing + amplitude — tuned so the legs visibly step
    # rather than glide. ``_gait_phase`` advances with commanded
    # linear speed so fast walks take bigger strides; ``turn_in_place``
    # keeps the legs stepping even when only angular velocity is
    # commanded so the humanoid doesn't rotate with stiff legs.
    _GAIT_HIP_AMP = 0.45  # rad — hip pitch swing
    _GAIT_KNEE_AMP = 0.7  # rad — knee flex during swing
    _GAIT_ANKLE_AMP = 0.2  # rad
    _GAIT_SHOULDER_AMP = 0.4  # rad — arm swing counter-phased to opposite leg
    _GAIT_ELBOW_HOLD = 0.4  # rad — static elbow bend during walk
    _GAIT_CADENCE_HZ = 1.6  # full cycles per second at max walking speed
    _GAIT_MIN_PHASE_RATE = 0.5  # fraction of cadence that fires on turn-in-place

    def _apply_locomotion_stub_unlocked(self) -> None:
        """Drive the freejoint + procedural leg / arm gait.

        The freejoint's translation + yaw advance at the commanded
        linear / angular speed (mocap-style) while leg + arm
        actuators follow a simple sinusoidal walk cycle so the robot
        visibly steps. A real ONNX policy replaces both when attached
        via :meth:`set_walking_policy`.
        """
        now = time.monotonic()
        active = now < self._manual_deadline
        lin = float(self._cmd_vel[0])
        ang = float(self._cmd_vel[1])

        if not active or (lin == 0.0 and ang == 0.0):
            # Reset legs + arms toward home so the humanoid clearly
            # returns to a standing pose when it stops.
            self._relax_gait_toward_home_unlocked()
            if not active:
                self._cmd_vel[:] = 0.0
            return

        dt = self._tick_interval
        yaw = self._root_yaw_unlocked()

        # Advance root pose along current facing.
        new_x = self._data.qpos[self._root_qpos_adr] + lin * math.cos(yaw) * dt
        new_y = self._data.qpos[self._root_qpos_adr + 1] + lin * math.sin(yaw) * dt
        self._data.qpos[self._root_qpos_adr] = new_x
        self._data.qpos[self._root_qpos_adr + 1] = new_y
        self._set_root_yaw_unlocked(yaw + ang * dt)
        # Damp root velocity so the physics solver doesn't fight the override.
        self._data.qvel[0:6] = 0.0

        # Advance gait phase — linearly with forward speed, but keep a
        # minimum rate during in-place turns so legs still step.
        speed_norm = max(abs(lin) / _WALK_SPEED_MPS, self._GAIT_MIN_PHASE_RATE if ang != 0 else 0.0)
        phase_rate = self._GAIT_CADENCE_HZ * 2 * math.pi * speed_norm
        self._gait_phase = (self._gait_phase + phase_rate * dt) % (2 * math.pi)

        # Direction flip for walking backward — reverses the swing leg.
        direction = 1.0 if lin >= 0 else -1.0

        phase_left = self._gait_phase
        phase_right = self._gait_phase + math.pi

        self._apply_leg_gait_unlocked(phase_left, direction, side="left")
        self._apply_leg_gait_unlocked(phase_right, direction, side="right")
        # Arms counter-swing: right arm with left leg and vice-versa.
        self._apply_arm_gait_unlocked(phase_right, direction, side="left")
        self._apply_arm_gait_unlocked(phase_left, direction, side="right")

    def _home_ctrl_at(self, aid: int) -> float:
        if self._home_ctrl is not None and aid < self._home_ctrl.size:
            return float(self._home_ctrl[aid])
        return 0.0

    def _apply_leg_gait_unlocked(self, phase: float, direction: float, *, side: str) -> None:
        hip = self._gait_actuators.get(f"{side}_hip")
        knee = self._gait_actuators.get(f"{side}_knee")
        ankle = self._gait_actuators.get(f"{side}_ankle")
        if hip is not None:
            target = self._home_ctrl_at(hip) + direction * self._GAIT_HIP_AMP * math.sin(phase)
            self._data.ctrl[hip] = self._clamp_ctrl(hip, target)
        if knee is not None:
            # Knee flexes during swing (negative cosine lobe).
            flex = max(0.0, -math.cos(phase))
            target = self._home_ctrl_at(knee) + self._GAIT_KNEE_AMP * flex
            self._data.ctrl[knee] = self._clamp_ctrl(knee, target)
        if ankle is not None:
            target = self._home_ctrl_at(ankle) - self._GAIT_ANKLE_AMP * math.sin(phase)
            self._data.ctrl[ankle] = self._clamp_ctrl(ankle, target)

    def _apply_arm_gait_unlocked(self, phase: float, direction: float, *, side: str) -> None:
        shoulder = self._gait_actuators.get(f"{side}_shoulder")
        elbow = self._gait_actuators.get(f"{side}_elbow")
        if shoulder is not None:
            target = self._home_ctrl_at(shoulder) + direction * self._GAIT_SHOULDER_AMP * math.sin(phase)
            self._data.ctrl[shoulder] = self._clamp_ctrl(shoulder, target)
        if elbow is not None:
            self._data.ctrl[elbow] = self._clamp_ctrl(elbow, self._home_ctrl_at(elbow) + self._GAIT_ELBOW_HOLD)

    def _clamp_ctrl(self, aid: int, value: float) -> float:
        lo, hi = self._model.actuator_ctrlrange[aid]
        return float(np.clip(value, lo, hi))

    def _relax_gait_toward_home_unlocked(self) -> None:
        """Interpolate gait-driven actuators back toward home so the
        legs / arms aren't frozen mid-swing when the robot stops."""
        if self._home_ctrl is None or self._home_ctrl.size != self._data.ctrl.size:
            return
        # Low-pass toward home so the transition is smooth, not snappy.
        alpha = 0.15
        for aid in self._gait_actuators.values():
            self._data.ctrl[aid] = (1 - alpha) * float(self._data.ctrl[aid]) + alpha * float(self._home_ctrl[aid])

    def _apply_walking_policy_unlocked(self) -> None:
        if self._walking_policy is None:
            return
        try:
            obs = self._build_observation_unlocked()
            command = np.array([self._cmd_vel[0], 0.0, self._cmd_vel[1]], dtype=np.float32)
            action = self._walking_policy.step(obs, command)
            for i, a in enumerate(action[: self._model.nu]):
                lo, hi = self._model.actuator_ctrlrange[i]
                self._data.ctrl[i] = float(np.clip(a, lo, hi))
        except Exception:
            log.exception("walking policy step failed; falling back to home pose")
            if self._home_ctrl is not None and self._home_ctrl.size == self._data.ctrl.size:
                self._data.ctrl[:] = self._home_ctrl

    def _build_observation_unlocked(self) -> np.ndarray:
        # Minimal, policy-agnostic observation: joint qpos + qvel + root
        # orientation. A specific policy implementation can remap /
        # normalise as needed via its own wrapper.
        return np.concatenate(
            [
                np.asarray(self._data.qpos, dtype=np.float32),
                np.asarray(self._data.qvel, dtype=np.float32),
            ]
        )

    # ----- main-thread render pump --------------------------------------

    def pump_render(self) -> None:
        if not self._render or self._viewer is None:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            with self._lock:
                if self._viewer.is_running():
                    self._viewer.sync()
        except Exception:
            log.exception("humanoid viewer sync failed")

    pump_events = pump_render

    # ----- goal progression ---------------------------------------------

    def _update_active_goal_unlocked(self) -> None:
        goal = self._active_goal
        if goal is None:
            return

        if goal.should_cancel():
            self._cmd_vel[:] = 0.0
            self._manual_deadline = 0.0
            # Snap gait actuators straight back to home so the legs stop
            # mid-swing instead of drifting through the relax phase.
            if self._home_ctrl is not None and self._home_ctrl.size == self._data.ctrl.size:
                for aid in self._gait_actuators.values():
                    self._data.ctrl[aid] = float(self._home_ctrl[aid])
            goal.mark_cancelled()
            self._active_goal = None
            self._active_kind = None
            self._active_target = None
            self._active_target_heading = None
            self._active_command = None
            return

        if self._active_kind == "walk":
            self._progress_walk_unlocked(goal)
        elif self._active_kind == "turn":
            self._progress_turn_unlocked(goal)
        elif self._active_kind == "stand":
            self._progress_stand_unlocked(goal)

    def _progress_walk_unlocked(self, goal: GoalHandle) -> None:
        target = self._active_target
        if target is None:
            goal.fail("walk had no target")
            self._active_goal = None
            return
        cur = self._root_xy_unlocked()
        delta = target - cur
        remaining = float(np.linalg.norm(delta))
        goal.set_feedback({"remaining_m": round(remaining, 3), "pos": cur.round(3).tolist()})
        if remaining < _POSITION_TOL:
            self._cmd_vel[:] = 0.0
            goal.succeed({"pos": cur.round(3).tolist(), "remaining_m": 0.0})
            self._active_goal = None
            self._active_kind = None
            self._active_target = None
            return
        # Steer by keeping the commanded twist forward; heading was set
        # at goal start, so no steering adjustment here.
        self._manual_deadline = time.monotonic() + 0.2

    def _progress_turn_unlocked(self, goal: GoalHandle) -> None:
        target_yaw = self._active_target_heading
        if target_yaw is None:
            goal.fail("turn had no heading")
            self._active_goal = None
            return
        cur = self._root_yaw_unlocked()
        diff = math.atan2(math.sin(target_yaw - cur), math.cos(target_yaw - cur))
        goal.set_feedback({"remaining_rad": round(diff, 3)})
        if abs(diff) < _HEADING_TOL:
            self._cmd_vel[:] = 0.0
            goal.succeed({"heading_rad": round(cur, 3), "heading_deg": round(math.degrees(cur), 1)})
            self._active_goal = None
            self._active_kind = None
            self._active_target_heading = None
            return
        ang = _TURN_SPEED_RAD * (1.0 if diff > 0 else -1.0)
        self._cmd_vel[:] = [0.0, ang]
        self._manual_deadline = time.monotonic() + 0.2

    def _progress_stand_unlocked(self, goal: GoalHandle) -> None:
        # Reset actuators to the model's home keyframe — for the Unitree
        # humanoids this is a balanced standing pose.
        if self._home_ctrl is not None and self._home_ctrl.size == self._data.ctrl.size:
            self._data.ctrl[:] = self._home_ctrl
        vz = float(self._data.qvel[2])
        if abs(vz) < 0.05:
            goal.succeed({"standing": True, "pelvis_z": round(float(self._data.xpos[self._pelvis_bid][2]), 3)})
            self._active_goal = None
            self._active_kind = None
        else:
            goal.set_feedback({"vz": round(vz, 3)})

    # ----- SimEnvironment protocol --------------------------------------

    def reset(self) -> None:
        with self._lock:
            self._data.qpos[:] = self._initial_qpos
            self._data.qvel[:] = self._initial_qvel
            self._data.ctrl[:] = self._initial_ctrl
            mujoco.mj_forward(self._model, self._data)
            self._cmd_vel[:] = 0.0
            self._active_goal = None
            self._active_kind = None
            self._active_target = None
            self._active_target_heading = None
            if self._walking_policy is not None:
                with contextlib.suppress(Exception):
                    self._walking_policy.reset()

    def step(self, dt: float) -> None:  # pragma: no cover
        del dt
        self.tick_physics()

    def get_world_state(self) -> dict[str, Any]:
        with self._lock:
            pos = self._root_xy_unlocked()
            yaw = self._root_yaw_unlocked()
            pelvis_z = float(self._data.xpos[self._pelvis_bid][2])
            return {
                "robot": "humanoid",
                "pose": {
                    "x": round(float(pos[0]), 3),
                    "y": round(float(pos[1]), 3),
                    "heading_deg": round(math.degrees(yaw) % 360, 1),
                    "pelvis_z": round(pelvis_z, 3),
                    "standing": pelvis_z > 0.75,
                },
                "policy": "onnx" if self._walking_policy is not None else "stub",
                "busy": self._active_goal is not None,
            }

    def render(self) -> None:
        pass

    def apply_action(self, action: str, **kwargs: Any) -> GoalHandle:
        dispatcher = getattr(self, f"_action_{action}", None)
        if dispatcher is None:
            h = GoalHandle()
            h.fail(f"unknown action {action!r}")
            return h
        return dispatcher(**kwargs)

    def close(self) -> None:
        self._phys_stop.set()
        self._phys_thread.join(timeout=1.5)
        if self._viewer is not None:
            with contextlib.suppress(Exception):
                self._viewer.close()
            self._viewer = None
        if self._offscreen_renderer is not None:
            with contextlib.suppress(Exception):
                self._offscreen_renderer.close()
            self._offscreen_renderer = None

    # ----- policy hookup -------------------------------------------------

    def set_walking_policy(self, policy: WalkingPolicy | None) -> None:
        """Register / unregister a walking policy. Passing ``None``
        reverts to the mocap-style stub."""
        with self._lock:
            if policy is not None:
                with contextlib.suppress(Exception):
                    policy.reset()
            self._walking_policy = policy
            log.info("humanoid walking policy: %s", "attached" if policy else "detached")

    # ----- ROS2 adapter hooks -------------------------------------------

    def get_pose2d(self) -> tuple[float, float, float]:
        """Root (base_link-equivalent) pose in the map frame."""
        with self._lock:
            pos = self._root_xy_unlocked()
            return float(pos[0]), float(pos[1]), self._root_yaw_unlocked()

    def get_ee_pose(self) -> tuple[float, float, float]:
        """Head position — useful TF target for rviz gaze/camera rigs."""
        with self._lock:
            head = self._data.xpos[self._head_bid]
            return float(head[0]), float(head[1]), float(head[2])

    def apply_velocity(self, linear: float, angular: float) -> None:
        """Command base-twist, Nav2-style. The mocap stub consumes these
        directly; an ONNX policy receives them as the ``command`` input.
        Any active skill goal is terminally cancelled so callers see a
        consistent CANCELLED state instead of an orphaned RUNNING handle.
        """
        with self._lock:
            if self._active_goal is not None:
                self._active_goal.mark_cancelled()
                self._active_goal = None
                self._active_kind = None
                self._active_target = None
                self._active_target_heading = None
            self._cmd_vel[:] = [float(linear), float(angular)]
            self._manual_deadline = time.monotonic() + 0.5

    def get_camera_frame(self, width: int = 320, height: int = 240, camera_id: int = -1) -> np.ndarray | None:
        """Render the scene from the head-mounted camera."""
        with self._lock:
            if self._offscreen_renderer is None or self._offscreen_size != (width, height):
                if self._offscreen_renderer is not None:
                    with contextlib.suppress(Exception):
                        self._offscreen_renderer.close()
                try:
                    self._offscreen_renderer = mujoco.Renderer(self._model, width=width, height=height)
                    self._offscreen_size = (width, height)
                except Exception:
                    log.debug(
                        "humanoid offscreen renderer init failed (set MUJOCO_GL=egl on headless hosts)",
                        exc_info=True,
                    )
                    self._offscreen_renderer = None
                    self._offscreen_size = None
                    return None
            try:
                # Prefer the head camera when the caller doesn't specify.
                if camera_id == -1:
                    try:
                        camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
                    except Exception:
                        camera_id = -1
                self._offscreen_renderer.update_scene(self._data, camera=camera_id)
                return np.ascontiguousarray(self._offscreen_renderer.render())
            except Exception:
                log.debug("humanoid offscreen render failed", exc_info=True)
                return None

    # ----- skill dispatchers --------------------------------------------

    def _action_walk_forward(self, distance: float = 1.0, speed: float | None = None) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            yaw = self._root_yaw_unlocked()
            cur = self._root_xy_unlocked()
            target = cur + float(distance) * np.array([math.cos(yaw), math.sin(yaw)])
            lin = float(speed) if speed is not None else _WALK_SPEED_MPS
            lin = max(-_WALK_SPEED_MPS, min(_WALK_SPEED_MPS, lin))
            if distance < 0:
                lin = -abs(lin)
            self._cmd_vel[:] = [lin, 0.0]
            self._manual_deadline = time.monotonic() + 0.5
            handle.status = GoalStatus.RUNNING
            self._active_goal = handle
            self._active_kind = "walk"
            self._active_target = target
        return handle

    def _action_walk_backward(self, distance: float = 1.0) -> GoalHandle:
        return self._action_walk_forward(distance=-abs(distance))

    def _action_turn(self, degrees: float = 90.0) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            yaw = self._root_yaw_unlocked()
            target_yaw = yaw + math.radians(float(degrees))
            target_yaw = math.atan2(math.sin(target_yaw), math.cos(target_yaw))
            handle.status = GoalStatus.RUNNING
            self._active_goal = handle
            self._active_kind = "turn"
            self._active_target_heading = target_yaw
        return handle

    def _action_turn_left(self, degrees: float = 90.0) -> GoalHandle:
        return self._action_turn(degrees=abs(degrees))

    def _action_turn_right(self, degrees: float = 90.0) -> GoalHandle:
        return self._action_turn(degrees=-abs(degrees))

    def _action_stand(self) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            self._cmd_vel[:] = 0.0
            handle.status = GoalStatus.RUNNING
            self._active_goal = handle
            self._active_kind = "stand"
        return handle

    def _action_sit(self) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            if not self._using_bundled_actuators:
                handle.fail("sit is only defined for the bundled humanoid — external models need a locomotion policy")
                return handle
            self._cmd_vel[:] = 0.0
            handle.status = GoalStatus.RUNNING
            self._active_goal = handle
            self._active_kind = "sit"
        return handle

    def _action_wave(self) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            if not self._using_bundled_actuators:
                handle.fail(
                    "wave is only defined for the bundled humanoid — external models need a joint-space controller"
                )
                return handle
            handle.status = GoalStatus.RUNNING
            self._active_goal = handle
            self._active_kind = "wave"
        return handle

    def _action_get_pose(self) -> GoalHandle:
        h = GoalHandle()
        h.succeed(self.get_world_state()["pose"])
        return h


__all__ = ["MujocoHumanoidEnvironment", "WalkingPolicy"]
