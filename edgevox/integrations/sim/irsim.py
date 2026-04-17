"""IR-SIM adapter — visual 2D robot simulator for EdgeVox agents.

Wraps `hanruihua/ir-sim <https://github.com/hanruihua/ir-sim>`_ as a
:class:`~edgevox.agents.SimEnvironment` so any EdgeVox agent that
already works against ``ToyWorld`` can drive a real visible robot in
matplotlib without changing a line of agent code.

Design notes (important for the threading model):

- **Physics is thread-safe** but **rendering is main-thread only**
  (matplotlib's Tk backend refuses off-main-thread ``plt.pause``).
  ``tick_physics()`` advances ``env.step()`` and tracks goal progress
  under a lock — any thread may call it. ``pump_render()`` must only
  run on the main thread.
- A background **physics thread** ticks physics at ``tick_interval``
  whether or not a goal is active. This means the robot animates even
  while the agent is idle.
- The main thread runs a **render pump** via
  :class:`edgevox.agents.bus.MainThreadScheduler`. On every idle tick
  it calls ``pump_render()`` which redraws matplotlib.
- Skill cancellation flows through ``GoalHandle.cancel()`` — the
  physics thread observes it on its next tick and stops the robot.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from edgevox.agents.skills import GoalHandle, GoalStatus

try:
    import irsim
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "IR-SIM is not installed. Run `pip install ir-sim` or "
        "`pip install 'edgevox[sim]'` to enable the IrSimEnvironment adapter."
    ) from e

log = logging.getLogger(__name__)

_BUNDLED_WORLDS_DIR = Path(__file__).parent / "worlds"


class IrSimEnvironment:
    """IR-SIM-backed :class:`SimEnvironment` implementation.

    Args:
        world_yaml: path to an IR-SIM world YAML (see
            ``edgevox/integrations/sim/worlds/`` for bundled examples).
            ``None`` selects the bundled ``edgevox_apartment.yaml``.
        render: whether to open a matplotlib window. Set ``False`` for
            headless tests.
        tick_interval: real-world seconds between physics ticks.
            Controls how fast the robot appears to move.
        waypoints: override the default named room positions.
    """

    def __init__(
        self,
        world_yaml: str | Path | None = None,
        *,
        render: bool = True,
        tick_interval: float = 0.05,
        waypoints: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if world_yaml is None:
            world_yaml = _BUNDLED_WORLDS_DIR / "edgevox_apartment.yaml"
        world_yaml = Path(world_yaml).expanduser()
        if not world_yaml.exists():
            raise FileNotFoundError(f"IR-SIM world file not found: {world_yaml}")

        self._world_yaml = world_yaml
        self._render = render
        self._tick_interval = tick_interval
        self._env = irsim.make(str(world_yaml), display=render)
        self._lock = threading.RLock()
        self._active_goal: GoalHandle | None = None
        self._active_target: tuple[float, float] | None = None
        self._battery_pct = 95.0
        self._waypoints: dict[str, tuple[float, float]] = dict(waypoints or {})
        if not self._waypoints:
            self._waypoints = {
                "kitchen": (8.0, 2.0),
                "living_room": (2.0, 2.0),
                "bedroom": (8.0, 8.0),
                "office": (2.0, 8.0),
                "center": (5.0, 5.0),
            }

        # Background physics thread — advances the sim even when the
        # agent is idle so the robot animates continuously. It does NOT
        # render; rendering is main-thread-only (see pump_render).
        self._phys_stop = threading.Event()
        # Manual-velocity override: when set, ``tick_physics`` bypasses
        # the behavior-driven ``_env.step()`` and advances the robot
        # with the given velocity directly. The override auto-expires
        # after ``_MANUAL_VEL_HOLD_S`` seconds so a dropped cmd_vel
        # stream doesn't leave the robot running forever.
        self._manual_vel: Any = None
        self._manual_vel_deadline = 0.0
        self._phys_thread = threading.Thread(target=self._physics_loop, name="irsim-physics", daemon=True)
        self._phys_thread.start()

        if render:
            try:
                import matplotlib.pyplot as plt

                plt.ion()
                self._env.render(0.001)  # initial draw on main thread
            except Exception:
                log.exception("Initial IR-SIM render failed")

    # ----- thread-safe physics -----

    def _physics_loop(self) -> None:
        """Advance physics at a fixed rate. Runs on a daemon thread."""
        while not self._phys_stop.is_set():
            try:
                self.tick_physics()
            except Exception:
                log.exception("IR-SIM physics tick failed")
            time.sleep(self._tick_interval)

    _MANUAL_VEL_HOLD_S = 0.5

    def tick_physics(self) -> None:
        """Advance the sim one physics step and update any active goal.
        Thread-safe; called by the physics loop and tests."""
        with self._lock:
            try:
                if self._manual_vel is not None and time.monotonic() < self._manual_vel_deadline:
                    # cmd_vel-driven manual override — advance the robot
                    # directly with the commanded velocity, bypassing
                    # behavior. Other world objects aren't stepped, but
                    # the apartment sim has no moving obstacles so that's
                    # a non-issue.
                    self._env.robot.stop_flag = False
                    self._env.robot.arrive_flag = False
                    self._env.robot.step(self._manual_vel, sensor_step=True)
                else:
                    if self._manual_vel is not None:
                        # stale cmd_vel — fall back to zero so behavior
                        # can take over cleanly.
                        with contextlib.suppress(Exception):
                            self._env.robot.set_velocity([0.0, 0.0])
                        self._manual_vel = None
                    self._env.step()
            except Exception:
                log.exception("IR-SIM env.step failed")
                return

            goal = self._active_goal
            if goal is None:
                return

            if goal.should_cancel():
                with contextlib.suppress(Exception):
                    self._env.robot.set_velocity([0.0, 0.0])
                self._env.robot.stop_flag = True
                goal.mark_cancelled()
                self._active_goal = None
                return

            if self._env.robot.arrive:
                pose = self._robot_pose_unlocked()
                label = self._target_label_unlocked()
                goal.succeed(
                    {
                        "arrived_at": label,
                        "pose": (round(pose[0], 2), round(pose[1], 2)),
                    }
                )
                self._active_goal = None
                self._active_target = None
                return

            pose = self._robot_pose_unlocked()
            tx, ty = self._active_target or (0.0, 0.0)
            remaining = math.hypot(tx - pose[0], ty - pose[1])
            goal.set_feedback(
                {
                    "pose": (round(pose[0], 2), round(pose[1], 2)),
                    "remaining": round(remaining, 2),
                }
            )
            self._battery_pct = max(0.0, self._battery_pct - 0.0005 * remaining)

    # ----- main-thread rendering -----

    def pump_render(self) -> None:
        """Redraw matplotlib on the main thread. Safe no-op if
        ``render=False`` or if called from a non-main thread.

        Callers should invoke this from the main thread only (e.g. via
        a :class:`~edgevox.agents.bus.MainThreadScheduler`).
        """
        if not self._render:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            with self._lock:
                self._env.render(0.001)
        except Exception:
            log.exception("IR-SIM render failed")

    # Back-compat alias: pump_events is what older callers checked for.
    pump_events = pump_render

    # Back-compat alias: tick_main used to do both — now it's physics only
    # since render is main-thread gated. Safe to call from any thread.
    tick_main = tick_physics

    def _robot_pose_unlocked(self) -> tuple[float, float, float]:
        state = self._env.robot.state
        return (
            float(state[0, 0]),
            float(state[1, 0]),
            float(state[2, 0] if state.shape[0] >= 3 else 0.0),
        )

    def _target_label_unlocked(self) -> str:
        if self._active_target is None:
            return "<none>"
        tx, ty = self._active_target
        for name, (wx, wy) in self._waypoints.items():
            if math.hypot(tx - wx, ty - wy) < 0.3:
                return name
        return f"({tx:.1f},{ty:.1f})"

    # ----- SimEnvironment protocol -----

    def reset(self) -> None:
        with self._lock:
            self._env.reset()
            self._active_goal = None
            self._active_target = None
            self._battery_pct = 95.0

    def step(self, dt: float) -> None:  # pragma: no cover
        del dt
        self.tick_physics()

    def get_world_state(self) -> dict[str, Any]:
        with self._lock:
            pose = self._robot_pose_unlocked()
            return {
                "robot": {
                    "x": round(pose[0], 3),
                    "y": round(pose[1], 3),
                    "heading_deg": round(math.degrees(pose[2]) % 360, 1),
                    "battery_pct": round(self._battery_pct, 1),
                    "moving": self._active_goal is not None,
                },
                "waypoints": dict(self._waypoints),
                "goal": (self._active_target if self._active_target is not None else None),
                "arrived": bool(self._env.robot.arrive),
            }

    def render(self) -> None:
        """No-op; rendering happens via :meth:`pump_render` on main."""
        pass

    def apply_action(self, action: str, **kwargs: Any) -> GoalHandle:
        dispatcher = getattr(self, f"_action_{action}", None)
        if dispatcher is None:
            handle = GoalHandle()
            handle.fail(f"unknown action {action!r}")
            return handle
        return dispatcher(**kwargs)

    def room_names(self) -> list[str]:
        with self._lock:
            return sorted(self._waypoints)

    # ----- ROS2 adapter hooks — sensor / velocity surfaces -----
    # These are optional capability hooks consumed by
    # :class:`edgevox.integrations.ros2_robot.RobotROS2Adapter`. They
    # let the ROS2 layer publish ``LaserScan`` / ``PoseStamped`` and
    # accept ``Twist`` ``cmd_vel`` commands without the sim adapter
    # itself knowing about ROS2.

    def get_lidar_scan(self) -> dict[str, Any] | None:
        """Return the current 2D lidar scan in a ROS2-compatible dict
        (keys: ``angle_min``, ``angle_max``, ``angle_increment``,
        ``time_increment``, ``scan_time``, ``range_min``, ``range_max``,
        ``ranges``). Returns ``None`` if the robot has no lidar."""
        with self._lock:
            try:
                return self._env.robot.get_lidar_scan()
            except Exception:
                return None

    def apply_velocity(self, linear: float, angular: float) -> None:
        """Command instantaneous linear (m/s) + angular (rad/s) velocity.

        Cancels any active navigation goal so external ``cmd_vel``
        callers have exclusive control until they stop sending. The
        override lasts ``_MANUAL_VEL_HOLD_S`` seconds after the last
        call — drop the cmd_vel stream and the robot halts, which is
        the standard ROS2 safety behavior.
        """
        with self._lock:
            if self._active_goal is not None:
                self._active_goal.cancel()
                self._active_goal = None
                self._active_target = None
            self._manual_vel = np.array([[float(linear)], [float(angular)]])
            self._manual_vel_deadline = time.monotonic() + self._MANUAL_VEL_HOLD_S

    def get_pose2d(self) -> tuple[float, float, float]:
        """Return ``(x, y, theta_rad)`` pose in the world frame."""
        with self._lock:
            return self._robot_pose_unlocked()

    def close(self) -> None:
        self._phys_stop.set()
        self._phys_thread.join(timeout=1.0)
        with contextlib.suppress(Exception):
            self._env.end(0)

    # ----- action implementations -----

    def _action_get_pose(self) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            pose = self._robot_pose_unlocked()
        handle.succeed(
            {
                "x": round(pose[0], 2),
                "y": round(pose[1], 2),
                "heading_deg": round(math.degrees(pose[2]) % 360, 1),
            }
        )
        return handle

    def _action_battery_level(self) -> GoalHandle:
        handle = GoalHandle()
        handle.succeed({"battery_pct": round(self._battery_pct, 1)})
        return handle

    def _action_list_rooms(self) -> GoalHandle:
        handle = GoalHandle()
        handle.succeed(self.room_names())
        return handle

    def _action_navigate_to(
        self,
        room: str | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            if room is not None:
                if room not in self._waypoints:
                    handle.fail(f"unknown room {room!r}. Known: {sorted(self._waypoints)}")
                    return handle
                tx, ty = self._waypoints[room]
            elif x is not None and y is not None:
                tx, ty = float(x), float(y)
            else:
                handle.fail("navigate_to needs room= or x=/y=")
                return handle

            pose = self._robot_pose_unlocked()
            theta = math.atan2(ty - pose[1], tx - pose[0])
            try:
                self._env.robot.stop_flag = False
                self._env.robot.arrive_flag = False
                self._env.robot.set_goal([tx, ty, theta])
            except Exception as e:
                log.exception("IR-SIM set_goal failed")
                handle.fail(f"set_goal failed: {e}")
                return handle

            handle.status = GoalStatus.RUNNING
            self._active_goal = handle
            self._active_target = (tx, ty)
        return handle
