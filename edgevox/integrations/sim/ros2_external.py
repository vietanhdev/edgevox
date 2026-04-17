"""Tier 3 adapter — drive an *external* ROS2 simulator / real robot.

EdgeVox's Tier 1 (IR-SIM) and Tier 2 (MuJoCo) adapters run the physics
in-process. Tier 3 flips that: the physics lives in a separate ROS2
process — Gazebo Harmonic, Isaac Sim via ROS2 bridge, a real Unitree
Go2, anything that speaks standard ROS2 messages — and this adapter
presents the same :class:`~edgevox.agents.sim.SimEnvironment` protocol
to the agent layer by subscribing to the robot's state topics and
publishing velocity / goal commands.

Topic contract (all paths are relative to ``namespace``):

| Direction | Topic               | Type                          | Optional |
|-----------|---------------------|-------------------------------|----------|
| sub       | ``odom``            | ``nav_msgs/Odometry``         | required |
| sub       | ``scan``            | ``sensor_msgs/LaserScan``     | optional |
| sub       | ``camera/image_raw``| ``sensor_msgs/Image``         | optional |
| pub       | ``cmd_vel``         | ``geometry_msgs/Twist``       | required |
| pub       | ``goal_pose``       | ``geometry_msgs/PoseStamped`` | required |

The adapter itself doesn't care whether the other end is Gazebo, Isaac,
or a real Husky — any node that publishes ``odom`` and accepts
``cmd_vel`` will work. Skills (``navigate_to``, ``stop``,
``apply_velocity``) are implemented on top of this contract.

A brief note on graduation: because the adapter's *interface* is
identical to the in-process sims, an agent written against
``ToyWorld`` / ``IrSimEnvironment`` / ``MujocoHumanoidEnvironment``
runs unchanged against a real robot — swap the ``deps`` and ship.
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
import time
from typing import Any

import numpy as np

from edgevox.agents.skills import GoalHandle, GoalStatus

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import Image, LaserScan

    from edgevox.integrations.ros2_qos import reliable_qos, sensor_qos

    ROS2_AVAILABLE = True
except ImportError:  # pragma: no cover — import-guarded module
    ROS2_AVAILABLE = False

log = logging.getLogger(__name__)


_ODOM_STALE_S = 2.0


class ExternalROS2Environment:
    """SimEnvironment that talks to a separate ROS2 process.

    Args:
        namespace: ROS2 namespace prefixing every topic (default ``""``
            so topics land at ``/odom`` etc.). Set to ``"/robot1"`` for
            multi-robot scenarios.
        node_name: name of the rclpy node created by this adapter.
        arrival_tol_m: distance to ``navigate_to`` target considered
            arrived.
        arrival_tol_rad: heading tolerance.
        vel_linear_max: max linear speed commanded by skills.
        vel_angular_max: max angular speed commanded by skills.
        goal_timeout_s: default timeout for :meth:`_action_navigate_to`
            before the goal fails.
    """

    def __init__(
        self,
        *,
        namespace: str = "",
        node_name: str = "edgevox_external_sim",
        arrival_tol_m: float = 0.15,
        arrival_tol_rad: float = 0.15,
        vel_linear_max: float = 0.5,
        vel_angular_max: float = 1.0,
        goal_timeout_s: float = 60.0,
    ) -> None:
        if not ROS2_AVAILABLE:
            raise RuntimeError("rclpy not available — ExternalROS2Environment needs a sourced ROS2 workspace")

        self._namespace = namespace.rstrip("/")
        self._node_name = node_name
        self._arrival_tol_m = arrival_tol_m
        self._arrival_tol_rad = arrival_tol_rad
        self._vel_linear_max = vel_linear_max
        self._vel_angular_max = vel_angular_max
        self._goal_timeout_s = goal_timeout_s

        # rclpy init is idempotent — only init if no context exists yet.
        if not rclpy.ok():
            rclpy.init()

        self._node: Node = rclpy.create_node(node_name, namespace=namespace or "/")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        # ---- state cache ------------------------------------------------
        self._lock = threading.RLock()
        self._pose_xy = np.zeros(2)
        self._pose_yaw = 0.0
        self._pose_stamp: float = 0.0
        self._twist = np.zeros(2)
        self._last_scan: Any = None
        self._last_image: Any = None

        # ---- goal state -------------------------------------------------
        self._active_goal: GoalHandle | None = None
        self._active_kind: str | None = None
        self._active_target_xy: np.ndarray | None = None
        self._active_target_yaw: float | None = None
        self._active_deadline = 0.0

        # ---- publishers / subscribers -----------------------------------
        self._cmd_vel_pub = self._node.create_publisher(Twist, "cmd_vel", reliable_qos())
        self._goal_pose_pub = self._node.create_publisher(PoseStamped, "goal_pose", reliable_qos())

        self._odom_sub = self._node.create_subscription(Odometry, "odom", self._on_odom, reliable_qos())
        self._scan_sub = self._node.create_subscription(LaserScan, "scan", self._on_scan, sensor_qos())
        self._image_sub = self._node.create_subscription(Image, "camera/image_raw", self._on_image, sensor_qos())

        # ---- background spin + goal driver ------------------------------
        self._stop = threading.Event()
        self._spin_thread = threading.Thread(target=self._spin_loop, name="ros2-external-spin", daemon=True)
        self._spin_thread.start()
        self._goal_thread = threading.Thread(target=self._goal_loop, name="ros2-external-goal", daemon=True)
        self._goal_thread.start()

        log.info(
            "ExternalROS2Environment up (namespace=%r). Topics: %s/cmd_vel, %s/odom, %s/scan, %s/camera/image_raw",
            namespace,
            namespace,
            namespace,
            namespace,
            namespace,
        )

    # ----- background loops ----------------------------------------------

    def _spin_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except Exception:
                log.debug("executor spin_once failed", exc_info=True)

    def _goal_loop(self) -> None:
        while not self._stop.wait(0.05):
            with self._lock:
                self._update_goal_unlocked()

    # ----- callbacks ------------------------------------------------------

    def _on_odom(self, msg: Any) -> None:
        with self._lock:
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            # quaternion (w,x,y,z) -> z-axis yaw
            w, x, y, z = float(o.w), float(o.x), float(o.y), float(o.z)
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            self._pose_xy[:] = [float(p.x), float(p.y)]
            self._pose_yaw = yaw
            self._pose_stamp = time.monotonic()
            t = msg.twist.twist
            self._twist[:] = [float(t.linear.x), float(t.angular.z)]

    def _on_scan(self, msg: Any) -> None:
        with self._lock:
            self._last_scan = msg

    def _on_image(self, msg: Any) -> None:
        with self._lock:
            self._last_image = msg

    # ----- goal progression ----------------------------------------------

    def _update_goal_unlocked(self) -> None:
        goal = self._active_goal
        if goal is None:
            return

        if goal.should_cancel():
            self._stop_robot_unlocked()
            goal.mark_cancelled()
            self._active_goal = None
            self._active_kind = None
            return

        if time.monotonic() > self._active_deadline:
            self._stop_robot_unlocked()
            goal.fail("navigate_to: timed out before reaching target")
            self._active_goal = None
            self._active_kind = None
            return

        if time.monotonic() - self._pose_stamp > _ODOM_STALE_S:
            # Don't control without fresh odometry — publish zero twist.
            self._publish_twist_unlocked(0.0, 0.0)
            goal.set_feedback({"warn": "stale odometry"})
            return

        if self._active_kind == "navigate_to" and self._active_target_xy is not None:
            target = self._active_target_xy
            delta = target - self._pose_xy
            dist = float(np.linalg.norm(delta))
            if dist < self._arrival_tol_m:
                self._stop_robot_unlocked()
                goal.succeed({"pos": self._pose_xy.round(3).tolist(), "remaining_m": 0.0})
                self._active_goal = None
                self._active_kind = None
                self._active_target_xy = None
                return
            # Bang-bang steering: cap linear velocity by forward angle error.
            target_yaw = math.atan2(delta[1], delta[0])
            yaw_err = math.atan2(
                math.sin(target_yaw - self._pose_yaw),
                math.cos(target_yaw - self._pose_yaw),
            )
            ang = float(np.clip(2.0 * yaw_err, -self._vel_angular_max, self._vel_angular_max))
            lin = float(np.clip(self._vel_linear_max * max(0.0, math.cos(yaw_err)), 0.0, self._vel_linear_max))
            if dist < 0.5:
                lin *= max(0.2, dist / 0.5)
            self._publish_twist_unlocked(lin, ang)
            goal.set_feedback(
                {
                    "pos": self._pose_xy.round(3).tolist(),
                    "remaining_m": round(dist, 3),
                    "yaw_err_deg": round(math.degrees(yaw_err), 1),
                }
            )

    # ----- helpers --------------------------------------------------------

    def _publish_twist_unlocked(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self._cmd_vel_pub.publish(msg)

    def _stop_robot_unlocked(self) -> None:
        self._publish_twist_unlocked(0.0, 0.0)

    # ----- SimEnvironment protocol ---------------------------------------

    def reset(self) -> None:
        # Nothing to "reset" on an external sim — at best we stop the
        # robot. Callers managing Gazebo should reset that separately.
        with self._lock:
            self._stop_robot_unlocked()
            self._active_goal = None
            self._active_kind = None
            self._active_target_xy = None
            self._active_target_yaw = None

    def step(self, dt: float) -> None:  # pragma: no cover — ticked by the spin loop
        del dt

    def get_world_state(self) -> dict[str, Any]:
        with self._lock:
            stale = time.monotonic() - self._pose_stamp > _ODOM_STALE_S if self._pose_stamp else True
            state = {
                "robot": "external_ros2",
                "pose": {
                    "x": round(float(self._pose_xy[0]), 3),
                    "y": round(float(self._pose_xy[1]), 3),
                    "heading_deg": round(math.degrees(self._pose_yaw) % 360, 1),
                },
                "twist": {
                    "linear": round(float(self._twist[0]), 3),
                    "angular": round(float(self._twist[1]), 3),
                },
                "odom_stale": stale,
                "has_scan": self._last_scan is not None,
                "has_image": self._last_image is not None,
                "busy": self._active_goal is not None,
            }
            return state

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
        self._stop.set()
        self._spin_thread.join(timeout=1.5)
        self._goal_thread.join(timeout=1.5)
        with contextlib.suppress(Exception):
            self._stop_robot_unlocked()
        with contextlib.suppress(Exception):
            self._executor.shutdown()
        with contextlib.suppress(Exception):
            self._node.destroy_node()
        # Don't shut rclpy down — the ROS2Bridge may also be running
        # in the same process. Caller manages rclpy lifecycle.

    # ----- skill dispatchers ---------------------------------------------

    def _action_navigate_to(
        self,
        x: float | None = None,
        y: float | None = None,
        timeout_s: float | None = None,
    ) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            if x is None or y is None:
                handle.fail("navigate_to needs x= and y=")
                return handle
            target = np.array([float(x), float(y)])
            handle.status = GoalStatus.RUNNING
            self._active_goal = handle
            self._active_kind = "navigate_to"
            self._active_target_xy = target
            self._active_deadline = time.monotonic() + float(
                timeout_s if timeout_s is not None else self._goal_timeout_s
            )
            # Also publish a goal_pose for Nav2-style stacks that
            # consume the goal rather than our bang-bang driver.
            self._publish_goal_pose_unlocked(target)
        return handle

    def _publish_goal_pose_unlocked(self, target: np.ndarray) -> None:
        ps = PoseStamped()
        ps.header.stamp = self._node.get_clock().now().to_msg()
        ps.header.frame_id = "map"
        ps.pose.position.x = float(target[0])
        ps.pose.position.y = float(target[1])
        ps.pose.orientation.w = 1.0
        self._goal_pose_pub.publish(ps)

    def _action_stop(self) -> GoalHandle:
        handle = GoalHandle()
        with self._lock:
            self._stop_robot_unlocked()
            if self._active_goal is not None:
                self._active_goal.mark_cancelled()
                self._active_goal = None
                self._active_kind = None
                self._active_target_xy = None
        handle.succeed({"stopped": True})
        return handle

    def _action_get_pose(self) -> GoalHandle:
        h = GoalHandle()
        h.succeed(self.get_world_state()["pose"])
        return h

    # ----- ROS2 adapter capability hooks ---------------------------------

    def get_pose2d(self) -> tuple[float, float, float]:
        with self._lock:
            return float(self._pose_xy[0]), float(self._pose_xy[1]), float(self._pose_yaw)

    def apply_velocity(self, linear: float, angular: float) -> None:
        """Direct twist passthrough — cancels any active goal first."""
        with self._lock:
            if self._active_goal is not None:
                self._active_goal.mark_cancelled()
                self._active_goal = None
                self._active_kind = None
                self._active_target_xy = None
            self._publish_twist_unlocked(float(linear), float(angular))


__all__ = ["ExternalROS2Environment"]
