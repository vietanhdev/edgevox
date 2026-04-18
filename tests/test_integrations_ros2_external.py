"""Tier 3 external-ROS2 adapter tests.

Rather than requiring a running Gazebo/Isaac/real robot, these tests
simulate the "external" side with a second rclpy node that:
- publishes fake ``nav_msgs/Odometry`` as the robot moves;
- consumes ``geometry_msgs/Twist`` and integrates it into the fake pose;
- exposes a minimal world loop so the ``navigate_to`` bang-bang driver
  converges to a real target.

Auto-skipped without ``rclpy``.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from geometry_msgs.msg import Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402

from edgevox.agents.skills import GoalStatus  # noqa: E402
from edgevox.integrations.ros2_qos import reliable_qos  # noqa: E402
from edgevox.integrations.sim.ros2_external import ExternalROS2Environment  # noqa: E402


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5))


def _ensure_rclpy_down() -> None:
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


class _FakeRobot:
    """Minimal stand-in for an external ROS2 sim."""

    def __init__(self, namespace: str = "") -> None:
        if not rclpy.ok():
            rclpy.init()
        ns = namespace.rstrip("/") or "/"
        self.node: Node = rclpy.create_node("fake_external_robot", namespace=ns)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.lin = 0.0
        self.ang = 0.0
        self._lock = threading.RLock()
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)

        self._odom_pub = self.node.create_publisher(Odometry, "odom", reliable_qos())
        self.node.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, reliable_qos())

        self._stop = threading.Event()
        self._tick_thread = threading.Thread(target=self._tick, name="fake-robot-tick", daemon=True)
        self._tick_thread.start()
        self._spin_thread = threading.Thread(target=self._spin, name="fake-robot-spin", daemon=True)
        self._spin_thread.start()

    def _on_cmd_vel(self, msg) -> None:
        with self._lock:
            self.lin = float(msg.linear.x)
            self.ang = float(msg.angular.z)

    def _tick(self) -> None:
        dt = 0.05
        while not self._stop.wait(dt):
            with self._lock:
                self.x += self.lin * math.cos(self.yaw) * dt
                self.y += self.lin * math.sin(self.yaw) * dt
                self.yaw += self.ang * dt
                self._publish_odom_unlocked()

    def _publish_odom_unlocked(self) -> None:
        msg = Odometry()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0
        w, qx, qy, qz = _yaw_to_quat(self.yaw)
        msg.pose.pose.orientation.w = w
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.twist.twist.linear.x = self.lin
        msg.twist.twist.angular.z = self.ang
        self._odom_pub.publish(msg)

    def _spin(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self._executor.spin_once(timeout_sec=0.05)

    def shutdown(self) -> None:
        self._stop.set()
        self._tick_thread.join(timeout=1.5)
        self._spin_thread.join(timeout=1.5)
        with contextlib.suppress(Exception):
            self._executor.shutdown()
        with contextlib.suppress(Exception):
            self.node.destroy_node()


class TestExternalROS2Environment:
    @pytest.fixture
    def robot_and_env(self):
        _ensure_rclpy_down()
        robot = _FakeRobot()
        # Give the fake robot's first odom message time to land before
        # the env reads its pose.
        time.sleep(0.3)
        env = ExternalROS2Environment()
        # Wait until fresh odom has arrived in the env.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not env.get_world_state()["odom_stale"]:
                break
            time.sleep(0.1)
        yield robot, env
        env.close()
        robot.shutdown()
        _ensure_rclpy_down()

    def test_odom_is_fresh(self, robot_and_env):
        _, env = robot_and_env
        state = env.get_world_state()
        assert not state["odom_stale"], state
        assert state["robot"] == "external_ros2"

    def test_apply_velocity_moves_fake_robot(self, robot_and_env):
        robot, env = robot_and_env
        x0 = robot.x
        env.apply_velocity(0.3, 0.0)
        time.sleep(1.0)
        env.apply_velocity(0.0, 0.0)
        assert robot.x - x0 > 0.1, f"fake robot did not move: {x0:.3f} → {robot.x:.3f}"

    def test_navigate_to_converges(self, robot_and_env):
        robot, env = robot_and_env
        h = env.apply_action("navigate_to", x=1.0, y=0.0, timeout_s=30.0)
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if h.status in (GoalStatus.SUCCEEDED, GoalStatus.FAILED, GoalStatus.CANCELLED):
                break
            time.sleep(0.1)
        assert h.status is GoalStatus.SUCCEEDED, f"navigate_to got {h.status}: {h.error}"
        # robot should be near (1, 0) within tolerance
        assert math.hypot(robot.x - 1.0, robot.y) < 0.2, f"robot at ({robot.x:.2f}, {robot.y:.2f})"

    def test_stop_action_cancels_active_goal(self, robot_and_env):
        _, env = robot_and_env
        h = env.apply_action("navigate_to", x=5.0, y=0.0)
        time.sleep(0.3)
        assert h.status is GoalStatus.RUNNING
        stop = env.apply_action("stop")
        assert stop.status is GoalStatus.SUCCEEDED
        time.sleep(0.1)
        assert h.status is GoalStatus.CANCELLED

    def test_get_pose_action(self, robot_and_env):
        _, env = robot_and_env
        h = env.apply_action("get_pose")
        assert h.status is GoalStatus.SUCCEEDED
        assert "x" in h.result and "heading_deg" in h.result

    def test_unknown_action_fails(self, robot_and_env):
        _, env = robot_and_env
        h = env.apply_action("fly")
        assert h.status is GoalStatus.FAILED
