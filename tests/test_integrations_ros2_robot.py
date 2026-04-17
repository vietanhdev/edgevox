"""P4 integration tests — services, TF2, cmd_vel/pose, LaserScan, Image, Actions.

Each class tests one P4 feature end-to-end against a real rclpy
runtime. Auto-skipped without rclpy / ir-sim / mujoco / edgevox_msgs.
"""

from __future__ import annotations

import json
import math
import os
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from geometry_msgs.msg import PoseStamped, Twist  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from sensor_msgs.msg import Image, LaserScan  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from edgevox.integrations.ros2_bridge import ROS2Bridge, create_bridge  # noqa: E402
from edgevox.integrations.ros2_qos import reliable_qos, sensor_qos  # noqa: E402
from edgevox.integrations.ros2_robot import create_robot_adapter  # noqa: E402


def _ensure_rclpy_down() -> None:
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


def _make_node_with_executor(name: str):
    node = rclpy.create_node(name)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    return node, executor


def _wait_for(predicate, timeout_s: float = 5.0, interval_s: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


@pytest.fixture
def bridge():
    _ensure_rclpy_down()
    b = create_bridge(enabled=True, namespace="/p4_test")
    assert isinstance(b, ROS2Bridge)
    yield b
    b.shutdown()
    _ensure_rclpy_down()


# ---------------------------------------------------------------------------
# P4.1 — std_srvs/srv/Trigger services for query commands
# ---------------------------------------------------------------------------


class TestQueryServices:
    def test_list_voices_service(self, bridge):
        bridge.set_query_callback(lambda q: {"voices": ["a", "b", "c"]} if q == "list_voices" else None)

        client_node, executor = _make_node_with_executor("p4_query_client")
        client = client_node.create_client(Trigger, "/p4_test/list_voices")
        assert client.wait_for_service(timeout_sec=5.0), "service never appeared"

        fut = client.call_async(Trigger.Request())

        def done() -> bool:
            executor.spin_once(timeout_sec=0.05)
            return fut.done()

        assert _wait_for(done, timeout_s=5.0), "service call never completed"
        resp = fut.result()
        assert resp.success is True
        payload = json.loads(resp.message)
        assert payload["query"] == "list_voices"
        assert payload["voices"] == ["a", "b", "c"]
        executor.shutdown()
        client_node.destroy_node()

    def test_service_returns_error_when_callback_returns_none(self, bridge):
        bridge.set_query_callback(lambda q: None)

        client_node, executor = _make_node_with_executor("p4_query_client_none")
        client = client_node.create_client(Trigger, "/p4_test/hardware_info")
        assert client.wait_for_service(timeout_sec=5.0)
        fut = client.call_async(Trigger.Request())
        assert _wait_for(lambda: (executor.spin_once(timeout_sec=0.05) or True) and fut.done(), timeout_s=5.0)
        resp = fut.result()
        assert resp.success is False
        assert "no handler" in resp.message or "error" in resp.message
        executor.shutdown()
        client_node.destroy_node()


# ---------------------------------------------------------------------------
# P4.2 / P4.3 — TF2 broadcast + PoseStamped + cmd_vel (IR-SIM)
# ---------------------------------------------------------------------------


class TestRobotAdapterIrSim:
    def test_tf_pose_lidar_cmdvel(self, bridge):
        pytest.importorskip("irsim")
        from edgevox.integrations.sim.irsim import IrSimEnvironment

        env = IrSimEnvironment(render=False)
        try:
            adapter = create_robot_adapter(
                bridge._node,
                env,
                tf_hz=30.0,
                scan_hz=20.0,
            )
            assert adapter is not None
            bridge.attach_robot_adapter(adapter)
            assert adapter._has_pose2d
            assert adapter._has_velocity
            assert adapter._has_lidar

            listener, executor = _make_node_with_executor("p4_irsim_listener")
            tf_msgs: list[TFMessage] = []
            pose_msgs: list[PoseStamped] = []
            scan_msgs: list[LaserScan] = []

            listener.create_subscription(TFMessage, "/tf", lambda m: tf_msgs.append(m), reliable_qos())
            listener.create_subscription(PoseStamped, "/p4_test/pose", lambda m: pose_msgs.append(m), reliable_qos())
            listener.create_subscription(LaserScan, "/p4_test/scan", lambda m: scan_msgs.append(m), sensor_qos())

            # Drive discovery
            for _ in range(40):
                executor.spin_once(timeout_sec=0.05)

            # Wait for TF + pose + scan to arrive
            def drained() -> bool:
                executor.spin_once(timeout_sec=0.05)
                return bool(tf_msgs) and bool(pose_msgs) and bool(scan_msgs)

            assert _wait_for(drained, timeout_s=10.0), (
                f"missing tf={len(tf_msgs)} pose={len(pose_msgs)} scan={len(scan_msgs)}"
            )

            # TF contains a transform whose child is base_link
            found_base_link = False
            for m in tf_msgs:
                for t in m.transforms:
                    if t.child_frame_id == "base_link":
                        found_base_link = True
                        break
            assert found_base_link, "no base_link transform on /tf"

            # LaserScan has 180 ranges (IR-SIM default)
            assert len(scan_msgs[-1].ranges) == 180
            assert scan_msgs[-1].angle_min < 0 < scan_msgs[-1].angle_max

            # PoseStamped is in the map frame
            assert pose_msgs[-1].header.frame_id == "map"

            # Drive via cmd_vel and confirm pose changes
            pub = listener.create_publisher(Twist, "/p4_test/cmd_vel", reliable_qos())
            for _ in range(20):
                executor.spin_once(timeout_sec=0.05)
            msg = Twist()
            msg.linear.x = 0.3
            msg.angular.z = 0.0
            t_end = time.monotonic() + 3.0
            initial_pose = env.get_pose2d()
            while time.monotonic() < t_end:
                pub.publish(msg)
                executor.spin_once(timeout_sec=0.05)
                time.sleep(0.05)
            pub.publish(Twist())  # zero-stop
            final_pose = env.get_pose2d()
            dist = math.hypot(final_pose[0] - initial_pose[0], final_pose[1] - initial_pose[1])
            assert dist > 0.1, f"robot did not move under cmd_vel (moved {dist:.3f} m)"

            executor.shutdown()
            listener.destroy_node()
        finally:
            env.close()


# ---------------------------------------------------------------------------
# P4.3 — PoseStamped goal_pose → move_to (MuJoCo arm)
# P4.5 — Image publishing (MuJoCo)
# ---------------------------------------------------------------------------


class TestRobotAdapterMujoco:
    def test_tf_pose_goal_pose(self, bridge):
        pytest.importorskip("mujoco")
        from edgevox.integrations.sim.mujoco_arm import MujocoArmEnvironment

        env = MujocoArmEnvironment(model_source="gantry", render=False, allow_hf_download=False)
        try:
            adapter = create_robot_adapter(bridge._node, env, tf_hz=30.0)
            assert adapter is not None
            bridge.attach_robot_adapter(adapter)
            assert adapter._has_ee_pose
            # apply_velocity isn't on the arm adapter — that's fine
            assert not adapter._has_velocity

            listener, executor = _make_node_with_executor("p4_mujoco_listener")
            tf_msgs: list[TFMessage] = []
            pose_msgs: list[PoseStamped] = []
            listener.create_subscription(TFMessage, "/tf", lambda m: tf_msgs.append(m), reliable_qos())
            listener.create_subscription(PoseStamped, "/p4_test/pose", lambda m: pose_msgs.append(m), reliable_qos())
            for _ in range(40):
                executor.spin_once(timeout_sec=0.05)

            assert _wait_for(
                lambda: (executor.spin_once(timeout_sec=0.05) or True) and bool(tf_msgs) and bool(pose_msgs),
                timeout_s=5.0,
            )
            found_ee = any(t.child_frame_id == "ee_link" for m in tf_msgs for t in m.transforms)
            assert found_ee, "no ee_link transform on /tf"

            # Send goal_pose → should dispatch move_to
            goal_pub = listener.create_publisher(PoseStamped, "/p4_test/goal_pose", reliable_qos())
            for _ in range(20):
                executor.spin_once(timeout_sec=0.05)

            home = env.get_ee_pose()
            goal = PoseStamped()
            goal.header.stamp = bridge._node.get_clock().now().to_msg()
            goal.header.frame_id = "map"
            goal.pose.position.x = home[0] + 0.03
            goal.pose.position.y = home[1]
            goal.pose.position.z = home[2]
            goal.pose.orientation.w = 1.0
            goal_pub.publish(goal)

            def ee_moved() -> bool:
                executor.spin_once(timeout_sec=0.05)
                now = env.get_ee_pose()
                return abs(now[0] - home[0]) > 0.01

            assert _wait_for(ee_moved, timeout_s=10.0), "goal_pose did not drive the arm"
            executor.shutdown()
            listener.destroy_node()
        finally:
            env.close()

    def test_image_raw_publishes_when_mujoco_gl_set(self, bridge):
        pytest.importorskip("mujoco")
        if os.environ.get("MUJOCO_GL") not in ("egl", "osmesa", "glfw"):
            pytest.skip("MUJOCO_GL not set for headless rendering — skipping camera test")
        from edgevox.integrations.sim.mujoco_arm import MujocoArmEnvironment

        env = MujocoArmEnvironment(model_source="gantry", render=False, allow_hf_download=False)
        try:
            adapter = create_robot_adapter(bridge._node, env, image_hz=10.0, camera_size=(160, 120))
            assert adapter is not None
            bridge.attach_robot_adapter(adapter)
            assert adapter._has_camera

            listener, executor = _make_node_with_executor("p4_image_listener")
            received: list[Image] = []
            listener.create_subscription(Image, "/p4_test/image_raw", lambda m: received.append(m), sensor_qos())
            for _ in range(40):
                executor.spin_once(timeout_sec=0.05)

            def drained() -> bool:
                executor.spin_once(timeout_sec=0.05)
                return bool(received)

            assert _wait_for(drained, timeout_s=10.0), "no Image received"
            img = received[-1]
            assert img.width == 160
            assert img.height == 120
            assert img.encoding == "rgb8"
            assert len(img.data) == 160 * 120 * 3
            executor.shutdown()
            listener.destroy_node()
        finally:
            env.close()


# ---------------------------------------------------------------------------
# P4.6 — ExecuteSkill action server
# ---------------------------------------------------------------------------


class TestExecuteSkillAction:
    def test_execute_skill_drives_navigate_to(self, bridge):
        pytest.importorskip("irsim")
        edgevox_msgs = pytest.importorskip("edgevox_msgs.action")
        from edgevox.integrations.ros2_actions import create_skill_action_server
        from edgevox.integrations.sim.irsim import IrSimEnvironment

        env = IrSimEnvironment(render=False)
        try:

            def _dispatch(skill_name: str, kwargs: dict) -> object:
                return env.apply_action(skill_name, **kwargs)

            server = create_skill_action_server(bridge._node, _dispatch)
            assert server is not None
            bridge.attach_skill_action_server(server)

            # Build an ActionClient on a sibling node
            from rclpy.action import ActionClient

            client_node, executor = _make_node_with_executor("p4_skill_client")
            client = ActionClient(client_node, edgevox_msgs.ExecuteSkill, "/p4_test/execute_skill")
            assert client.wait_for_server(timeout_sec=10.0)

            goal = edgevox_msgs.ExecuteSkill.Goal()
            goal.skill_name = "navigate_to"
            goal.arguments_json = json.dumps({"room": "kitchen"})

            feedback_seen: list[str] = []

            def on_fb(fb) -> None:
                feedback_seen.append(fb.feedback.feedback_json)

            send_fut = client.send_goal_async(goal, feedback_callback=on_fb)

            def send_done() -> bool:
                executor.spin_once(timeout_sec=0.05)
                return send_fut.done()

            assert _wait_for(send_done, timeout_s=10.0)
            goal_handle = send_fut.result()
            assert goal_handle.accepted

            result_fut = goal_handle.get_result_async()
            # Drive the action server's executor while also pumping the client
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.05)
                if result_fut.done():
                    break
            assert result_fut.done(), "action never terminated"
            wrap = result_fut.result()
            # Status code 4 = SUCCEEDED per action_msgs.msg.GoalStatus.STATUS_SUCCEEDED
            from action_msgs.msg import GoalStatus as ActionGoalStatus

            assert wrap.status == ActionGoalStatus.STATUS_SUCCEEDED
            assert wrap.result.ok is True
            payload = json.loads(wrap.result.result_json)
            assert payload.get("arrived_at") in {"kitchen", "center", "(8.0,2.0)"}

            # At least one feedback delivered
            assert feedback_seen, "no feedback packets delivered"

            client.destroy()
            executor.shutdown()
            client_node.destroy_node()
        finally:
            env.close()
