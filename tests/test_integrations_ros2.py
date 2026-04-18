"""ROS2 bridge integration tests — exercise the real rclpy runtime.

These tests are auto-skipped when ``rclpy`` is unavailable (e.g. no
sourced ROS2 workspace), so the suite stays green on plain dev machines
and CI. To run them locally::

    source /opt/ros/jazzy/setup.bash
    pytest tests/test_integrations_ros2.py -v

Each test owns its own bridge and tears down ``rclpy`` on exit so
independent tests don't bleed node state into each other.
"""

from __future__ import annotations

import json
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    ReliabilityPolicy,
)
from std_msgs.msg import Float32, String  # noqa: E402

from edgevox.integrations.ros2_bridge import (  # noqa: E402
    NullBridge,
    ROS2Bridge,
    create_bridge,
)
from edgevox.integrations.ros2_qos import reliable_qos, sensor_qos, state_qos  # noqa: E402


def _make_node_with_executor(name: str):
    """Create a fresh node with its own dedicated executor.

    ``rclpy.spin_once(node)`` uses the default global executor — if the
    bridge's spin thread is already spinning that executor, a sibling
    spin_once raises ``RuntimeError('Executor is already spinning')``.
    Using a dedicated ``SingleThreadedExecutor`` per test listener keeps
    spins independent.
    """
    node = rclpy.create_node(name)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    return node, executor


def _ensure_rclpy_down() -> None:
    """Best-effort rclpy shutdown so the next test starts clean."""
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


@pytest.fixture
def bridge():
    _ensure_rclpy_down()
    b = create_bridge(enabled=True, namespace="/edgevox_test")
    assert isinstance(b, ROS2Bridge)
    yield b
    b.shutdown()
    _ensure_rclpy_down()


@pytest.fixture
def bare_bridge():
    """Bridge without a background spin thread — useful when a test
    drives ``spin_once`` explicitly on a sibling node."""
    _ensure_rclpy_down()
    rclpy.init()
    b = ROS2Bridge(namespace="/edgevox_test")
    yield b
    b.shutdown()
    _ensure_rclpy_down()


# ---------------------------------------------------------------------------
# Structural checks — node, topics, QoS, parameters
# ---------------------------------------------------------------------------


class TestROS2BridgeStructure:
    def test_create_bridge_returns_ros2bridge(self, bridge):
        assert isinstance(bridge, ROS2Bridge)

    def test_all_publishers_present(self, bridge):
        topics = {p.topic_name for p in bridge._node.publishers}
        expected = {
            "/edgevox_test/transcription",
            "/edgevox_test/response",
            "/edgevox_test/state",
            "/edgevox_test/audio_level",
            "/edgevox_test/metrics",
            "/edgevox_test/bot_token",
            "/edgevox_test/bot_sentence",
            "/edgevox_test/wakeword",
            "/edgevox_test/info",
            "/edgevox_test/robot_state",
            "/edgevox_test/agent_event",
        }
        missing = expected - topics
        assert not missing, f"missing publishers: {missing}"

    def test_all_subscribers_present(self, bridge):
        topics = {s.topic_name for s in bridge._node.subscriptions}
        expected = {
            "/edgevox_test/tts_request",
            "/edgevox_test/command",
            "/edgevox_test/text_input",
            "/edgevox_test/interrupt",
            "/edgevox_test/set_language",
            "/edgevox_test/set_voice",
        }
        missing = expected - topics
        assert not missing, f"missing subscribers: {missing}"

    def test_parameter_defaults(self, bridge):
        params = bridge._node.get_parameters(["language", "voice", "muted"])
        assert params[0].value == "en"
        assert params[1].value == ""
        assert params[2].value is False

    def test_qos_state_is_transient_local(self, bridge):
        pubs = {p.topic_name: p for p in bridge._node.publishers}
        q = pubs["/edgevox_test/state"].qos_profile
        assert q.durability == DurabilityPolicy.TRANSIENT_LOCAL
        assert q.reliability == ReliabilityPolicy.RELIABLE

    def test_qos_sensor_is_best_effort(self, bridge):
        pubs = {p.topic_name: p for p in bridge._node.publishers}
        for topic in ("/edgevox_test/audio_level", "/edgevox_test/bot_token"):
            q = pubs[topic].qos_profile
            assert q.reliability == ReliabilityPolicy.BEST_EFFORT, topic

    def test_qos_reliable_is_reliable_volatile(self, bridge):
        pubs = {p.topic_name: p for p in bridge._node.publishers}
        q = pubs["/edgevox_test/transcription"].qos_profile
        assert q.reliability == ReliabilityPolicy.RELIABLE
        assert q.durability == DurabilityPolicy.VOLATILE

    def test_publish_methods_do_not_raise(self, bridge):
        bridge.publish_state("listening")
        bridge.publish_transcription("hello")
        bridge.publish_response("hi there")
        bridge.publish_audio_level(0.5)
        bridge.publish_metrics({"stt": 0.1, "llm": 0.2})
        bridge.publish_bot_token("hi")
        bridge.publish_bot_sentence("Hello there.")
        bridge.publish_wakeword("hey jarvis")
        bridge.publish_info({"query": "test", "ok": True})
        bridge.publish_robot_state({"x": 1.0, "y": 2.0, "moving": False})
        bridge.publish_agent_event({"kind": "tool_call", "tool": "noop"})


class TestCustomNamespace:
    def test_nested_namespace_prefixes_all_topics(self):
        _ensure_rclpy_down()
        rclpy.init()
        try:
            b = ROS2Bridge(namespace="/robot7/voice")
            pubs = {p.topic_name for p in b._node.publishers}
            assert "/robot7/voice/transcription" in pubs
            assert "/robot7/voice/state" in pubs
            subs = {s.topic_name for s in b._node.subscriptions}
            assert "/robot7/voice/tts_request" in subs
            b.shutdown()
        finally:
            _ensure_rclpy_down()


class TestNullBridge:
    def test_null_bridge_is_returned_when_disabled(self):
        assert isinstance(create_bridge(enabled=False), NullBridge)

    def test_null_bridge_accepts_all_calls(self):
        n = NullBridge()
        n.publish_transcription("")
        n.publish_response("")
        n.publish_state("")
        n.publish_audio_level(0.0)
        n.publish_metrics({})
        n.publish_bot_token("")
        n.publish_bot_sentence("")
        n.publish_wakeword("")
        n.publish_info({})
        n.set_tts_callback(lambda t: None)
        n.set_command_callback(lambda c: None)
        n.set_text_input_callback(lambda t: None)
        n.set_interrupt_callback(lambda: None)
        n.set_set_language_callback(lambda s: None)
        n.set_set_voice_callback(lambda s: None)
        n.set_query_callback(lambda q: None)
        n.spin()
        n.shutdown()


# ---------------------------------------------------------------------------
# Live publish / subscribe round-trips on the DDS bus
# ---------------------------------------------------------------------------


def _wait_for(predicate, timeout_s: float = 5.0, interval_s: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


class TestROS2LivePubSub:
    """Exercise the DDS transport — publish on the bridge, subscribe
    from a sibling node, and confirm the message actually lands."""

    def test_transcription_delivered_to_external_subscriber(self, bridge):
        listener, executor = _make_node_with_executor("test_listener")
        received: list[str] = []

        def on_msg(msg: String) -> None:
            received.append(msg.data)

        listener.create_subscription(String, "/edgevox_test/transcription", on_msg, reliable_qos())

        # Give discovery a moment to settle
        for _ in range(20):
            executor.spin_once(timeout_sec=0.05)

        bridge.publish_transcription("test transcription value")

        def drain() -> bool:
            executor.spin_once(timeout_sec=0.05)
            return bool(received)

        assert _wait_for(drain, timeout_s=5.0), "transcription never arrived"
        assert "test transcription value" in received
        executor.shutdown()
        listener.destroy_node()

    def test_state_late_joiner_gets_last_value(self, bridge):
        bridge.publish_state("listening")
        time.sleep(0.1)

        listener, executor = _make_node_with_executor("test_state_late_joiner")
        received: list[str] = []

        def on_msg(msg: String) -> None:
            received.append(msg.data)

        listener.create_subscription(String, "/edgevox_test/state", on_msg, state_qos())

        def drain() -> bool:
            executor.spin_once(timeout_sec=0.05)
            return bool(received)

        assert _wait_for(drain, timeout_s=5.0), "transient-local state never arrived"
        assert received[0] == "listening"
        executor.shutdown()
        listener.destroy_node()

    def test_audio_level_best_effort_delivery(self, bridge):
        listener, executor = _make_node_with_executor("test_audio_listener")
        received: list[float] = []

        def on_msg(msg: Float32) -> None:
            received.append(msg.data)

        listener.create_subscription(Float32, "/edgevox_test/audio_level", on_msg, sensor_qos())

        for _ in range(20):
            executor.spin_once(timeout_sec=0.05)

        for v in (0.1, 0.2, 0.3):
            bridge.publish_audio_level(v)

        def drain() -> bool:
            executor.spin_once(timeout_sec=0.05)
            return len(received) >= 3

        assert _wait_for(drain, timeout_s=5.0), f"only got {received}"
        assert received[0] == pytest.approx(0.1, abs=1e-6)
        executor.shutdown()
        listener.destroy_node()

    def test_tts_request_triggers_callback(self, bridge):
        got: list[str] = []
        bridge.set_tts_callback(lambda text: got.append(text))

        pub_node, executor = _make_node_with_executor("test_tts_publisher")
        pub = pub_node.create_publisher(String, "/edgevox_test/tts_request", reliable_qos())

        for _ in range(20):
            executor.spin_once(timeout_sec=0.05)

        msg = String()
        msg.data = "please say this"
        pub.publish(msg)

        assert _wait_for(lambda: bool(got), timeout_s=5.0), "callback never fired"
        assert got[0] == "please say this"
        executor.shutdown()
        pub_node.destroy_node()

    def test_query_command_publishes_info(self, bridge):
        bridge.set_query_callback(lambda q: {"q": q, "voices": ["a", "b"]} if q == "list_voices" else None)

        pub_node, executor = _make_node_with_executor("test_query_publisher")
        pub = pub_node.create_publisher(String, "/edgevox_test/command", reliable_qos())

        info_received: list[dict] = []

        def on_info(msg: String) -> None:
            info_received.append(json.loads(msg.data))

        pub_node.create_subscription(String, "/edgevox_test/info", on_info, reliable_qos())

        for _ in range(20):
            executor.spin_once(timeout_sec=0.05)

        msg = String()
        msg.data = "list_voices"
        pub.publish(msg)

        def drain() -> bool:
            executor.spin_once(timeout_sec=0.05)
            return bool(info_received)

        assert _wait_for(drain, timeout_s=5.0), "info topic never published"
        payload = info_received[0]
        assert payload["query"] == "list_voices"
        assert payload["voices"] == ["a", "b"]
        executor.shutdown()
        pub_node.destroy_node()


# ---------------------------------------------------------------------------
# Sim-adapter round-trip: ROS2 text_input → agent skill → IR-SIM navigation
# ---------------------------------------------------------------------------


class TestROS2WithIrSim:
    """Full integration: ROS2 bridge receives a text_input command, a
    callback dispatches a navigation skill on IR-SIM, and the resulting
    robot pose is surfaced back through a custom ``sim_state`` topic.

    Mirrors how a downstream user would compose the bridge with a sim
    adapter — the bridge doesn't ship its own sim integration, but any
    callback can drive ``SimEnvironment.apply_action`` and publish
    results on an extension topic.
    """

    def test_text_input_drives_navigation(self, bridge):
        irsim = pytest.importorskip("irsim")  # noqa: F841
        from edgevox.agents.skills import GoalStatus
        from edgevox.integrations.sim.irsim import IrSimEnvironment

        env = IrSimEnvironment(render=False)
        try:
            # Set up a side-channel: publish sim robot pose on a custom
            # topic whenever text_input arrives.
            pose_pub = bridge._node.create_publisher(String, "sim_state", reliable_qos())

            handle_ref: dict[str, object] = {}

            def on_text_input(text: str) -> None:
                # expect strings like "goto kitchen"
                parts = text.strip().split()
                if len(parts) == 2 and parts[0] == "goto":
                    h = env.apply_action("navigate_to", room=parts[1])
                    handle_ref["handle"] = h

            bridge.set_text_input_callback(on_text_input)

            # Publish ROS2 text_input from a sibling node
            pub_node, pub_exec = _make_node_with_executor("test_text_input_pub")
            pub = pub_node.create_publisher(String, "/edgevox_test/text_input", reliable_qos())
            for _ in range(20):
                pub_exec.spin_once(timeout_sec=0.05)

            msg = String()
            msg.data = "goto kitchen"
            pub.publish(msg)

            assert _wait_for(lambda: "handle" in handle_ref, timeout_s=5.0), "text_input callback never ran"

            # Drive the sim until the goal succeeds
            handle = handle_ref["handle"]
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if handle.status in (
                    GoalStatus.SUCCEEDED,
                    GoalStatus.FAILED,
                    GoalStatus.CANCELLED,
                ):
                    break
                time.sleep(0.05)

            assert handle.status is GoalStatus.SUCCEEDED, f"navigate_to did not succeed, got {handle.status}"

            pose = env.get_world_state()["robot"]
            assert abs(pose["x"] - 8.0) < 0.5
            assert abs(pose["y"] - 2.0) < 0.5

            # Publish the final pose through our custom sim_state topic
            pose_pub.publish(String(data=json.dumps(pose)))

            # A sibling node should be able to subscribe and receive it
            sub_node, sub_exec = _make_node_with_executor("test_sim_state_sub")
            received: list[dict] = []
            sub_node.create_subscription(
                String,
                "/edgevox_test/sim_state",
                lambda m: received.append(json.loads(m.data)),
                reliable_qos(),
            )
            # Republish after the subscriber is up (volatile QoS)
            for _ in range(10):
                sub_exec.spin_once(timeout_sec=0.05)
            pose_pub.publish(String(data=json.dumps(pose)))

            def drain() -> bool:
                sub_exec.spin_once(timeout_sec=0.05)
                return bool(received)

            assert _wait_for(drain, timeout_s=5.0), "sim_state never arrived"
            assert received[0]["x"] == pose["x"]

            pub_exec.shutdown()
            sub_exec.shutdown()
            pub_node.destroy_node()
            sub_node.destroy_node()
        finally:
            env.close()


# ---------------------------------------------------------------------------
# MuJoCo arm — gantry variant is bundled, no network fetch
# ---------------------------------------------------------------------------


class TestROS2WithMujocoArm:
    def test_tts_request_drives_move_to(self, bridge):
        mujoco = pytest.importorskip("mujoco")  # noqa: F841
        from edgevox.agents.skills import GoalStatus
        from edgevox.integrations.sim.mujoco_arm import MujocoArmEnvironment

        env = MujocoArmEnvironment(
            model_source="gantry",  # bundled, zero network
            render=False,
            allow_hf_download=False,
        )
        try:
            handle_ref: dict[str, object] = {}

            def on_tts(text: str) -> None:
                # "move_to 0.1 0.0 0.2" → drive the arm
                parts = text.strip().split()
                if parts and parts[0] == "move_to" and len(parts) == 4:
                    x, y, z = (float(p) for p in parts[1:])
                    handle_ref["handle"] = env.apply_action("move_to", x=x, y=y, z=z)

            bridge.set_tts_callback(on_tts)

            pub_node, pub_exec = _make_node_with_executor("test_mujoco_driver")
            pub = pub_node.create_publisher(String, "/edgevox_test/tts_request", reliable_qos())
            for _ in range(20):
                pub_exec.spin_once(timeout_sec=0.05)

            # Pick a target inside the gantry workspace
            home = env.get_world_state()["ee"]
            target_x = round(home["x"] + 0.05, 3)
            target_y = round(home["y"], 3)
            target_z = round(home["z"], 3)

            msg = String()
            msg.data = f"move_to {target_x} {target_y} {target_z}"
            pub.publish(msg)

            assert _wait_for(lambda: "handle" in handle_ref, timeout_s=5.0)
            handle = handle_ref["handle"]
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if handle.status in (
                    GoalStatus.SUCCEEDED,
                    GoalStatus.FAILED,
                    GoalStatus.CANCELLED,
                ):
                    break
                time.sleep(0.05)

            assert handle.status is GoalStatus.SUCCEEDED, f"move_to did not succeed, got {handle.status}"
            pub_exec.shutdown()
            pub_node.destroy_node()
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Parameter service — ``ros2 param set`` path exercises the param callback
# ---------------------------------------------------------------------------


class TestRobotStateAndAgentEvent:
    """New topics for sim/robot snapshots and agent events."""

    def test_robot_state_late_joiner_gets_last_snapshot(self, bridge):
        bridge.publish_robot_state({"x": 8.0, "y": 2.0, "moving": False})
        time.sleep(0.1)

        listener, executor = _make_node_with_executor("test_robot_state_joiner")
        received: list[dict] = []
        listener.create_subscription(
            String,
            "/edgevox_test/robot_state",
            lambda m: received.append(json.loads(m.data)),
            state_qos(),
        )

        def drain() -> bool:
            executor.spin_once(timeout_sec=0.05)
            return bool(received)

        assert _wait_for(drain, timeout_s=5.0), "robot_state never arrived"
        assert received[0]["x"] == 8.0
        executor.shutdown()
        listener.destroy_node()

    def test_agent_event_stream_round_trip(self, bridge):
        listener, executor = _make_node_with_executor("test_agent_event_listener")
        received: list[dict] = []
        listener.create_subscription(
            String,
            "/edgevox_test/agent_event",
            lambda m: received.append(json.loads(m.data)),
            reliable_qos(),
        )
        for _ in range(20):
            executor.spin_once(timeout_sec=0.05)

        bridge.publish_agent_event({"kind": "tool_call", "tool": "list_rooms", "ok": True})
        bridge.publish_agent_event({"kind": "skill_goal", "skill": "navigate_to"})
        bridge.publish_agent_event({"kind": "safety_preempt", "reason": "stop_word"})

        def drain() -> bool:
            executor.spin_once(timeout_sec=0.05)
            return len(received) >= 3

        assert _wait_for(drain, timeout_s=5.0), f"only got {received}"
        kinds = [e["kind"] for e in received]
        assert "tool_call" in kinds
        assert "skill_goal" in kinds
        assert "safety_preempt" in kinds
        executor.shutdown()
        listener.destroy_node()


class TestParameterCallbacks:
    def test_set_language_parameter_fires_callback(self, bridge):
        from rclpy.parameter import Parameter

        got: list[str] = []
        bridge.set_set_language_callback(lambda lang: got.append(lang))

        # Spin from the bridge's own thread handled by create_bridge()
        result = bridge._node.set_parameters([Parameter("language", Parameter.Type.STRING, "vi")])
        assert result[0].successful

        # Give the bridge's own spin thread a moment to deliver the event
        assert _wait_for(lambda: bool(got), timeout_s=2.0), "language param callback did not fire"
        assert got[0] == "vi"

    def test_set_muted_parameter_fires_command(self, bridge):
        from rclpy.parameter import Parameter

        cmds: list[str] = []
        bridge.set_command_callback(lambda c: cmds.append(c))

        bridge._node.set_parameters([Parameter("muted", Parameter.Type.BOOL, True)])
        assert _wait_for(lambda: "mute" in cmds, timeout_s=2.0)

        bridge._node.set_parameters([Parameter("muted", Parameter.Type.BOOL, False)])
        assert _wait_for(lambda: "unmute" in cmds, timeout_s=2.0)
