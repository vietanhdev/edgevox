"""ROS2-driven end-to-end test for the ``AgentApp --ros2`` flow.

Exercises the path where ``edgevox-agent robot-irsim --ros2 --text-mode``
receives commands over ``/<ns>/text_input`` and publishes:

- ``robot_state`` with the sim world snapshot
- ``transcription`` with the text that triggered the turn
- ``response`` with the agent's reply
- ``agent_event`` for every tool/skill event

Auto-skips without ``rclpy`` or ``irsim``.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("irsim")

from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from edgevox.agents import AgentContext, AgentResult, LLMAgent  # noqa: E402
from edgevox.examples.agents.framework import AgentApp  # noqa: E402
from edgevox.integrations.ros2_qos import reliable_qos, state_qos  # noqa: E402


def _ensure_rclpy_down() -> None:
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


class _ScriptedAgent(LLMAgent):
    """LLMAgent stub that echoes the user turn and fires one canned event.

    Avoids a real LLM download — we're testing the ROS2 plumbing, not
    Gemma. ``on_event`` is called so the bridge's agent_event publisher
    actually sees something.
    """

    def __init__(self) -> None:
        super().__init__(
            name="ScoutStub",
            description="scripted stub",
            instructions="stub",
            tools=None,
            skills=None,
        )

    def run(self, user_message: str, ctx: AgentContext) -> AgentResult:
        if ctx.on_event is not None:
            from edgevox.agents.base import AgentEvent
            from edgevox.llm import ToolCallResult

            ctx.on_event(
                AgentEvent(
                    agent_name=self.name,
                    kind="tool_call",
                    payload=ToolCallResult(
                        name="echo",
                        arguments={"text": user_message},
                        result=user_message,
                    ),
                )
            )
        reply = f"echo: {user_message}"
        return AgentResult(reply=reply, agent_name=self.name, preempted=False)


def _start_agent_app_in_thread(app: AgentApp, argv: list[str]) -> tuple[threading.Thread, list[Exception]]:
    errors: list[Exception] = []

    def _runner() -> None:
        try:
            app.run(argv)
        except SystemExit:
            pass
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=_runner, name="agent-app-runner", daemon=True)
    t.start()
    return t, errors


class TestAgentAppRos2:
    def test_text_input_drives_agent_and_publishes_state(self, tmp_path, monkeypatch):
        """Full round trip: a sibling node publishes on ``text_input``,
        the AgentApp worker handles it, and the reply + robot_state
        snapshot land on their respective topics."""
        _ensure_rclpy_down()

        from edgevox.integrations.sim.irsim import IrSimEnvironment

        env = IrSimEnvironment(render=False)

        app = AgentApp(
            name="ScoutStub",
            description="stub",
            agent=_ScriptedAgent(),
            deps=env,
            stop_words=("stop", "halt"),
            greeting=None,
        )

        # Feed an empty-stdin so ``_run_text`` blocks on input() without
        # exiting. We short-circuit the LLM loader so the test doesn't
        # try to download Gemma.
        monkeypatch.setattr(
            "edgevox.examples.agents._repl._load_llm_with_progress",
            lambda *a, **kw: object(),
        )
        monkeypatch.setattr(
            "edgevox.agents.workflow._bind_llm_recursive",
            lambda *a, **kw: None,
        )

        # Pipe fake stdin that never produces a line, so the REPL's
        # ``input()`` blocks forever and we only drive the agent from
        # the ROS2 worker path.
        import os

        read_fd, write_fd = os.pipe()
        monkeypatch.setattr("sys.stdin", os.fdopen(read_fd, "r", buffering=1))
        # ``_input_with_pump`` checks ``sys.stdin.isatty()`` — force
        # False so it hits the blocking branch.
        # (Piped stdin already reports isatty=False.)

        argv = [
            "--text-mode",
            "--ros2",
            "--ros2-namespace",
            "/agent_test",
            "--ros2-state-hz",
            "20",
        ]
        runner, errors = _start_agent_app_in_thread(app, argv)

        try:
            # Wait for bridge to come up — ``rclpy`` will be inited by
            # the worker thread. Poll for a fresh rclpy context.
            for _ in range(100):
                if rclpy.ok():
                    break
                time.sleep(0.05)
            assert rclpy.ok(), "agent app did not initialise rclpy"

            # Sibling node to drive and observe.
            probe = rclpy.create_node("agent_probe")
            executor = SingleThreadedExecutor()
            executor.add_node(probe)

            responses: list[str] = []
            transcriptions: list[str] = []
            robot_states: list[dict] = []
            agent_events: list[dict] = []

            probe.create_subscription(
                String, "/agent_test/response", lambda m: responses.append(m.data), reliable_qos()
            )
            probe.create_subscription(
                String, "/agent_test/transcription", lambda m: transcriptions.append(m.data), reliable_qos()
            )
            probe.create_subscription(
                String,
                "/agent_test/robot_state",
                lambda m: robot_states.append(json.loads(m.data)),
                state_qos(),
            )
            probe.create_subscription(
                String,
                "/agent_test/agent_event",
                lambda m: agent_events.append(json.loads(m.data)),
                reliable_qos(),
            )

            # Wait for the agent node to appear in discovery before
            # publishing — otherwise the message is dropped on a
            # volatile QoS pub/sub pair that hasn't paired up yet.
            def _agent_node_visible() -> bool:
                names_and_ns = probe.get_node_names_and_namespaces()
                return any(ns.startswith("/agent_test") for _, ns in names_and_ns)

            discovery_deadline = time.monotonic() + 10.0
            while time.monotonic() < discovery_deadline and not _agent_node_visible():
                executor.spin_once(timeout_sec=0.1)
            assert _agent_node_visible(), "agent node never showed up in DDS discovery"

            pub = probe.create_publisher(String, "/agent_test/text_input", reliable_qos())
            # Pub/sub pairing: poke discovery a few more times with a
            # modest back-off so the subscriber on the agent side is
            # ready when we publish.
            for _ in range(30):
                executor.spin_once(timeout_sec=0.1)
                if pub.get_subscription_count() >= 1:
                    break

            msg = String()
            msg.data = "what is your battery level"
            pub.publish(msg)

            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.1)
                if transcriptions and responses and robot_states:
                    break

            assert transcriptions, "no transcription published"
            assert "battery" in transcriptions[0]
            assert responses, "no response published"
            assert responses[0].startswith("echo:")
            assert robot_states, "no robot_state published"
            assert "robot" in robot_states[0]
            # Agent event for our scripted tool_call should arrive.
            event_kinds = [e.get("kind") for e in agent_events]
            assert "tool_call" in event_kinds

            executor.shutdown()
            probe.destroy_node()
        finally:
            os.close(write_fd)  # EOF the pipe so input() returns, REPL ends
            runner.join(timeout=5.0)
            env.close()
            _ensure_rclpy_down()

        assert not errors, f"agent app errored: {errors}"
