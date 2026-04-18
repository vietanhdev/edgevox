"""agent_event_to_ws_message — pure dispatch for the per-turn forwarder.

We don't spin up a FastAPI server here; we just verify the mapping. The
end-to-end WebSocket path is covered by :mod:`tests.test_server_app`.
"""

from __future__ import annotations

from dataclasses import asdict

from edgevox.agents.base import AgentEvent
from edgevox.examples.agents.chess_robot.face_hook import RobotFaceEvent
from edgevox.llm.tools import ToolCallResult
from edgevox.server.ws import agent_event_to_ws_message


class TestRobotFaceForwarding:
    def test_forwards_robot_face_event(self):
        face = RobotFaceEvent(
            mood="amused",
            gaze_x=0.5,
            gaze_y=-0.2,
            persona="trash_talker",
            tempo="speaking",
            last_move_san="e4",
        )
        event = AgentEvent(kind="robot_face", agent_name="chess_robot", payload=asdict(face))
        msg = agent_event_to_ws_message(event)
        assert msg["type"] == "robot_face"
        assert msg["mood"] == "amused"
        assert msg["gaze_x"] == 0.5
        assert msg["gaze_y"] == -0.2
        assert msg["persona"] == "trash_talker"
        assert msg["tempo"] == "speaking"
        assert msg["last_move_san"] == "e4"

    def test_forwards_none_payload_as_empty(self):
        event = AgentEvent(kind="robot_face", agent_name="x", payload=None)
        msg = agent_event_to_ws_message(event)
        assert msg == {"type": "robot_face"}


class TestExistingForwardingUntouched:
    def test_tool_call(self):
        result = ToolCallResult(name="play_user_move", arguments={"move": "e4"}, result={"ok": True})
        event = AgentEvent(kind="tool_call", agent_name="chess", payload=result)
        msg = agent_event_to_ws_message(event)
        assert msg["type"] == "tool_call"
        assert msg["name"] == "play_user_move"
        assert msg["ok"] is True
        assert msg["error"] is None

    def test_skill_goal(self):
        event = AgentEvent(kind="skill_goal", agent_name="robot", payload={"goal": "pick"})
        msg = agent_event_to_ws_message(event)
        assert msg == {"type": "skill_goal", "agent": "robot", "payload": {"goal": "pick"}}

    def test_unknown_kind_returns_none(self):
        event = AgentEvent(kind="agent_start", agent_name="x", payload={})
        assert agent_event_to_ws_message(event) is None
