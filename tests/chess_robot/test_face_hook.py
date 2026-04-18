"""RobotFaceHook — fires on the right points, emits the right event."""

from __future__ import annotations

import pytest

from edgevox.agents.base import AgentContext, AgentEvent
from edgevox.agents.hooks import (
    AFTER_LLM,
    AFTER_TOOL,
    BEFORE_LLM,
    ON_RUN_END,
    ON_RUN_START,
)
from edgevox.examples.agents.chess_robot.face_hook import (
    MOVE_TOOLS,
    RobotFaceHook,
)
from edgevox.examples.agents.chess_robot.mood import Mood
from edgevox.llm.tools import ToolCallResult


@pytest.fixture
def captured_events():
    events: list[AgentEvent] = []
    return events


@pytest.fixture
def ctx(env, captured_events):
    return AgentContext(deps=env, on_event=captured_events.append)


class TestFireRegistration:
    def test_points_covered(self):
        hook = RobotFaceHook()
        assert hook.points == {ON_RUN_START, AFTER_TOOL, AFTER_LLM, ON_RUN_END}

    def test_priority_in_observability_tier(self):
        assert RobotFaceHook().priority == 50


class TestOnRunStart:
    def test_emits_thinking_face(self, ctx, captured_events):
        RobotFaceHook()(ON_RUN_START, ctx, {"task": "your move"})
        assert len(captured_events) == 1
        ev = captured_events[0]
        assert ev.kind == "robot_face"
        assert ev.payload["tempo"] == "thinking"
        assert ev.payload["mood"] == Mood.CALM.value

    def test_no_emit_without_chess_env(self, captured_events):
        plain = AgentContext(deps=object(), on_event=captured_events.append)
        RobotFaceHook()(ON_RUN_START, plain, {"task": "hi"})
        assert captured_events == []


class TestAfterTool:
    def test_engine_move_triggers_face(self, ctx, env, captured_events):
        env.play_user_move("e4")
        outcome = ToolCallResult(name="engine_move", arguments={}, result={"ok": True})
        RobotFaceHook()(AFTER_TOOL, ctx, outcome)
        assert len(captured_events) == 1
        assert captured_events[0].payload["tempo"] == "thinking"

    def test_play_user_move_triggers_face(self, ctx, env, captured_events):
        env.play_user_move("e4")
        outcome = ToolCallResult(name="play_user_move", arguments={"move": "e4"}, result={"ok": True})
        RobotFaceHook()(AFTER_TOOL, ctx, outcome)
        assert len(captured_events) == 1

    def test_non_move_tool_is_ignored(self, ctx, captured_events):
        outcome = ToolCallResult(name="get_board_state", arguments={}, result={"fen": "..."})
        RobotFaceHook()(AFTER_TOOL, ctx, outcome)
        assert captured_events == []

    def test_failed_tool_is_ignored(self, ctx, captured_events):
        outcome = ToolCallResult(name="engine_move", arguments={}, error="illegal")
        RobotFaceHook()(AFTER_TOOL, ctx, outcome)
        assert captured_events == []

    def test_all_move_tools_are_recognised(self):
        assert "engine_move" in MOVE_TOOLS
        assert "play_user_move" in MOVE_TOOLS
        assert "new_game" in MOVE_TOOLS
        assert "undo_last_move" in MOVE_TOOLS

    def test_gaze_follows_last_move(self, ctx, env, captured_events):
        env.play_user_move("e4")
        outcome = ToolCallResult(name="play_user_move", arguments={"move": "e4"}, result={"ok": True})
        RobotFaceHook()(AFTER_TOOL, ctx, outcome)
        gx = captured_events[0].payload["gaze_x"]
        gy = captured_events[0].payload["gaze_y"]
        # e4 is slightly right of centre, lower half.
        assert gx > 0
        assert gy < 0


class TestAfterLlmAndRunEnd:
    def test_after_llm_emits_speaking_tempo(self, ctx, captured_events):
        RobotFaceHook()(AFTER_LLM, ctx, {"content": "your move", "tool_calls": []})
        assert captured_events[0].payload["tempo"] == "speaking"

    def test_run_end_emits_idle(self, ctx, captured_events):
        RobotFaceHook()(ON_RUN_END, ctx, None)
        assert captured_events[0].payload["tempo"] == "idle"

    def test_before_llm_is_not_handled(self, ctx, captured_events):
        # BEFORE_LLM isn't in `points`; calling the hook with it is a
        # no-op (we defend against framework changes that might still
        # route stray points through).
        result = RobotFaceHook()(BEFORE_LLM, ctx, {"messages": [], "hop": 0})
        assert result is None
        assert captured_events == []


class TestHookStateIsolation:
    def test_two_instances_do_not_share_state(self, ctx):
        h1 = RobotFaceHook(persona="grandmaster")
        h2 = RobotFaceHook(persona="trash_talker")
        h1(ON_RUN_START, ctx, {"task": "go"})
        h2(ON_RUN_START, ctx, {"task": "go"})
        assert id(h1) in ctx.hook_state
        assert id(h2) in ctx.hook_state
        assert ctx.hook_state[id(h1)]["last"].persona == "grandmaster"
        assert ctx.hook_state[id(h2)]["last"].persona == "trash_talker"


class TestPersonaPropagates:
    def test_persona_in_payload(self, ctx, captured_events):
        RobotFaceHook(persona="trash_talker")(ON_RUN_START, ctx, {"task": "x"})
        assert captured_events[0].payload["persona"] == "trash_talker"


class TestEventShapeIsJsonSafe:
    """The server forwards ``event.payload`` via ``json.dumps``. Ensure
    every field is a JSON primitive."""

    def test_payload_is_json_round_trippable(self, ctx, captured_events):
        import json

        RobotFaceHook()(ON_RUN_START, ctx, {"task": "x"})
        encoded = json.dumps(captured_events[0].payload)
        decoded = json.loads(encoded)
        assert decoded["mood"] in {m.value for m in Mood}
        assert isinstance(decoded["gaze_x"], (int, float))
        assert isinstance(decoded["gaze_y"], (int, float))
        assert isinstance(decoded["tempo"], str)
