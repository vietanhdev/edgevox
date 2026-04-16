"""Smoke tests for the built-in agent examples shipped with edgevox.

No LLM is loaded — we verify that example modules import, produce valid
tool schemas, and dispatch correctly so the examples can't silently
bit-rot as the tool API evolves.
"""

from __future__ import annotations

import pytest

from edgevox.examples.agents import dev_toolbox, home_assistant, robot_commander
from edgevox.llm import ToolRegistry


@pytest.fixture
def home():
    return home_assistant


@pytest.fixture
def robot():
    return robot_commander


@pytest.fixture
def devbox():
    return dev_toolbox


class TestFrameworkImports:
    def test_agent_apps_export_framework_instance(self, home, robot, devbox):
        from edgevox.examples.agents.framework import AgentApp

        assert isinstance(home.APP, AgentApp)
        assert isinstance(robot.APP, AgentApp)
        assert isinstance(devbox.APP, AgentApp)

    def test_cli_dispatcher_lists_subcommands(self):
        from edgevox.examples.agents import cli

        cli._lazy_subcommands()
        assert {"home", "robot", "dev", "robot-scout"} <= set(cli.SUBCOMMANDS)


class TestHomeAssistantExample:
    def test_registry_roundtrip(self, home):
        reg = ToolRegistry().register(*home.HOME_TOOLS)
        assert {"list_rooms", "set_light", "set_thermostat", "get_weather"} <= set(reg.tools)

    def test_set_light_happy_path(self, home):
        reg = ToolRegistry().register(*home.HOME_TOOLS)
        out = reg.dispatch("set_light", {"room": "kitchen", "on": True})
        assert out.ok
        assert "kitchen" in out.result

    def test_set_light_unknown_room_reports_error(self, home):
        reg = ToolRegistry().register(*home.HOME_TOOLS)
        out = reg.dispatch("set_light", {"room": "garage", "on": True})
        assert not out.ok
        assert "garage" in out.error

    def test_thermostat_out_of_range(self, home):
        reg = ToolRegistry().register(*home.HOME_TOOLS)
        out = reg.dispatch("set_thermostat", {"celsius": 99.0})
        assert not out.ok
        assert "10 and 30" in out.error

    def test_weather_returns_dict(self, home):
        reg = ToolRegistry().register(*home.HOME_TOOLS)
        out = reg.dispatch("get_weather", {"city": "Hanoi"})
        assert out.ok
        assert out.result["city"] == "Hanoi"
        assert "temp_c" in out.result

    def test_schemas_have_descriptions(self, home):
        for t in home.HOME_TOOLS:
            descriptor = t.__edgevox_tool__
            assert descriptor.description, f"{descriptor.name} missing description"


class TestRobotExample:
    def test_move_and_pose(self, robot):
        # Reset state
        robot.ROBOT.x = 0.0
        robot.ROBOT.y = 0.0
        robot.ROBOT.heading_deg = 0.0
        reg = ToolRegistry().register(*robot.ROBOT_TOOLS)

        move = reg.dispatch("move_forward", {"meters": 2.0})
        assert move.ok

        pose = reg.dispatch("get_pose", {})
        assert pose.ok
        assert pytest.approx(pose.result["x"], abs=0.01) == 2.0

    def test_move_negative_rejected(self, robot):
        reg = ToolRegistry().register(*robot.ROBOT_TOOLS)
        out = reg.dispatch("move_forward", {"meters": -1.0})
        assert not out.ok

    def test_turn_wraps(self, robot):
        robot.ROBOT.heading_deg = 350.0
        reg = ToolRegistry().register(*robot.ROBOT_TOOLS)
        out = reg.dispatch("turn", {"degrees": 20.0})
        assert out.ok
        assert robot.ROBOT.heading_deg == 10.0

    def test_go_home_resets(self, robot):
        robot.ROBOT.x = 5.0
        robot.ROBOT.y = 3.0
        reg = ToolRegistry().register(*robot.ROBOT_TOOLS)
        reg.dispatch("go_home", {})
        assert robot.ROBOT.x == 0.0
        assert robot.ROBOT.y == 0.0


class TestDevToolboxExample:
    def test_calculate_ok(self, devbox):
        reg = ToolRegistry().register(*devbox.DEV_TOOLS)
        out = reg.dispatch("calculate", {"expression": "2 * (3 + 4)"})
        assert out.ok
        assert out.result == 14.0

    def test_calculate_rejects_code(self, devbox):
        reg = ToolRegistry().register(*devbox.DEV_TOOLS)
        out = reg.dispatch("calculate", {"expression": "__import__('os')"})
        assert not out.ok

    def test_unit_conversion(self, devbox):
        reg = ToolRegistry().register(*devbox.DEV_TOOLS)
        out = reg.dispatch("celsius_to_fahrenheit", {"celsius": 100})
        assert out.ok
        assert out.result == 212.0

    def test_notes_roundtrip(self, devbox):
        devbox.NOTES.clear()
        reg = ToolRegistry().register(*devbox.DEV_TOOLS)
        reg.dispatch("add_note", {"text": "buy milk"})
        reg.dispatch("add_note", {"text": "call mom"})
        out = reg.dispatch("list_notes", {})
        assert out.result == ["buy milk", "call mom"]
