"""Unit tests for the SimEnvironment protocol and ToyWorld reference."""

from __future__ import annotations

import time

from edgevox.agents.sim import SimEnvironment, ToyWorld
from edgevox.agents.skills import GoalStatus


class TestToyWorldBasics:
    def test_default_rooms(self):
        world = ToyWorld()
        assert set(world.room_names()) == {"living_room", "kitchen", "bedroom", "office"}

    def test_custom_rooms(self):
        world = ToyWorld(rooms=[("garage", 1.0, 2.0), ("attic", 3.0, 4.0)])
        assert set(world.room_names()) == {"garage", "attic"}

    def test_initial_robot_pose(self):
        world = ToyWorld(robot_pose=(1.0, 2.0, 30.0))
        state = world.get_world_state()["robot"]
        assert state["x"] == 1.0
        assert state["y"] == 2.0
        assert state["heading_deg"] == 30.0

    def test_reset_returns_robot_to_origin(self):
        world = ToyWorld()
        world._robot.x = 99.0
        world.reset()
        assert world.get_world_state()["robot"]["x"] == 0.0

    def test_implements_sim_environment_protocol(self):
        world = ToyWorld()
        assert isinstance(world, SimEnvironment)


class TestToyWorldActions:
    def test_unknown_action_returns_failed_handle(self):
        world = ToyWorld()
        handle = world.apply_action("bogus_action")
        assert handle.status is GoalStatus.FAILED
        assert "unknown" in handle.error

    def test_list_rooms_action(self):
        world = ToyWorld()
        handle = world.apply_action("list_rooms")
        assert handle.status is GoalStatus.SUCCEEDED
        assert set(handle.result) == set(world.room_names())

    def test_set_light_mutates_state(self):
        world = ToyWorld()
        handle = world.apply_action("set_light", room="kitchen", on=True)
        assert handle.status is GoalStatus.SUCCEEDED
        rooms = world.get_world_state()["rooms"]
        assert rooms["kitchen"]["light_on"] is True

    def test_set_light_unknown_room(self):
        world = ToyWorld()
        handle = world.apply_action("set_light", room="garage", on=True)
        assert handle.status is GoalStatus.FAILED

    def test_get_pose_action(self):
        world = ToyWorld(robot_pose=(3.0, 4.0, 45.0))
        handle = world.apply_action("get_pose")
        assert handle.status is GoalStatus.SUCCEEDED
        assert handle.result["x"] == 3.0
        assert handle.result["y"] == 4.0

    def test_battery_level_action(self):
        world = ToyWorld()
        handle = world.apply_action("battery_level")
        assert handle.result["battery_pct"] > 0


class TestToyWorldNavigateTo:
    def test_navigate_to_kitchen_completes(self):
        world = ToyWorld(navigate_speed=5.0)  # fast
        handle = world.apply_action("navigate_to", room="kitchen")
        status = handle.poll(timeout=5.0)
        assert status is GoalStatus.SUCCEEDED
        pose = world.get_world_state()["robot"]
        # kitchen is (4, 0)
        assert abs(pose["x"] - 4.0) < 0.1
        assert abs(pose["y"] - 0.0) < 0.1

    def test_navigate_to_unknown_room_fails(self):
        world = ToyWorld()
        handle = world.apply_action("navigate_to", room="garage")
        assert handle.status is GoalStatus.FAILED

    def test_navigate_to_without_target_fails(self):
        world = ToyWorld()
        handle = world.apply_action("navigate_to")
        assert handle.status is GoalStatus.FAILED

    def test_navigate_to_xy_coords(self):
        world = ToyWorld(navigate_speed=5.0)
        handle = world.apply_action("navigate_to", x=2.0, y=3.0)
        status = handle.poll(timeout=5.0)
        assert status is GoalStatus.SUCCEEDED
        pose = world.get_world_state()["robot"]
        assert abs(pose["x"] - 2.0) < 0.1
        assert abs(pose["y"] - 3.0) < 0.1

    def test_navigate_cancellation_mid_flight(self):
        world = ToyWorld(navigate_speed=0.5)  # slow
        handle = world.apply_action("navigate_to", room="bedroom")
        time.sleep(0.3)
        assert handle.status is GoalStatus.RUNNING
        handle.cancel()
        status = handle.poll(timeout=1.0)
        assert status is GoalStatus.CANCELLED
        pose = world.get_world_state()["robot"]
        # Should not have reached (4, 4)
        assert abs(pose["x"] - 4.0) > 0.1 or abs(pose["y"] - 4.0) > 0.1
