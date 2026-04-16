"""Integration tests for the IR-SIM adapter.

Guarded by ``importorskip`` so the tests auto-skip on systems without
``ir-sim`` installed.
"""

from __future__ import annotations

import time

import pytest

irsim = pytest.importorskip("irsim")


from edgevox.agents.skills import GoalStatus  # noqa: E402
from edgevox.integrations.sim.irsim import IrSimEnvironment  # noqa: E402


def _drive_until_terminal(env, handle, max_ticks=600):
    """Drive the sim forward one tick at a time until the handle
    reaches a terminal state. Rendering is disabled so this works
    headless on CI.
    """
    for _ in range(max_ticks):
        env.tick_main()
        if handle.status in (GoalStatus.SUCCEEDED, GoalStatus.FAILED, GoalStatus.CANCELLED):
            return
        time.sleep(0.005)


@pytest.fixture
def env():
    e = IrSimEnvironment(render=False)
    yield e
    e.close()


class TestIrSimEnvironment:
    def test_loads_bundled_world(self, env):
        state = env.get_world_state()
        assert "robot" in state
        assert "waypoints" in state
        assert "kitchen" in state["waypoints"]

    def test_rooms_match_waypoints(self, env):
        rooms = env.room_names()
        assert "kitchen" in rooms
        assert "office" in rooms
        assert "bedroom" in rooms
        assert "living_room" in rooms

    def test_initial_pose(self, env):
        state = env.get_world_state()["robot"]
        assert abs(state["x"] - 5.0) < 0.1
        assert abs(state["y"] - 5.0) < 0.1

    def test_get_pose_action(self, env):
        handle = env.apply_action("get_pose")
        assert handle.status is GoalStatus.SUCCEEDED
        assert "x" in handle.result
        assert "y" in handle.result

    def test_battery_level_action(self, env):
        handle = env.apply_action("battery_level")
        assert handle.status is GoalStatus.SUCCEEDED
        assert handle.result["battery_pct"] > 0

    def test_list_rooms_action(self, env):
        handle = env.apply_action("list_rooms")
        assert handle.status is GoalStatus.SUCCEEDED
        assert set(handle.result) == set(env.room_names())


class TestIrSimNavigation:
    def test_navigate_to_kitchen(self, env):
        handle = env.apply_action("navigate_to", room="kitchen")
        _drive_until_terminal(env, handle)
        assert handle.status is GoalStatus.SUCCEEDED
        pose = env.get_world_state()["robot"]
        assert abs(pose["x"] - 8.0) < 0.5
        assert abs(pose["y"] - 2.0) < 0.5

    def test_navigate_unknown_room_fails(self, env):
        handle = env.apply_action("navigate_to", room="garage")
        assert handle.status is GoalStatus.FAILED

    def test_navigate_without_target_fails(self, env):
        handle = env.apply_action("navigate_to")
        assert handle.status is GoalStatus.FAILED

    def test_mid_flight_cancellation(self, env):
        handle = env.apply_action("navigate_to", room="bedroom")
        for _ in range(20):
            env.tick_main()
        assert handle.status is GoalStatus.RUNNING
        handle.cancel()
        env.tick_main()
        assert handle.status is GoalStatus.CANCELLED
        pose = env.get_world_state()["robot"]
        reached = abs(pose["x"] - 8.0) < 0.5 and abs(pose["y"] - 8.0) < 0.5
        assert not reached

    def test_sequential_navigations(self, env):
        handle = env.apply_action("navigate_to", room="kitchen")
        _drive_until_terminal(env, handle)
        assert handle.status is GoalStatus.SUCCEEDED

        handle2 = env.apply_action("navigate_to", room="office")
        _drive_until_terminal(env, handle2)
        assert handle2.status is GoalStatus.SUCCEEDED
        pose = env.get_world_state()["robot"]
        assert abs(pose["x"] - 2.0) < 0.5
        assert abs(pose["y"] - 8.0) < 0.5
