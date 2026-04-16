"""Integration tests for the MuJoCo tabletop-arm adapter.

Guarded by ``importorskip`` so the tests auto-skip on systems without
``mujoco`` installed. All tests run headless (``render=False``) so they
work on CI without a display.
"""

from __future__ import annotations

import time

import pytest

mujoco = pytest.importorskip("mujoco")


from edgevox.agents.skills import GoalStatus  # noqa: E402
from edgevox.integrations.sim.mujoco_arm import MujocoArmEnvironment  # noqa: E402


def _drive_until_terminal(env, handle, max_ticks=500):
    for _ in range(max_ticks):
        env.tick_physics()
        if handle.status in (GoalStatus.SUCCEEDED, GoalStatus.FAILED, GoalStatus.CANCELLED):
            return
        time.sleep(0.005)


@pytest.fixture(params=["franka", "gantry"])
def env(request):
    """Both scene variants for basic/structural tests."""
    e = MujocoArmEnvironment(model_source=request.param, render=False)
    yield e
    e.close()


@pytest.fixture
def arm_env():
    """Franka scene only — used for motion-intensive tests that rely on
    tight convergence. The bundled gantry has steady-state gravity
    offset that makes sub-4 cm position tracking unreliable."""
    e = MujocoArmEnvironment(model_source="franka", render=False)
    yield e
    e.close()


class TestBasics:
    def test_loads_scene(self, env):
        state = env.get_world_state()
        assert "ee" in state
        assert "objects" in state
        assert "red_cube" in state["objects"]

    def test_object_names(self, env):
        names = env.object_names()
        assert set(names) == {"red_cube", "green_cube", "blue_cube"}

    def test_get_ee_pose_action(self, env):
        handle = env.apply_action("get_ee_pose")
        assert handle.status is GoalStatus.SUCCEEDED
        assert "x" in handle.result
        assert "y" in handle.result
        assert "z" in handle.result

    def test_list_objects_action(self, env):
        handle = env.apply_action("list_objects")
        assert handle.status is GoalStatus.SUCCEEDED
        assert len(handle.result) == 3
        names = {o["name"] for o in handle.result}
        assert names == {"red_cube", "green_cube", "blue_cube"}

    def test_cubes_on_table(self, env):
        state = env.get_world_state()
        for name in ("red_cube", "green_cube", "blue_cube"):
            obj = state["objects"][name]
            assert obj["z"] > 0.2, f"{name} z={obj['z']}, expected on the table"

    def test_unknown_action_fails(self, env):
        handle = env.apply_action("fly_away")
        assert handle.status is GoalStatus.FAILED
        assert "unknown" in handle.error

    def test_robot_kind(self, env):
        assert env.robot_kind() in ("franka", "gantry")


class TestMoveTo:
    def test_move_to_succeeds(self, arm_env):
        state = arm_env.get_world_state()
        red = state["objects"]["red_cube"]
        handle = arm_env.apply_action("move_to", x=red["x"], y=red["y"], z=red["z"] + 0.1)
        _drive_until_terminal(arm_env, handle)
        assert handle.status is GoalStatus.SUCCEEDED

    def test_mid_flight_cancellation(self, arm_env):
        state = arm_env.get_world_state()
        red = state["objects"]["red_cube"]
        handle = arm_env.apply_action("move_to", x=red["x"], y=red["y"], z=red["z"] + 0.1)
        arm_env.tick_physics()
        arm_env.tick_physics()
        if handle.status is GoalStatus.RUNNING:
            handle.cancel()
            arm_env.tick_physics()
            assert handle.status is GoalStatus.CANCELLED
        else:
            assert handle.status is GoalStatus.SUCCEEDED


class TestGraspRelease:
    def test_grasp_and_release(self, arm_env):
        handle = arm_env.apply_action("grasp", object="red_cube")
        _drive_until_terminal(arm_env, handle)
        assert handle.status is GoalStatus.SUCCEEDED
        assert arm_env.get_world_state()["grasped"] == "red_cube"

        handle = arm_env.apply_action("release")
        _drive_until_terminal(arm_env, handle)
        assert handle.status is GoalStatus.SUCCEEDED
        assert arm_env.get_world_state()["grasped"] is None

    def test_grasp_unknown_object_fails(self, env):
        handle = env.apply_action("grasp", object="gold_cube")
        assert handle.status is GoalStatus.FAILED

    def test_release_without_grasp_fails(self, env):
        handle = env.apply_action("release")
        assert handle.status is GoalStatus.FAILED

    def test_double_grasp_fails(self, arm_env):
        handle = arm_env.apply_action("grasp", object="red_cube")
        _drive_until_terminal(arm_env, handle)
        assert handle.status is GoalStatus.SUCCEEDED
        handle2 = arm_env.apply_action("grasp", object="green_cube")
        assert handle2.status is GoalStatus.FAILED


class TestGotoHome:
    def test_goto_home_succeeds(self, arm_env):
        state = arm_env.get_world_state()
        red = state["objects"]["red_cube"]
        h1 = arm_env.apply_action("move_to", x=red["x"], y=red["y"], z=red["z"] + 0.1)
        _drive_until_terminal(arm_env, h1, max_ticks=800)
        h2 = arm_env.apply_action("goto_home")
        _drive_until_terminal(arm_env, h2, max_ticks=1000)
        assert h2.status is GoalStatus.SUCCEEDED


class TestReset:
    def test_reset_clears_state(self, arm_env):
        h = arm_env.apply_action("grasp", object="blue_cube")
        _drive_until_terminal(arm_env, h)
        arm_env.reset()
        state = arm_env.get_world_state()
        assert state["grasped"] is None
        assert not state["busy"]


class TestPickAndPlace:
    def test_pick_red_place_on_blue(self, arm_env):
        h = arm_env.apply_action("grasp", object="red_cube")
        _drive_until_terminal(arm_env, h, max_ticks=600)
        assert h.status is GoalStatus.SUCCEEDED

        state = arm_env.get_world_state()
        blue = state["objects"]["blue_cube"]
        h = arm_env.apply_action("move_to", x=blue["x"], y=blue["y"], z=blue["z"] + 0.08)
        _drive_until_terminal(arm_env, h, max_ticks=600)
        assert h.status is GoalStatus.SUCCEEDED

        h = arm_env.apply_action("release")
        _drive_until_terminal(arm_env, h)
        assert h.status is GoalStatus.SUCCEEDED

        state = arm_env.get_world_state()
        red_z = state["objects"]["red_cube"]["z"]
        blue_z = state["objects"]["blue_cube"]["z"]
        assert red_z > blue_z - 0.05, "red cube should be near or above blue after placement"
