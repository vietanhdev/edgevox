"""Tier 2b humanoid adapter tests — skill round-trips + capability hooks.

Auto-skipped without ``mujoco``. All tests run ``render=False`` so they
work on headless CI without a display.
"""

from __future__ import annotations

import time

import pytest

mujoco = pytest.importorskip("mujoco")

from edgevox.agents.skills import GoalStatus  # noqa: E402
from edgevox.integrations.sim.mujoco_humanoid import MujocoHumanoidEnvironment  # noqa: E402


def _drive_until_terminal(handle, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if handle.status in (GoalStatus.SUCCEEDED, GoalStatus.FAILED, GoalStatus.CANCELLED):
            return
        time.sleep(0.05)


@pytest.fixture
def env():
    # Default model is Unitree G1 — the HF fetch / git fallback handles
    # pulling the assets on first run. Tests auto-skip if the fetch fails
    # (offline CI box).
    try:
        e = MujocoHumanoidEnvironment(render=False)
    except Exception as exc:
        pytest.skip(f"Unitree model fetch failed: {exc}")
    yield e
    e.close()


class TestBasics:
    def test_stands_on_init(self, env):
        state = env.get_world_state()
        assert state["pose"]["standing"], f"humanoid not standing after init: {state}"
        assert state["policy"] in {"stub", "onnx"}

    def test_get_pose_action(self, env):
        h = env.apply_action("get_pose")
        assert h.status is GoalStatus.SUCCEEDED
        assert "x" in h.result
        assert "heading_deg" in h.result

    def test_unknown_action_fails(self, env):
        h = env.apply_action("teleport")
        assert h.status is GoalStatus.FAILED


class TestLocomotion:
    def test_walk_forward_advances_position(self, env):
        x0 = env.get_world_state()["pose"]["x"]
        h = env.apply_action("walk_forward", distance=0.5)
        _drive_until_terminal(h, timeout_s=15.0)
        assert h.status is GoalStatus.SUCCEEDED, f"walk_forward got {h.status}"
        x1 = env.get_world_state()["pose"]["x"]
        assert x1 - x0 > 0.3, f"robot did not walk forward: {x0:.3f} → {x1:.3f}"

    def test_turn_left_changes_heading(self, env):
        h0 = env.get_world_state()["pose"]["heading_deg"]
        h = env.apply_action("turn_left", degrees=45.0)
        _drive_until_terminal(h, timeout_s=10.0)
        assert h.status is GoalStatus.SUCCEEDED, f"turn_left got {h.status}"
        h1 = env.get_world_state()["pose"]["heading_deg"]
        # Heading grew by ~45 deg (allow for tolerance wobble)
        delta = (h1 - h0) % 360
        assert 30 < delta < 70, f"heading change {delta:.1f}° not in [30, 70]"

    def test_walk_backward_retreats(self, env):
        # Walk forward first, then backward — net should be near-zero
        h = env.apply_action("walk_forward", distance=0.3)
        _drive_until_terminal(h, timeout_s=10.0)
        x1 = env.get_world_state()["pose"]["x"]
        h = env.apply_action("walk_backward", distance=0.3)
        _drive_until_terminal(h, timeout_s=10.0)
        x2 = env.get_world_state()["pose"]["x"]
        assert x2 < x1, f"walk_backward did not reduce x: {x1:.3f} → {x2:.3f}"

    def test_mid_walk_cancellation(self, env):
        h = env.apply_action("walk_forward", distance=2.0)
        time.sleep(0.3)
        assert h.status is GoalStatus.RUNNING
        h.cancel()
        time.sleep(0.2)
        assert h.status is GoalStatus.CANCELLED
        # Robot should decelerate to a near-stop. The 29-DoF Unitree
        # humanoid settles physics a little after legs snap back to
        # home, so we allow up to 15 cm of residual drift rather than
        # requiring a hard stop.
        x_a = env.get_world_state()["pose"]["x"]
        time.sleep(0.5)
        x_b = env.get_world_state()["pose"]["x"]
        assert abs(x_b - x_a) < 0.15, f"robot kept moving after cancel: {x_a} → {x_b}"


class TestStand:
    def test_stand_succeeds_from_initial_pose(self, env):
        h = env.apply_action("stand")
        _drive_until_terminal(h, timeout_s=5.0)
        assert h.status is GoalStatus.SUCCEEDED


class TestCapabilityHooks:
    def test_get_pose2d(self, env):
        x, y, theta = env.get_pose2d()
        assert isinstance(x, float) and isinstance(y, float) and isinstance(theta, float)

    def test_get_ee_pose_returns_head(self, env):
        _x, _y, z = env.get_ee_pose()
        assert z > 1.0, f"head too low: z={z:.3f}"

    def test_apply_velocity_cancels_goal(self, env):
        h = env.apply_action("walk_forward", distance=5.0)
        time.sleep(0.2)
        assert h.status is GoalStatus.RUNNING
        env.apply_velocity(0.0, 0.0)
        time.sleep(0.1)
        assert h.status is GoalStatus.CANCELLED


class TestWalkingPolicy:
    def test_pluggable_policy_is_called(self, env):
        import numpy as np

        class StubPolicy:
            called = 0

            def reset(self) -> None:
                self.called = 0

            def step(self, obs, command):
                self.called += 1
                # Return zeros (keeps current ctrl) but pick the actuator
                # count from the model.
                return np.zeros(env._model.nu, dtype=np.float32)

        policy = StubPolicy()
        env.set_walking_policy(policy)
        time.sleep(0.25)
        assert policy.called > 3, f"policy.step never ran (called={policy.called})"
        state = env.get_world_state()
        assert state["policy"] == "onnx"

        env.set_walking_policy(None)
        assert env.get_world_state()["policy"] == "stub"
