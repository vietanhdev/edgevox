"""Humanoid in MuJoCo — voice-controlled biped (Tier 2b).

Same agent authoring pattern as ``robot_panda``, but swaps the arm
adapter for :class:`MujocoHumanoidEnvironment` so the demo is a
standing biped that can walk, turn, sit, stand, and wave on voice
commands.

By default the adapter uses a PD standing controller + mocap-style
locomotion stub — see that module's docstring for what that means and
how to plug in a real ONNX walking policy.

Launch::

    edgevox-agent robot-humanoid                 # full TUI voice
    edgevox-agent robot-humanoid --simple-ui     # rich CLI voice
    edgevox-agent robot-humanoid --text-mode     # keyboard chat
    edgevox-agent robot-humanoid --no-render     # headless, tests only

Requires ``pip install 'edgevox[sim-mujoco]'``.
"""

from __future__ import annotations

import argparse

from edgevox.agents import AgentContext, GoalHandle, skill
from edgevox.examples.agents.framework import AgentApp
from edgevox.llm import tool

HUMANOID_PERSONA = (
    "You are G1, a Unitree humanoid robot. You can walk forward or backward "
    "a small distance, turn left or right by a given angle in degrees, and "
    "return to a balanced standing pose. Pick ONE skill per user turn — never "
    "chain. When the user asks where you are, call get_pose. Always reply in "
    "one short sentence. Never read raw JSON aloud."
)


@skill(latency_class="slow", timeout_s=20.0)
def walk_forward(distance: float, ctx: AgentContext) -> GoalHandle:
    """Walk forward by ``distance`` metres (positive values only, 0.1-2.0 m).

    Args:
        distance: metres to walk forward.
    """
    return ctx.deps.apply_action("walk_forward", distance=distance)


@skill(latency_class="slow", timeout_s=20.0)
def walk_backward(distance: float, ctx: AgentContext) -> GoalHandle:
    """Walk backward by ``distance`` metres (positive values only, 0.1-1.0 m).

    Args:
        distance: metres to walk backward.
    """
    return ctx.deps.apply_action("walk_backward", distance=distance)


@skill(latency_class="slow", timeout_s=10.0)
def turn_left(degrees: float, ctx: AgentContext) -> GoalHandle:
    """Turn left (counterclockwise) by ``degrees``.

    Args:
        degrees: turn angle in degrees, typically 15-180.
    """
    return ctx.deps.apply_action("turn_left", degrees=degrees)


@skill(latency_class="slow", timeout_s=10.0)
def turn_right(degrees: float, ctx: AgentContext) -> GoalHandle:
    """Turn right (clockwise) by ``degrees``.

    Args:
        degrees: turn angle in degrees, typically 15-180.
    """
    return ctx.deps.apply_action("turn_right", degrees=degrees)


@skill(latency_class="slow", timeout_s=8.0)
def stand(ctx: AgentContext) -> GoalHandle:
    """Return the humanoid to its home standing pose."""
    return ctx.deps.apply_action("stand")


@skill(latency_class="fast")
def get_pose(ctx: AgentContext) -> dict:
    """Report the robot's current (x, y, heading) and whether it's standing."""
    return ctx.deps.get_world_state()["pose"]


@tool
def world_state(ctx: AgentContext) -> dict:
    """Full sim snapshot — robot kind, pose, current policy, busy flag."""
    return ctx.deps.get_world_state()


def _pre_run(args: argparse.Namespace) -> None:
    from edgevox.integrations.sim.mujoco_humanoid import MujocoHumanoidEnvironment

    # Viewer is on by default; the adapter runs a subprocess probe
    # first and falls back to headless if the local GL stack can't
    # open a GLFW window — no segfault.
    APP.deps = MujocoHumanoidEnvironment(
        model_source=getattr(args, "model_source", "unitree_g1"),
        model_path=getattr(args, "mjcf", None),
        render=not getattr(args, "no_render", False),
    )


APP = AgentApp(
    name="G1",
    description="Voice-controlled Unitree humanoid (G1/H1) running in MuJoCo.",
    instructions=HUMANOID_PERSONA,
    tools=[world_state],
    skills=[walk_forward, walk_backward, turn_left, turn_right, stand, get_pose],
    deps=None,
    stop_words=("stop", "halt", "freeze", "abort", "emergency"),
    greeting=(
        "G1 online. I can walk forward or backward, turn left or right, and "
        "return to a standing pose. What should I do?"
    ),
    extra_args=[
        (("--no-render",), {"action": "store_true", "help": "Run headless (skip the MuJoCo viewer)."}),
        (
            ("--model-source",),
            {
                "dest": "model_source",
                "default": "unitree_g1",
                "choices": ["unitree_g1", "unitree_h1"],
                "help": (
                    "Which Unitree humanoid to load. The model is auto-fetched "
                    "from the EdgeVox HuggingFace mirror on first use (~20 MB)."
                ),
            },
        ),
        (
            ("--mjcf",),
            {
                "default": None,
                "metavar": "PATH",
                "help": "Absolute path to a custom MJCF scene.xml; overrides --model-source.",
            },
        ),
    ],
    pre_run=_pre_run,
)


def main(argv: list[str] | None = None) -> None:
    APP.run(argv)


if __name__ == "__main__":
    main()
