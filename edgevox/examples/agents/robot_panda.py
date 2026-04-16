"""Panda in MuJoCo — voice-controlled tabletop pick-and-place.

This is the Tier-2a simulation demo: a Franka Emika Panda (or the
bundled XYZ gantry fallback) drives a 3D arm to pick, move, and place
coloured cubes on a table in a MuJoCo viewer window.

Launch:

    edgevox-agent robot-panda                 # full TUI voice (default)
    edgevox-agent robot-panda --simple-ui     # lightweight CLI voice
    edgevox-agent robot-panda --text-mode     # keyboard chat + MuJoCo viewer
    edgevox-agent robot-panda --no-render     # headless, tests only
    edgevox-agent robot-panda --gantry        # force the bundled gantry fallback

Requires ``pip install 'edgevox[sim-mujoco]'`` (or ``pip install 'mujoco>=3.2'``).
The Franka scene is auto-fetched from HuggingFace Hub on first run (~33 MB).
"""

from __future__ import annotations

import argparse
from typing import Any

from edgevox.agents import AgentContext, GoalHandle, skill
from edgevox.examples.agents.framework import AgentApp
from edgevox.llm import tool

PANDA_PERSONA = (
    "You are Panda, a terse pick-and-place robot arm on a tabletop. "
    "The table has three coloured cubes: red_cube, green_cube, blue_cube. "
    "You can move to an x/y/z position, grasp a named cube, release it, "
    "list objects and their positions, or return home. "
    "When the user asks you to pick something up, call grasp(object). "
    "To place it somewhere, move_to the target position then release. "
    "Always reply in one short sentence. Never read raw JSON aloud."
)


@skill(latency_class="slow", timeout_s=30.0)
def move_to(x: float, y: float, z: float, ctx: AgentContext) -> GoalHandle:
    """Move the end-effector to an x/y/z position in metres.

    Args:
        x: target x position.
        y: target y position.
        z: target z position.
    """
    return ctx.deps.apply_action("move_to", x=x, y=y, z=z)


@skill(latency_class="slow", timeout_s=30.0)
def grasp(object: str, ctx: AgentContext) -> GoalHandle:
    """Approach and grasp a named cube on the table.

    Args:
        object: name of the cube to grasp (red_cube, green_cube, or blue_cube).
    """
    return ctx.deps.apply_action("grasp", object=object)


@skill(latency_class="slow", timeout_s=15.0)
def release(ctx: AgentContext) -> GoalHandle:
    """Open the gripper and release the currently held object."""
    return ctx.deps.apply_action("release")


@skill(latency_class="slow", timeout_s=30.0)
def goto_home(ctx: AgentContext) -> GoalHandle:
    """Return the arm to its home (ready) position."""
    return ctx.deps.apply_action("goto_home")


@skill(latency_class="fast")
def get_ee_pose(ctx: AgentContext) -> dict:
    """Report the end-effector's current x/y/z position."""
    h = ctx.deps.apply_action("get_ee_pose")
    return h.result


@tool
def list_objects(ctx: AgentContext) -> list[dict[str, Any]]:
    """List every object on the table with its current x/y/z position."""
    h = ctx.deps.apply_action("list_objects")
    return h.result


def _pre_run(args: argparse.Namespace) -> None:
    from edgevox.integrations.sim.mujoco_arm import MujocoArmEnvironment

    source = "gantry" if getattr(args, "gantry", False) else "franka"
    APP.deps = MujocoArmEnvironment(
        model_source=source,
        render=not getattr(args, "no_render", False),
    )


APP = AgentApp(
    name="Panda",
    description="Voice-controlled tabletop arm running in MuJoCo.",
    instructions=PANDA_PERSONA,
    tools=[list_objects],
    skills=[move_to, grasp, release, goto_home, get_ee_pose],
    deps=None,
    stop_words=("stop", "halt", "freeze", "abort", "emergency"),
    greeting=(
        "Panda online. I see three cubes on the table — red, green, and blue. What would you like me to pick up?"
    ),
    extra_args=[
        (("--no-render",), {"action": "store_true", "help": "Run headless (no MuJoCo viewer)."}),
        (("--gantry",), {"action": "store_true", "help": "Use the bundled gantry arm instead of Franka."}),
    ],
    pre_run=_pre_run,
)


def main(argv: list[str] | None = None) -> None:
    APP.run(argv)


if __name__ == "__main__":
    main()
