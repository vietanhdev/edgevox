"""Scout in IR-SIM — voice-controlled robot in a visible simulator.

This is the Tier-1 simulation demo: same agent code as ``robot_scout``,
but swaps ``ToyWorld`` for :class:`IrSimEnvironment` so you actually
see a matplotlib window with a diff-drive robot driving between rooms
in a four-room apartment, sensing the walls with a 2D LiDAR.

Launch:

    edgevox-agent robot-irsim                 # full TUI voice (default)
    edgevox-agent robot-irsim --simple-ui     # lightweight CLI voice
    edgevox-agent robot-irsim --text-mode     # keyboard chat + matplotlib window
    edgevox-agent robot-irsim --no-render     # headless, tests only

Requires ``pip install ir-sim>=2.9`` (or ``pip install 'edgevox[sim]'``).
"""

from __future__ import annotations

import argparse

from edgevox.agents import AgentContext, GoalHandle, skill
from edgevox.examples.agents.framework import AgentApp
from edgevox.llm import tool

SCOUT_PERSONA = (
    "You are Scout, a home robot driving through a four-room apartment "
    "(kitchen, living_room, bedroom, office). You navigate by room name "
    "or by x/y coordinates. When the user asks you to go somewhere, call "
    "navigate_to. When they ask where you are or what your battery level "
    "is, call get_pose or battery_level. Always reply in one short sentence. "
    "Never read raw JSON aloud."
)


@skill(latency_class="slow", timeout_s=60.0)
def navigate_to(room: str, ctx: AgentContext) -> GoalHandle:
    """Drive the robot to a named room.

    Args:
        room: target room — one of kitchen, living_room, bedroom, office, center.
    """
    return ctx.deps.apply_action("navigate_to", room=room)


@skill(latency_class="fast")
def get_pose(ctx: AgentContext) -> dict:
    """Report the robot's current x/y pose and heading in degrees."""
    return ctx.deps.get_world_state()["robot"]


@skill(latency_class="fast")
def battery_level(ctx: AgentContext) -> str:
    """Report the robot's battery level as a percentage."""
    pct = ctx.deps.get_world_state()["robot"]["battery_pct"]
    return f"battery at {pct:.0f}%"


@tool
def list_rooms(ctx: AgentContext) -> list[str]:
    """List every room the robot knows how to drive to."""
    return ctx.deps.room_names()


def _pre_run(args: argparse.Namespace) -> None:
    """Build the IrSimEnvironment lazily so ``--help`` stays fast and
    doesn't require the adapter's deps."""
    from edgevox.integrations.sim.irsim import IrSimEnvironment

    APP.deps = IrSimEnvironment(render=not getattr(args, "no_render", False))


APP = AgentApp(
    name="Scout",
    description="Voice-controlled apartment robot running in IR-SIM.",
    instructions=SCOUT_PERSONA,
    tools=[list_rooms],
    skills=[navigate_to, get_pose, battery_level],
    deps=None,  # attached in pre_run so the matplotlib window only opens on launch
    stop_words=("stop", "halt", "freeze", "abort", "emergency"),
    greeting=(
        "Scout online. I am at the center of a four-room apartment — "
        "kitchen, living room, bedroom, or office. Where would you like me to go?"
    ),
    extra_args=[
        (("--no-render",), {"action": "store_true", "help": "Run IR-SIM headless (no matplotlib window)."}),
    ],
    pre_run=_pre_run,
)


def main(argv: list[str] | None = None) -> None:
    APP.run(argv)


if __name__ == "__main__":
    main()
