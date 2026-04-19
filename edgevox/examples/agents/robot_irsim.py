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
import math

from edgevox.agents import AgentContext, GoalHandle, skill
from edgevox.examples.agents.framework import AgentApp
from edgevox.llm import tool

SCOUT_PERSONA = (
    "You are Scout, a home robot driving through a four-room apartment "
    "(kitchen, living_room, bedroom, office). "
    "Drive by room name with navigate_to_room(room=...); drive by coordinates "
    "with navigate_to_point(x=..., y=...). Call stop to halt mid-motion, or "
    "return_home to drive back to the apartment centre. "
    "Call current_room when the user asks where you are, describe_room(room) "
    "for a short summary of a room, get_pose for raw x/y/heading, and "
    "battery_level for battery %. Use list_rooms only when the user asks what "
    "rooms you know. "
    "Always reply in one short sentence. Never read raw JSON aloud."
)


@skill(latency_class="slow", timeout_s=60.0)
def navigate_to_room(room: str, ctx: AgentContext) -> GoalHandle:
    """Drive the robot to a named room by waypoint name.

    Args:
        room: target room — one of kitchen, living_room, bedroom, office, center.
    """
    return ctx.deps.apply_action("navigate_to", room=room)


@skill(latency_class="slow", timeout_s=60.0)
def navigate_to_point(x: float, y: float, ctx: AgentContext) -> GoalHandle:
    """Drive the robot to a specific x/y position in metres.

    Args:
        x: target x coordinate in metres.
        y: target y coordinate in metres.
    """
    return ctx.deps.apply_action("navigate_to", x=x, y=y)


@skill(latency_class="fast")
def stop(ctx: AgentContext) -> str:
    """Halt any active motion immediately."""
    ctx.deps.apply_velocity(0.0, 0.0)
    return "stopped"


@skill(latency_class="slow", timeout_s=60.0)
def return_home(ctx: AgentContext) -> GoalHandle:
    """Drive back to the centre of the apartment (the home waypoint)."""
    return ctx.deps.apply_action("navigate_to", room="center")


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


@tool
def current_room(ctx: AgentContext) -> str:
    """Return the name of the waypoint closest to the robot's current pose."""
    state = ctx.deps.get_world_state()
    rx, ry = state["robot"]["x"], state["robot"]["y"]
    best: tuple[str, float] | None = None
    for name, (wx, wy) in state["waypoints"].items():
        d = math.hypot(rx - wx, ry - wy)
        if best is None or d < best[1]:
            best = (name, d)
    return best[0] if best else "unknown"


@tool
def describe_room(room: str, ctx: AgentContext) -> dict:
    """Describe a named room — its waypoint coordinates on the floor plan.

    Args:
        room: the room name to describe.
    """
    state = ctx.deps.get_world_state()
    waypoints: dict[str, tuple[float, float]] = state["waypoints"]
    if room not in waypoints:
        return {"error": f"unknown room: {room}", "known": sorted(waypoints)}
    x, y = waypoints[room]
    return {"room": room, "x": round(float(x), 2), "y": round(float(y), 2)}


def _pre_run(args: argparse.Namespace) -> None:
    """Build the IrSimEnvironment lazily so ``--help`` stays fast and
    doesn't require the adapter's deps."""
    from edgevox.integrations.sim.irsim import IrSimEnvironment

    APP.deps = IrSimEnvironment(render=not getattr(args, "no_render", False))


APP = AgentApp(
    name="Scout",
    description="Voice-controlled apartment robot running in IR-SIM.",
    instructions=SCOUT_PERSONA,
    tools=[list_rooms, current_room, describe_room],
    skills=[navigate_to_room, navigate_to_point, stop, return_home, get_pose, battery_level],
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
