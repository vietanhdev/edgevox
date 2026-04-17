"""Tier 3 demo — drive an external ROS2 sim / real robot by voice.

Runs against any ROS2 process that publishes ``odom`` (nav_msgs) and
subscribes to ``cmd_vel`` (geometry_msgs/Twist). Tested targets include
Gazebo Harmonic's ``turtlebot3_world``, Nav2 tutorials, and a real
Unitree Go2 over the ROS2 bridge.

Launch::

    # In terminal A — your external sim:
    ros2 launch nav2_bringup tb3_simulation_launch.py use_sim_time:=True

    # In terminal B — the voice agent:
    edgevox-agent robot-external --text-mode
    # optional: --namespace /robot1

Requires a sourced ROS2 workspace (``source /opt/ros/jazzy/setup.bash``).
"""

from __future__ import annotations

import argparse

from edgevox.agents import AgentContext, GoalHandle, skill
from edgevox.examples.agents.framework import AgentApp
from edgevox.llm import tool

EXTERNAL_PERSONA = (
    "You are Ranger, a voice-controlled mobile robot running in an external "
    "ROS2 simulator or on real hardware. You can navigate to a named waypoint "
    "or to explicit x/y coordinates, stop at any time, and report your current "
    "pose. Always reply in one short sentence. Never read raw JSON aloud."
)

# Named waypoints for quick demos. Override or extend via the AgentApp
# pre_run if you have a specific world map.
DEFAULT_WAYPOINTS: dict[str, tuple[float, float]] = {
    "origin": (0.0, 0.0),
    "ahead_1m": (1.0, 0.0),
    "ahead_3m": (3.0, 0.0),
    "left_2m": (0.0, 2.0),
    "right_2m": (0.0, -2.0),
}


@skill(latency_class="slow", timeout_s=60.0)
def navigate_to(location: str, ctx: AgentContext) -> GoalHandle:
    """Drive the robot to a named waypoint.

    Args:
        location: one of origin, ahead_1m, ahead_3m, left_2m, right_2m
            (or whatever the AgentApp's ``waypoints`` override defines).
    """
    waypoints = ctx.deps.waypoints if hasattr(ctx.deps, "waypoints") else DEFAULT_WAYPOINTS
    if location not in waypoints:
        h = GoalHandle()
        h.fail(f"unknown waypoint {location!r}; known: {sorted(waypoints)}")
        return h
    x, y = waypoints[location]
    return ctx.deps.apply_action("navigate_to", x=float(x), y=float(y))


@skill(latency_class="slow", timeout_s=60.0)
def navigate_xy(x: float, y: float, ctx: AgentContext) -> GoalHandle:
    """Drive the robot to explicit ``(x, y)`` coordinates in the map frame.

    Args:
        x: target x coordinate in metres.
        y: target y coordinate in metres.
    """
    return ctx.deps.apply_action("navigate_to", x=float(x), y=float(y))


@skill(latency_class="fast")
def stop(ctx: AgentContext) -> GoalHandle:
    """Stop the robot immediately and cancel the current goal."""
    return ctx.deps.apply_action("stop")


@skill(latency_class="fast")
def get_pose(ctx: AgentContext) -> dict:
    """Report the robot's current ``(x, y, heading_deg)`` pose."""
    return ctx.deps.get_world_state()["pose"]


@tool
def world_state(ctx: AgentContext) -> dict:
    """Full sim/robot snapshot — pose, twist, odom freshness, sensor presence."""
    return ctx.deps.get_world_state()


@tool
def list_waypoints(ctx: AgentContext) -> list[str]:
    """List every named waypoint the robot knows about."""
    waypoints = getattr(ctx.deps, "waypoints", None) or DEFAULT_WAYPOINTS
    return sorted(waypoints)


def _pre_run(args: argparse.Namespace) -> None:
    from edgevox.integrations.sim.ros2_external import ExternalROS2Environment

    env = ExternalROS2Environment(namespace=args.namespace or "")
    # Attach waypoints so the skills + tools can reach them without
    # module-level globals.
    env.waypoints = dict(DEFAULT_WAYPOINTS)  # type: ignore[attr-defined]
    APP.deps = env


APP = AgentApp(
    name="Ranger",
    description="Voice-controlled external ROS2 robot (Gazebo, Isaac, real hardware).",
    instructions=EXTERNAL_PERSONA,
    tools=[world_state, list_waypoints],
    skills=[navigate_to, navigate_xy, stop, get_pose],
    deps=None,
    stop_words=("stop", "halt", "freeze", "abort", "emergency"),
    greeting=("Ranger online. I'm listening for a waypoint — try origin, ahead_1m, or send me to an x/y coordinate."),
    extra_args=[
        (
            ("--namespace",),
            {
                "default": "",
                "help": "ROS2 namespace prefixing odom / cmd_vel (default: root).",
            },
        ),
    ],
    pre_run=_pre_run,
)


def main(argv: list[str] | None = None) -> None:
    APP.run(argv)


if __name__ == "__main__":
    main()
