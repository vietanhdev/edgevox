"""Robot commander demo — move, rotate, stop, read sensors.

Stand-in for a real robot. Swap the tool bodies with ROS2 publishers
(see ``edgevox/integrations``) or direct hardware calls and you have a
voice-driven robot. The agent structure doesn't change.

Launch with:

    edgevox-agent robot
    edgevox-agent robot --simple-ui
    edgevox-agent robot --text-mode
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from edgevox.examples.agents.framework import AgentApp
from edgevox.llm import tool

log = logging.getLogger(__name__)


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    heading_deg: float = 0.0
    battery_pct: float = 92.0
    moving: bool = False


ROBOT = RobotState()


@tool
def move_forward(meters: float) -> str:
    """Drive the robot forward.

    Args:
        meters: positive distance in meters, max 5
    """
    if meters <= 0 or meters > 5:
        raise ValueError("meters must be between 0 and 5")
    ROBOT.moving = True
    theta = math.radians(ROBOT.heading_deg)
    ROBOT.x += meters * math.cos(theta)
    ROBOT.y += meters * math.sin(theta)
    ROBOT.battery_pct = max(0.0, ROBOT.battery_pct - 0.5 * meters)
    ROBOT.moving = False
    return f"moved forward {meters}m to ({ROBOT.x:.2f}, {ROBOT.y:.2f})"


@tool
def turn(degrees: float) -> str:
    """Turn the robot in place.

    Args:
        degrees: positive = counter-clockwise, negative = clockwise
    """
    ROBOT.heading_deg = (ROBOT.heading_deg + degrees) % 360
    return f"now facing {ROBOT.heading_deg:.0f} degrees"


@tool
def stop() -> str:
    """Emergency stop — halt any motion immediately."""
    ROBOT.moving = False
    return "stopped"


@tool
def get_pose() -> dict:
    """Return the robot's current pose and heading."""
    return {
        "x": round(ROBOT.x, 2),
        "y": round(ROBOT.y, 2),
        "heading_deg": round(ROBOT.heading_deg, 1),
        "moving": ROBOT.moving,
    }


@tool
def battery_level() -> str:
    """Report the battery level."""
    return f"battery at {ROBOT.battery_pct:.0f}%"


@tool
def go_home() -> str:
    """Drive back to the origin (0, 0) facing 0 degrees."""
    ROBOT.x = 0.0
    ROBOT.y = 0.0
    ROBOT.heading_deg = 0.0
    return "returned to home position"


ROBOT_TOOLS = [move_forward, turn, stop, get_pose, battery_level, go_home]


APP = AgentApp(
    name="EdgeVox Robot Commander",
    description="Voice-controlled robot demo with pose tracking and battery.",
    tools=ROBOT_TOOLS,
    greeting="Robot online. I can move, turn, report pose and battery, or stop on command.",
)


def main(argv: list[str] | None = None) -> None:
    APP.run(argv)


if __name__ == "__main__":
    main()
