"""Scout — the voice robot demo.

Shows the full agent framework in action:

- A distinct persona (``instructions=``), not "Vox".
- Cancellable skills via ``@skill`` — ``navigate_to_room`` runs on a
  worker thread and can be preempted mid-flight.
- ``ToyWorld`` as ``deps``, the stdlib-only reference simulator.
- SafetyMonitor stop-word preemption: saying "stop" during a long
  ``navigate_to_room`` cancels the goal without consulting the LLM.
- Live feedback streaming via ``on_event`` into the TUI / text REPL.

Launch:

    edgevox-agent robot-scout --text-mode
    edgevox-agent robot-scout --simple-ui
    edgevox-agent robot-scout              # full TUI (default)
"""

from __future__ import annotations

from edgevox.agents import AgentContext, GoalHandle, LLMAgent, ToyWorld, skill
from edgevox.examples.agents.framework import AgentApp
from edgevox.llm import tool

SCOUT_PERSONA = (
    "You are Scout, a concise home robot. "
    "To move, call navigate_to_room(room=...); to go back to the starting "
    "room call return_home. To toggle lights call turn_on_light(room=...) "
    "or turn_off_light(room=...) — never both at once. "
    "For queries: list_rooms lists rooms you know, current_room reports which "
    "room you are in, light_state(room) reports whether that room's light is "
    "on, get_pose returns raw x/y/heading, and battery_level returns battery %. "
    "Every reply is one short sentence. Never read raw JSON aloud."
)


@skill(latency_class="slow", timeout_s=30.0)
def navigate_to_room(room: str, ctx: AgentContext) -> GoalHandle:
    """Drive the robot to a named room.

    Args:
        room: target room — one of living_room, kitchen, bedroom, office.
    """
    return ctx.deps.apply_action("navigate_to", room=room)


@skill(latency_class="slow", timeout_s=30.0)
def return_home(ctx: AgentContext) -> GoalHandle:
    """Drive back to the starting room (living_room)."""
    return ctx.deps.apply_action("navigate_to", room="living_room")


@skill(latency_class="fast")
def turn_on_light(room: str, ctx: AgentContext) -> str:
    """Turn on the light in the named room.

    Args:
        room: the room name.
    """
    result = ctx.deps.apply_action("set_light", room=room, on=True)
    if result.error:
        raise ValueError(result.error)
    return f"{room} light is now on"


@skill(latency_class="fast")
def turn_off_light(room: str, ctx: AgentContext) -> str:
    """Turn off the light in the named room.

    Args:
        room: the room name.
    """
    result = ctx.deps.apply_action("set_light", room=room, on=False)
    if result.error:
        raise ValueError(result.error)
    return f"{room} light is now off"


@skill(latency_class="fast")
def get_pose(ctx: AgentContext) -> dict:
    """Report the robot's current pose and whether it is moving."""
    return ctx.deps.get_world_state()["robot"]


@skill(latency_class="fast")
def battery_level(ctx: AgentContext) -> str:
    """Report the battery level as a percentage."""
    pct = ctx.deps.get_world_state()["robot"]["battery_pct"]
    return f"battery at {pct:.0f}%"


@tool
def list_rooms(ctx: AgentContext) -> list[str]:
    """List every room the robot knows how to drive to."""
    return ctx.deps.room_names()


@tool
def current_room(ctx: AgentContext) -> str:
    """Return the name of the room the robot is currently in (nearest waypoint)."""
    state = ctx.deps.get_world_state()
    rx, ry = state["robot"]["x"], state["robot"]["y"]
    rooms: dict[str, dict] = state["rooms"]
    best: tuple[str, float] | None = None
    for name, room in rooms.items():
        dx, dy = rx - float(room["x"]), ry - float(room["y"])
        d = dx * dx + dy * dy
        if best is None or d < best[1]:
            best = (name, d)
    return best[0] if best else "unknown"


@tool
def light_state(room: str, ctx: AgentContext) -> dict:
    """Report whether a given room's light is on.

    Args:
        room: the room name to query.
    """
    rooms: dict[str, dict] = ctx.deps.get_world_state()["rooms"]
    if room not in rooms:
        return {"error": f"unknown room: {room}", "known": sorted(rooms)}
    return {"room": room, "on": bool(rooms[room]["light_on"])}


def build_app() -> AgentApp:
    world = ToyWorld()
    agent = LLMAgent(
        name="Scout",
        description="Voice-controlled home robot with navigation and lights.",
        instructions=SCOUT_PERSONA,
        tools=[list_rooms, current_room, light_state],
        skills=[
            navigate_to_room,
            return_home,
            turn_on_light,
            turn_off_light,
            get_pose,
            battery_level,
        ],
    )
    return AgentApp(
        name="Scout",
        description="Voice-controlled home robot with navigation and lights.",
        agent=agent,
        deps=world,
        greeting="Scout online. Ask me to move between rooms, toggle lights, or check battery.",
        stop_words=("stop", "halt", "freeze", "abort", "emergency"),
    )


APP = build_app()


def main(argv: list[str] | None = None) -> None:
    APP.run(argv)


if __name__ == "__main__":
    main()
