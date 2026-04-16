"""Scout — the voice robot demo.

Shows the full agent framework in action:

- A distinct persona (``instructions=``), not "Vox".
- Cancellable skills via ``@skill`` — ``navigate_to`` runs on a worker
  thread and can be preempted mid-flight.
- ``ToyWorld`` as ``deps``, the stdlib-only reference simulator.
- SafetyMonitor stop-word preemption: saying "stop" during a long
  ``navigate_to`` cancels the goal without consulting the LLM.
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


@skill(latency_class="slow", timeout_s=30.0)
def navigate_to(room: str, ctx: AgentContext) -> GoalHandle:
    """Drive the robot to a named room.

    Args:
        room: target room — one of living_room, kitchen, bedroom, office.
    """
    return ctx.deps.apply_action("navigate_to", room=room)


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
def set_light(room: str, on: bool, ctx: AgentContext) -> str:
    """Turn a room's light on or off.

    Args:
        room: the room name.
        on: true to turn on, false to turn off.
    """
    result = ctx.deps.apply_action("set_light", room=room, on=on)
    if result.error:
        raise ValueError(result.error)
    return f"{room} light is now {'on' if on else 'off'}"


SCOUT_PERSONA = (
    "You are Scout, a concise home robot. You move through rooms, toggle lights, "
    "and report your pose and battery. Every reply is one short sentence. When a "
    "command needs action, call the matching skill or tool; otherwise answer "
    "directly. Never read raw JSON aloud."
)


def build_app() -> AgentApp:
    world = ToyWorld()
    agent = LLMAgent(
        name="Scout",
        description="Voice-controlled home robot with navigation and lights.",
        instructions=SCOUT_PERSONA,
        tools=[list_rooms, set_light],
        skills=[navigate_to, get_pose, battery_level],
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
