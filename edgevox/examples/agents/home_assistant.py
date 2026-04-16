"""Home assistant demo — lights, thermostat, timers, weather stub.

Launch with any of:

    edgevox-agent home                 # full TUI voice (default)
    edgevox-agent home --simple-ui     # lightweight CLI voice
    edgevox-agent home --text-mode     # keyboard chat
    edgevox-agent home --model hf:unsloth/Qwen2.5-3B-Instruct-GGUF:Qwen2.5-3B-Instruct-Q4_K_M.gguf

State lives in a tiny in-memory ``House`` so you can see the agent's
effects without real smart-home hardware. Replace tool bodies with calls
to Home Assistant, MQTT, Matter, etc. and the rest stays the same.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from edgevox.examples.agents.framework import AgentApp
from edgevox.llm import tool


@dataclass
class House:
    lights: dict[str, bool] = field(default_factory=lambda: {"living_room": False, "kitchen": False, "bedroom": False})
    thermostat_c: float = 21.0
    timers: list[tuple[str, float]] = field(default_factory=list)


STATE = House()


@tool
def list_rooms() -> list[str]:
    """List every room that has a controllable light."""
    return sorted(STATE.lights)


@tool
def set_light(room: str, on: bool) -> str:
    """Turn a room's light on or off.

    Args:
        room: the room name, e.g. "kitchen"
        on: true to turn on, false to turn off
    """
    if room not in STATE.lights:
        raise ValueError(f"unknown room {room!r}. Known rooms: {sorted(STATE.lights)}")
    STATE.lights[room] = on
    return f"{room} light is now {'on' if on else 'off'}"


@tool
def get_light_status(room: str) -> str:
    """Check whether a specific room's light is on.

    Args:
        room: the room name
    """
    if room not in STATE.lights:
        raise ValueError(f"unknown room {room!r}")
    return f"{room}: {'on' if STATE.lights[room] else 'off'}"


@tool
def set_thermostat(celsius: float) -> str:
    """Set the thermostat target temperature in Celsius.

    Args:
        celsius: target temperature, must be between 10 and 30
    """
    if not 10 <= celsius <= 30:
        raise ValueError("temperature must be between 10 and 30 C")
    STATE.thermostat_c = celsius
    return f"thermostat set to {celsius:.1f}C"


@tool
def get_thermostat() -> str:
    """Read the current thermostat set-point."""
    return f"thermostat is {STATE.thermostat_c:.1f}C"


@tool
def start_timer(label: str, seconds: int) -> str:
    """Start a kitchen timer.

    Args:
        label: what the timer is for, e.g. "pasta"
        seconds: duration in seconds
    """
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    STATE.timers.append((label, time.time() + seconds))
    return f"timer '{label}' set for {seconds}s"


@tool
def get_weather(city: str) -> dict:
    """Return a (stubbed) weather report for a city.

    Args:
        city: city name
    """
    rng = random.Random(city)  # deterministic stub
    return {
        "city": city,
        "temp_c": round(rng.uniform(5, 28), 1),
        "conditions": rng.choice(["sunny", "cloudy", "light rain", "clear"]),
    }


HOME_TOOLS = [
    list_rooms,
    set_light,
    get_light_status,
    set_thermostat,
    get_thermostat,
    start_timer,
    get_weather,
]


APP = AgentApp(
    name="EdgeVox Home Assistant",
    description="Voice agent that controls lights, thermostat, timers, and fetches weather.",
    tools=HOME_TOOLS,
    greeting="Hi — I can control lights, thermostat, timers, and fetch weather. What do you need?",
)


def main(argv: list[str] | None = None) -> None:
    APP.run(argv)


if __name__ == "__main__":
    main()
