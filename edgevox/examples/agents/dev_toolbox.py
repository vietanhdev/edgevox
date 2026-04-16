"""Developer toolbox demo — arithmetic, time, notes, unit conversion.

The simplest of the three examples. Good starting point for copying
into your own project: no shared state beyond a note list.

Launch with:

    edgevox-agent dev
    edgevox-agent dev --text-mode
"""

from __future__ import annotations

from datetime import datetime, timezone

from edgevox.examples.agents.framework import AgentApp
from edgevox.llm import tool

NOTES: list[str] = []


@tool
def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: e.g. "2 * (3 + 4)" — only +, -, *, /, (), numbers allowed
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError("expression contains disallowed characters")
    return float(eval(expression, {"__builtins__": {}}, {}))


@tool
def now_utc() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


@tool
def add_note(text: str) -> str:
    """Save a short note for the user.

    Args:
        text: the note content
    """
    NOTES.append(text)
    return f"saved note #{len(NOTES)}"


@tool
def list_notes() -> list[str]:
    """Return every saved note in insertion order."""
    return list(NOTES)


@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit.

    Args:
        celsius: temperature in Celsius
    """
    return celsius * 9 / 5 + 32


@tool
def km_to_miles(km: float) -> float:
    """Convert kilometers to miles.

    Args:
        km: distance in kilometers
    """
    return km * 0.621371


DEV_TOOLS = [
    calculate,
    now_utc,
    add_note,
    list_notes,
    celsius_to_fahrenheit,
    km_to_miles,
]


APP = AgentApp(
    name="EdgeVox Dev Toolbox",
    description="Quick-start agent with arithmetic, unit conversion, notes, and time.",
    tools=DEV_TOOLS,
    greeting="I can do arithmetic, unit conversions, the current time, and keep notes.",
)


def main(argv: list[str] | None = None) -> None:
    APP.run(argv)


if __name__ == "__main__":
    main()
