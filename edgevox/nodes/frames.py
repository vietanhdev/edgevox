"""Typed frame dataclasses for the EdgeVox node graph.

Every topic on the bus carries a single frame type. Frames are frozen
dataclasses so they're hashable, picklable, and safe to share across
threads / processes. Naming convention: ``<TopicName>Frame`` for the
default frame type on topic ``<topic_name>``.

Topic conventions (forward-slash, lower_snake_case):

  world/state            WorldStateFrame
  actuator/cmd           ActuatorCmd
  actuator/feedback      ActuatorFeedback
  tool/request           ToolRequest
  tool/result            ToolResult
  agent/thinking         AgentThinking
  agent/response         AgentResponse
  safety/stop            SafetyStop
  user/utterance         UserUtterance
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def _now() -> float:
    return time.monotonic()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# World state -- the canonical "what does the sim look like right now" frame.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldStateFrame:
    """Snapshot of a simulation environment at a moment in time.

    Published by physics nodes (``MujocoArmEnvironment``, ``IrSimEnvironment``,
    ``ToyWorld``) at a fixed rate. Subscribed by visualizer, completion
    checks, recorders, world-state observers.

    The shape mirrors what each env's ``get_world_state()`` already
    returns -- the dataclass is just a typed wrapper. Existing code
    that walks the dict still works during migration.
    """

    timestamp: float = field(default_factory=_now)
    sim: str = ""  # "toyworld" / "irsim" / "mujoco_arm" / "mujoco_humanoid"
    state: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Convenience for code migrating from ``world.get_world_state()``."""
        return self.state.get(key, default)


# ---------------------------------------------------------------------------
# Actuator command + feedback (sim and real-robot common surface).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActuatorCmd:
    """A request to take a physical action in the world.

    Published by the agent / dispatcher. Subscribed by the actuator
    node which calls into the underlying sim / robot driver.

    ``request_id`` lets multiple in-flight commands be distinguished
    in the corresponding :class:`ActuatorFeedback` and
    :class:`ToolResult` frames.
    """

    kind: str  # "grasp" / "move_to" / "release" / "navigate_to" / ...
    args: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=_new_id)
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class ActuatorFeedback:
    """In-flight progress signal from a long-running actuator goal.

    Mirrors ROS2 action feedback: published while a goal executes,
    so subscribers (UIs, dashboards) can render progress. Skill
    feedback (e.g. ``{"phase": "approach", "remaining": 0.13}``)
    flows here.
    """

    request_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=_now)


# ---------------------------------------------------------------------------
# Tool / agent loop frames.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolRequest:
    """The agent wants a tool to fire. Equivalent to ActuatorCmd for
    non-physical tools (``list_objects``, ``locate_object``, etc.)."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=_new_id)
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class ToolResult:
    """A tool finished -- success or failure. Both physical actuators
    and pure tools land here."""

    request_id: str
    name: str
    ok: bool
    result: Any = None
    error: str | None = None
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class AgentThinking:
    """The agent emitted reasoning text for this hop. Subscribed by
    trace loggers, TUI, dashboards."""

    text: str
    hop: int = 0
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class AgentResponse:
    """The agent's user-facing reply for this turn (terminal)."""

    text: str
    elapsed: float = 0.0
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class UserUtterance:
    """A new user input -- from the REPL, STT, or an external test
    harness. Source identifies provenance for safety / audit."""

    text: str
    source: str = "stdin"  # "stdin" / "stt" / "ros2" / "test"
    timestamp: float = field(default_factory=_now)


@dataclass(frozen=True)
class SafetyStop:
    """A safety-priority stop signal. Highest-priority subscribers
    cancel in-flight goals on receipt."""

    reason: str
    triggered_by: str = ""  # "stop_word" / "geofence" / "operator" / ...
    timestamp: float = field(default_factory=_now)


__all__ = [
    "ActuatorCmd",
    "ActuatorFeedback",
    "AgentResponse",
    "AgentThinking",
    "SafetyStop",
    "ToolRequest",
    "ToolResult",
    "UserUtterance",
    "WorldStateFrame",
]
