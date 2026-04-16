"""EdgeVox agent framework.

First-class Agent abstractions layered on top of the existing tool system:

- ``Agent`` — polymorphic protocol every agent and workflow implements.
- ``LLMAgent`` — concrete agent backed by an ``edgevox.llm.LLM``.
- ``Session``, ``AgentContext``, ``AgentEvent``, ``AgentResult`` — runtime
  plumbing and observability.
- ``Handoff`` — sentinel return value that transfers control between
  agents (OpenAI-Agents-SDK-style).
- ``Skill``, ``GoalHandle``, ``@skill`` — cancellable robot actions
  distinct from pure ``@tool`` functions.
- ``SimEnvironment``, ``ToyWorld`` — simulation protocol + stdlib-only
  reference env used for examples and tests.
- Workflows: ``Sequence``, ``Fallback``, ``Loop``, ``Router``, ``Retry``,
  ``Timeout`` — Behaviour-Tree-shaped composition of agents.
"""

from edgevox.agents.base import (
    Agent,
    AgentContext,
    AgentEvent,
    AgentResult,
    Handoff,
    LLMAgent,
    Session,
)
from edgevox.agents.bus import EventBus, MainThreadScheduler, RenderRequest
from edgevox.agents.sim import SimEnvironment, ToyWorld
from edgevox.agents.skills import GoalHandle, GoalStatus, Skill, skill
from edgevox.agents.workflow import (
    Fallback,
    Loop,
    Parallel,
    Retry,
    Router,
    Sequence,
    Timeout,
)

__all__ = [
    "Agent",
    "AgentContext",
    "AgentEvent",
    "AgentResult",
    "EventBus",
    "Fallback",
    "GoalHandle",
    "GoalStatus",
    "Handoff",
    "LLMAgent",
    "Loop",
    "MainThreadScheduler",
    "Parallel",
    "RenderRequest",
    "Retry",
    "Router",
    "Sequence",
    "Session",
    "SimEnvironment",
    "Skill",
    "Timeout",
    "ToyWorld",
    "skill",
]
