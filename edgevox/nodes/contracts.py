"""Cross-sim / cross-robot contracts for the node graph.

The bus only works if every physics layer publishes the same shape
of state and accepts the same shape of command. These Protocols are
the documented contract that ``WorldStatePublisherNode``, future
``ActuatorNode``, and the eventual ``Ros2BridgeNode`` rely on.

We document them as Python Protocols (PEP 544) rather than ABCs so
existing duck-typed sim classes (``ToyWorld``, ``IrSimEnvironment``,
``MujocoArmEnvironment``) satisfy them without inheritance changes.
A real-robot driver for a Franka arm or a Unitree quadruped just has
to expose the same methods.

The skill-name conventions section names the LLM-visible tool surface
that the framework expects across robots. Demos are free to add
extra skills; standard ones should keep these names so an agent
written for one robot transfers cleanly to another.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Environment contract (what physics nodes / sims must expose)
# ---------------------------------------------------------------------------


@runtime_checkable
class SimulationEnvironment(Protocol):
    """Minimum surface a simulation / robot driver must expose for
    the EdgeVox node graph to connect.

    Required:

    - :meth:`get_world_state` -- returns a JSON-serialisable dict
      snapshot. The exact shape is sim-specific but should include
      enough state for downstream verifiers and visualizers to do
      their job: object poses, robot pose, gripper state, and any
      flags (e.g. ``"holding": "red_cube"``).

    Optional:

    - :meth:`pump_render` -- main-thread render call. Sims with
      passive-viewer constraints (MuJoCo) implement this; headless
      sims (``ToyWorld``) leave it as a no-op or omit it.
    - :meth:`close` -- release resources. Called at process shutdown.

    A class satisfies this Protocol structurally -- no inheritance
    required. ``isinstance(env, SimulationEnvironment)`` works at
    runtime via ``runtime_checkable``.
    """

    def get_world_state(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of current world state."""
        ...


# ---------------------------------------------------------------------------
# Canonical skill names
# ---------------------------------------------------------------------------
# These are the LLM-visible tool names that should be used across
# every robot/sim demo for the agent's high-level action surface.
# Keeping them stable means a persona / prompt written against one
# sim transfers to another with no edits.
#
# Manipulation (arm-like robots):
#     list_objects() -> list[{name, position, ...}]
#         Enumerate every visible object in the workspace.
#     locate_object(name: str) -> {position, ...}
#         Return precise pose of a named object.
#     move_to_point(x: float, y: float, z: float)
#         Open-loop end-effector move to a Cartesian point.
#     move_above_object(object: str)
#         Position the gripper above a named object (no descent).
#     grasp(object: str)
#         Close the gripper on a named object.
#     release()
#         Open the gripper.
#
# Mobile (base-driven robots):
#     navigate_to(target: str | tuple)
#         Move the base to a named landmark or coordinate.
#     get_pose() -> {x, y, theta}
#         Read current base pose.
#     scan() -> list[{...}]
#         Single sensor sweep returning observations.
#
# Vocal (voice-only agents):
#     speak(text: str)
#         Emit synthesized speech.
#     listen(timeout: float | None)
#         Capture one user utterance (blocking).
#
# Demos are encouraged to expose extras but should keep these names
# for the actions that map onto them. The ROS2 bridge maps each
# canonical name to a topic / service / action under the
# ``/edgevox/skill/<name>`` namespace -- new names need the bridge
# updated, but existing names work uniformly.

CANONICAL_MANIPULATION_SKILLS = (
    "list_objects",
    "locate_object",
    "move_to_point",
    "move_above_object",
    "grasp",
    "release",
)

CANONICAL_MOBILE_SKILLS = (
    "navigate_to",
    "get_pose",
    "scan",
)

CANONICAL_VOCAL_SKILLS = (
    "speak",
    "listen",
)


__all__ = [
    "CANONICAL_MANIPULATION_SKILLS",
    "CANONICAL_MOBILE_SKILLS",
    "CANONICAL_VOCAL_SKILLS",
    "SimulationEnvironment",
]
