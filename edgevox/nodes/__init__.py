"""ROS-shaped internal pub/sub for EdgeVox.

The core idea: every long-lived component in EdgeVox -- physics
simulation, visualizer, actuator dispatcher, agent loop, safety
monitor, recorder -- is a :class:`Node`. Nodes communicate
exclusively via typed frames published to named topics on a shared
:class:`Bus`. There are no direct method calls between nodes.

This shape gives us four properties:

1. **Decoupling.** A new visualizer subscribes to ``world/state`` and
   doesn't touch the physics code. A recorder subscribes to every
   topic and writes them to disk for replay. Neither changes the
   agent.
2. **Threading hygiene.** Nodes own their own loops; the bus
   serialises messages; backpressure is per-subscriber. No more
   "main thread is rendering and the LLM is blocking it" fights.
3. **Test-friendly.** Spin up nodes in isolation, drive them with
   scripted publishes, assert on observed publishes. Same shape
   as ROS2 testing without the ROS2 install.
4. **ROS2 compatible.** A future ``Ros2BridgeNode`` mirrors topics
   between this in-process bus and a real ROS2 graph -- no agent
   code changes when we deploy onto a robot stack.

Design choices:

- Topic names follow ROS conventions: ``namespace/topic``,
  forward-slash separators, lower_snake_case. Examples:
  ``world/state``, ``actuator/cmd``, ``tool/result``,
  ``agent/thinking``, ``safety/stop``.
- Frames are frozen dataclasses (immutable, hashable, picklable).
  See :mod:`edgevox.nodes.frames`.
- The bus is in-process today (one Python process). Cross-process
  / cross-machine comes via the ROS2 bridge or a future shm bus.

Usage::

    from edgevox.nodes import Bus, Node
    from edgevox.nodes.frames import WorldStateFrame

    bus = Bus()

    # A producer node publishes state frames on its own thread.
    class PhysicsNode(Node):
        def loop(self):
            while not self._stop.is_set():
                state = self._sim.snapshot()
                self.bus.publish("world/state", WorldStateFrame(...))
                time.sleep(1/60)

    # A consumer node subscribes and renders.
    class VisualizerNode(Node):
        def setup(self):
            self.bus.subscribe("world/state", self._on_state)

Nodes are started + stopped via lifecycle methods that mirror
ROS2's node lifecycle (configure / activate / deactivate / cleanup).
"""

from edgevox.nodes.bus import Bus, Subscription
from edgevox.nodes.contracts import (
    CANONICAL_MANIPULATION_SKILLS,
    CANONICAL_MOBILE_SKILLS,
    CANONICAL_VOCAL_SKILLS,
    SimulationEnvironment,
)
from edgevox.nodes.node import Node, NodeState

__all__ = [
    "CANONICAL_MANIPULATION_SKILLS",
    "CANONICAL_MOBILE_SKILLS",
    "CANONICAL_VOCAL_SKILLS",
    "Bus",
    "Node",
    "NodeState",
    "SimulationEnvironment",
    "Subscription",
]
