"""WorldStatePublisherNode -- adapts an existing sim environment into
a publisher of :class:`~edgevox.nodes.frames.WorldStateFrame` on the
``world/state`` topic.

The existing environment classes (``ToyWorld``, ``IrSimEnvironment``,
``MujocoArmEnvironment``) already have ``get_world_state()`` methods
that return a dict snapshot. This node wraps that method in a polling
loop so the rest of the node graph can subscribe to state changes
instead of polling synchronously.

A ``rate_hz`` of 60 means a state frame every ~16 ms -- fast enough
for visualizer interpolation, slow enough to not flood the bus on
single-machine setups.
"""

from __future__ import annotations

import time
from typing import Any

from edgevox.nodes.frames import WorldStateFrame
from edgevox.nodes.node import Node

DEFAULT_RATE_HZ = 60.0


class WorldStatePublisherNode(Node):
    """Polls an environment's ``get_world_state()`` and publishes
    :class:`WorldStateFrame` to ``world/state`` at a fixed rate.

    Args:
        name: node display name.
        bus: the shared :class:`~edgevox.nodes.bus.Bus` instance.
        env: object exposing ``get_world_state() -> dict``. Typically
            a ``ToyWorld`` / ``IrSimEnvironment`` / ``MujocoArmEnvironment``.
        sim_label: short identifier copied into the frame so
            subscribers can route on sim type. Examples:
            ``"toyworld"``, ``"irsim"``, ``"mujoco_arm"``.
        rate_hz: publish rate. 60 Hz is the visualizer-friendly
            default; bench loops can drop to 10-20 Hz.
        topic: override topic name. Defaults to ``"world/state"``.
    """

    def __init__(
        self,
        name: str,
        bus,
        env: Any,
        *,
        sim_label: str = "",
        rate_hz: float = DEFAULT_RATE_HZ,
        topic: str = "world/state",
    ) -> None:
        super().__init__(name, bus)
        self._env = env
        self._sim_label = sim_label
        self._period_s = 1.0 / max(rate_hz, 1.0)
        self._topic = topic

    def loop(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                state = self._env.get_world_state()
            except Exception:
                # Env tearing down underneath us -- exit cleanly.
                return
            self.publish(
                self._topic,
                WorldStateFrame(sim=self._sim_label, state=state),
            )
            next_tick += self._period_s
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                self._stop.wait(sleep)
            else:
                # Falling behind; resync without piling up.
                next_tick = time.monotonic()


__all__ = ["WorldStatePublisherNode"]
