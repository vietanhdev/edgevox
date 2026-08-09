"""Node base class with ROS2-style lifecycle.

A Node owns a thread (or runs on a caller-supplied loop), subscribes
to topics, publishes to topics, and obeys a four-state lifecycle
(unconfigured -> inactive -> active -> finalised). The lifecycle
mirrors ROS2 managed nodes so the same agent code can later be lifted
onto a real ROS2 graph via a bridge.

Subclasses override:

- :meth:`setup` -- subscribe + allocate. Called on ``configure``.
- :meth:`loop` -- the work loop. Called on its own thread when
  ``activate`` is called. Should poll ``self._stop`` regularly.
- :meth:`teardown` -- release resources. Called on ``cleanup``.

Most nodes don't need :meth:`loop` -- subscribe-only nodes (visualizer,
logger, completion-check observer) just register callbacks in
:meth:`setup` and process them when the bus delivers.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from edgevox.nodes.bus import Bus, Subscription

log = logging.getLogger(__name__)


class NodeState(str, Enum):
    """ROS2-aligned lifecycle states."""

    UNCONFIGURED = "unconfigured"
    INACTIVE = "inactive"
    ACTIVE = "active"
    FINALISED = "finalised"


class Node:
    """Base class for an EdgeVox node.

    Subclass + override :meth:`setup` / :meth:`loop` / :meth:`teardown`.
    Lifecycle is driven by :meth:`configure` / :meth:`activate` /
    :meth:`deactivate` / :meth:`cleanup`.

    ``loop`` runs on a daemon thread spawned at activate time and
    joined at deactivate. If a node is purely subscriber-driven
    (no work to poll), leave ``loop`` as the default no-op.
    """

    def __init__(self, name: str, bus: Bus) -> None:
        self.name = name
        self.bus = bus
        self._state = NodeState.UNCONFIGURED
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._subs: list[Subscription] = []

    # ----- lifecycle -----

    def configure(self) -> None:
        """Allocate resources, register subscribers."""
        if self._state != NodeState.UNCONFIGURED:
            raise RuntimeError(f"{self.name}: configure from {self._state.value}")
        self.setup()
        self._state = NodeState.INACTIVE

    def activate(self) -> None:
        """Start the work loop on a daemon thread (if any)."""
        if self._state != NodeState.INACTIVE:
            raise RuntimeError(f"{self.name}: activate from {self._state.value}")
        self._stop.clear()
        if type(self).loop is not Node.loop:
            self._thread = threading.Thread(target=self._run_loop, name=f"node-{self.name}", daemon=True)
            self._thread.start()
        self._state = NodeState.ACTIVE

    def deactivate(self) -> None:
        """Stop the work loop. Subscriptions stay registered."""
        if self._state != NodeState.ACTIVE:
            raise RuntimeError(f"{self.name}: deactivate from {self._state.value}")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._state = NodeState.INACTIVE

    def cleanup(self) -> None:
        """Release resources, drop subscriptions. After cleanup the
        node cannot be reactivated."""
        if self._state == NodeState.ACTIVE:
            self.deactivate()
        for sub in self._subs:
            sub.unsubscribe()
        self._subs.clear()
        self.teardown()
        self._state = NodeState.FINALISED

    # ----- subclass hooks -----

    def setup(self) -> None:
        """Override to register subscriptions / allocate resources.
        Called at ``configure``."""

    def loop(self) -> None:
        """Override for nodes that produce frames on their own
        schedule. Should poll ``self._stop`` regularly. Called on a
        daemon thread at ``activate`` time."""

    def teardown(self) -> None:
        """Override to release resources. Called at ``cleanup``."""

    # ----- helpers -----

    def subscribe(self, topic: str, callback, *, deliver_latest: bool = False):
        """Subscribe with automatic unsubscribe at cleanup."""
        sub = self.bus.subscribe(topic, callback, deliver_latest=deliver_latest)
        self._subs.append(sub)
        return sub

    def publish(self, topic: str, frame) -> None:
        """Publish to the bus."""
        self.bus.publish(topic, frame)

    @property
    def state(self) -> NodeState:
        return self._state

    def _run_loop(self) -> None:
        try:
            self.loop()
        except Exception:
            log.exception("Node %r loop raised", self.name)


__all__ = ["Node", "NodeState"]
