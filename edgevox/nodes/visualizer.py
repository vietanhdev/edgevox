"""VisualizerNode -- subscribes to ``world/state`` and renders.

Runs the rendering work entirely independently of the agent loop.
Today the MuJoCo viewer's ``pump_render()`` has to fire on the main
thread; an LLM hop on the same main thread starves the viewer. By
splitting the visualizer into its own node, the LLM blocking
disappears as a render concern -- the viewer pumps at its own rate
regardless of what the agent is doing.

Implementation note: the actual MuJoCo passive viewer still requires
the main thread for OpenGL context. So the VisualizerNode does NOT
spawn a render thread itself -- the *main* thread of the process
calls :meth:`pump` in a tight loop, and the node's subscription
just stashes the latest frame for that pump to pick up.

For sims whose viewers are thread-safe (matplotlib in IR-SIM with
non-interactive backends, ToyWorld which has no viewer), the
visualizer can run :meth:`loop` on its own thread.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from edgevox.nodes.node import Node

log = logging.getLogger(__name__)


class VisualizerNode(Node):
    """Subscribes to ``world/state``; calls a backend's render hook
    with the latest frame.

    Backends:

    - ``"main_thread_pump"``: keeps the latest frame in a slot. The
      caller's main loop polls :meth:`pump_main` and the backend
      reads the slot. Used for MuJoCo's passive viewer.
    - ``"thread_loop"``: runs render calls on the node's own thread
      via :meth:`loop`. Used for matplotlib-style sims where the
      viewer is thread-safe.

    The render callback is supplied by the caller -- the node
    doesn't bind to MuJoCo or matplotlib directly. That keeps the
    node sim-agnostic and lets a future ``BlenderVisualizerNode``
    subscribe to the same topic.
    """

    def __init__(
        self,
        name: str,
        bus,
        render_fn,
        *,
        backend: str = "main_thread_pump",
        topic: str = "world/state",
        rate_hz: float = 30.0,
    ) -> None:
        super().__init__(name, bus)
        if backend not in {"main_thread_pump", "thread_loop"}:
            raise ValueError(f"unknown backend {backend!r}")
        self._render_fn = render_fn
        self._backend = backend
        self._topic = topic
        self._period_s = 1.0 / max(rate_hz, 1.0)
        self._latest: Any = None

    def setup(self) -> None:
        # Subscribe with deliver_latest so a viewer that opens after
        # the sim has been ticking immediately gets the current
        # state on its first frame -- no blank window.
        self.subscribe(self._topic, self._on_state, deliver_latest=True)

    def _on_state(self, frame: Any) -> None:
        # Cheap stash; the actual render work runs on the main
        # thread (pump_main) or this node's loop, depending on backend.
        self._latest = frame

    def pump_main(self) -> None:
        """Main-thread render call. Caller invokes from their event
        loop; the visualizer pulls the latest frame and passes it
        to the render callback. Safe to call at any rate -- frames
        are dropped when the consumer is slower than the publisher,
        which is the right behaviour for a viewer."""
        if self._latest is None:
            return
        try:
            self._render_fn(self._latest)
        except Exception:
            log.exception("VisualizerNode render_fn raised")

    def loop(self) -> None:
        if self._backend != "thread_loop":
            return
        next_tick = time.monotonic()
        while not self._stop.is_set():
            self.pump_main()
            next_tick += self._period_s
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                self._stop.wait(sleep)
            else:
                next_tick = time.monotonic()


__all__ = ["VisualizerNode"]
