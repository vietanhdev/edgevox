"""Thread-safe pub-sub event bus for agent-framework observability.

Every ``AgentEvent`` (tool_call, skill_goal, handoff, safety_preempt,
render_request, etc.) flows through an :class:`EventBus`. Subscribers
register a handler for a specific event ``kind`` (or ``"*"`` for all
events) and receive callbacks as events are published. Publishing is
thread-safe; handler invocation happens on the publisher's thread.

Why a bus and not plain callbacks?

- **One observer can serve many agents** — the TUI chat log, metrics
  collector, and safety monitor all subscribe once and see every event
  from every agent in the program.
- **Decoupled producers/consumers** — an ``LLMAgent`` doesn't know or
  care which sinks consume its events; workflows can inject their own
  subscribers for durations they scope.
- **Main-thread scheduling** — the bus lets a main-thread render loop
  subscribe to ``render_request`` events and pump matplotlib from the
  right thread, even when the producer is on a worker thread.
- **Scalability** — multiple concurrent agent turns all publish to the
  same bus without holding each other up.

The bus is deliberately minimal (no priority queues, no ordering
guarantees across kinds, no backpressure). That's enough for agent-layer
observability; upgrade when a use case demands it.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# A flexible event type — agents publish their domain events into here
# without the bus caring about the concrete class. Consumers do duck
# typing on ``.kind`` / ``.agent_name`` / ``.payload``.
EventLike = Any
Handler = Callable[[EventLike], None]
WILDCARD = "*"


class EventBus:
    """Thread-safe pub-sub bus.

    Handlers registered for a specific kind fire only when an event of
    that kind is published. Handlers registered under :data:`WILDCARD`
    fire for every event.

    Publishing calls each handler synchronously on the publisher's
    thread. Handler exceptions are logged and swallowed so one bad
    sink cannot break the producer. If strict propagation is ever
    needed, subclass and override :meth:`_invoke`.

    The bus can be closed via :meth:`close`; further publishes become
    no-ops. This is used at program shutdown to stop background
    consumers from wedging the interpreter.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.RLock()
        self._closed = False

    # ----- subscription -----

    def subscribe(self, kind: str, handler: Handler) -> Callable[[], None]:
        """Register ``handler`` for events of the given ``kind``.

        Use :data:`WILDCARD` to receive every published event.

        Returns an unsubscribe callable. Calling it removes the handler.
        """
        with self._lock:
            self._subscribers[kind].append(handler)

        def unsubscribe() -> None:
            with self._lock, contextlib.suppress(ValueError):
                self._subscribers[kind].remove(handler)

        return unsubscribe

    def subscribe_all(self, handler: Handler) -> Callable[[], None]:
        """Convenience for :meth:`subscribe(WILDCARD, handler)`."""
        return self.subscribe(WILDCARD, handler)

    # ----- publication -----

    def publish(self, event: EventLike) -> None:
        """Fire an event to every matching handler."""
        with self._lock:
            if self._closed:
                return
            kind = getattr(event, "kind", None)
            handlers: list[Handler] = []
            if kind is not None:
                handlers.extend(self._subscribers.get(kind, ()))
            handlers.extend(self._subscribers.get(WILDCARD, ()))

        for h in handlers:
            self._invoke(h, event)

    def _invoke(self, handler: Handler, event: EventLike) -> None:
        try:
            handler(event)
        except Exception:
            log.exception("EventBus handler raised on %r", getattr(event, "kind", "?"))

    # ----- lifecycle -----

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._subscribers.clear()


# ---------------------------------------------------------------------------
# Main-thread render scheduler
# ---------------------------------------------------------------------------


@dataclass
class RenderRequest:
    """Published by deps that need main-thread GUI ticking (matplotlib
    Tk backend, Qt, etc.). A :class:`MainThreadScheduler` running on
    the main thread consumes these and calls the deps' render method.
    """

    kind: str = "render_request"
    agent_name: str = "scheduler"
    payload: Any = None


class MainThreadScheduler:
    """Tiny main-thread work loop for GUI-bound operations.

    The agent framework produces events on worker threads (pipeline
    loops, skill workers). Things like matplotlib rendering must run
    on the process main thread. This scheduler lets those threads
    enqueue no-arg callables; the main thread drains the queue at a
    fixed rate while *also* pumping any idle-tick callback so GUI
    backends stay responsive even between events.

    Usage (main thread)::

        sched = MainThreadScheduler(idle_tick=env.pump_events)
        sched.attach_to_bus(bus)        # subscribes to render_request
        threading.Thread(target=worker, daemon=True).start()
        sched.run_until(stop_event)     # blocks main thread

    Usage (worker thread)::

        bus.publish(RenderRequest(payload=env.render_once))

    ``payload`` may be a callable; if provided it's what the main
    thread invokes. If not, the scheduler calls ``idle_tick``.
    """

    def __init__(
        self,
        *,
        idle_tick: Callable[[], None] | None = None,
        idle_interval: float = 0.05,
    ) -> None:
        self._idle_tick = idle_tick
        self._idle_interval = idle_interval
        self._q: queue.Queue[Callable[[], None]] = queue.Queue()
        self._unsubscribe: Callable[[], None] | None = None

    def attach_to_bus(self, bus: EventBus) -> None:
        """Subscribe to ``render_request`` events on the given bus."""

        def _on_render(event: EventLike) -> None:
            payload = getattr(event, "payload", None)
            if callable(payload):
                self._q.put(payload)
            elif self._idle_tick is not None:
                self._q.put(self._idle_tick)

        self._unsubscribe = bus.subscribe("render_request", _on_render)

    def enqueue(self, fn: Callable[[], None]) -> None:
        """Direct enqueue from a worker thread (no bus needed)."""
        self._q.put(fn)

    def run_until(self, stop: threading.Event) -> None:
        """Drain the queue + call idle_tick at a fixed rate until
        ``stop`` fires. Must be called on the main thread."""
        while not stop.is_set():
            try:
                fn = self._q.get(timeout=self._idle_interval)
            except queue.Empty:
                if self._idle_tick is not None:
                    try:
                        self._idle_tick()
                    except Exception:
                        log.exception("idle_tick raised")
                continue
            try:
                fn()
            except Exception:
                log.exception("scheduled callable raised")

    def detach(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None


__all__ = [
    "WILDCARD",
    "EventBus",
    "Handler",
    "MainThreadScheduler",
    "RenderRequest",
]
