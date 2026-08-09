"""In-process typed pub/sub bus.

Models ROS2 topic semantics inside one Python process:

- Multiple publishers per topic, multiple subscribers per topic.
- Subscribers get a copy of every message via callback (sync) -- the
  bus does NOT spawn a thread per subscriber. Long-running work
  belongs in :class:`Node.loop`, not in a subscriber callback.
- ``latest(topic)`` returns the most recent message for late-joiners
  (mirrors ROS2 ``transient_local`` durability).
- Topic name is a string; type is enforced by convention (each topic
  carries a single dataclass type). The bus does not type-check.

Why no per-subscriber thread: ROS2's executor model has a thread
pool that fires callbacks. We could replicate that here, but the
common EdgeVox case is "render at viewer rate", "tick physics",
"run agent loop" -- all already on their own threads. Subscriber
callbacks are typically just "stash the latest frame in a slot,"
which doesn't need its own thread.

Backpressure: this bus drops nothing. A slow subscriber blocks
publishers if the lock is held. If you need queue-based
isolation, store the message in a ``queue.Queue`` from your
callback and process it on your own thread.
"""

from __future__ import annotations

import contextlib
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Subscription:
    """Handle returned by :meth:`Bus.subscribe`. Hold it; drop it to
    unsubscribe (or call :meth:`unsubscribe` explicitly)."""

    topic: str
    callback: Callable[[Any], None]
    _bus: Bus

    def unsubscribe(self) -> None:
        self._bus._unsubscribe(self)


class Bus:
    """ROS2-style topic broker, in-process.

    Thread-safe: ``publish`` and ``subscribe`` may be called from
    any thread. Subscriber callbacks fire on the publisher's thread
    (NOT a dedicated dispatch thread) so callbacks must return
    quickly -- typically just stash the frame in a slot.

    To inspect the latest published value at any time, call
    :meth:`latest`. This is the "latched topic" pattern from ROS,
    useful for late-joining subscribers (e.g. a viewer that opens
    after physics has been running for a while).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: dict[str, list[Subscription]] = defaultdict(list)
        self._latest: dict[str, Any] = {}
        # Topic name -> count of messages published. Exposed via
        # :meth:`stats` for observability (test assertions, dashboards).
        self._counts: dict[str, int] = defaultdict(int)

    def publish(self, topic: str, frame: Any) -> None:
        """Publish ``frame`` to ``topic``. Subscribers' callbacks run
        synchronously on the calling thread."""
        with self._lock:
            self._latest[topic] = frame
            self._counts[topic] += 1
            subs = list(self._subs.get(topic, ()))
        # Fire outside the lock to avoid deadlocks when a callback
        # publishes back to the bus (a common pattern).
        for sub in subs:
            try:
                sub.callback(frame)
            except Exception:
                # Swallow subscriber exceptions so one broken
                # subscriber doesn't halt the others. Future: route
                # these to a dead-letter topic for diagnostics.
                import logging

                logging.getLogger(__name__).exception("Subscriber callback raised on topic %r", topic)

    def subscribe(
        self,
        topic: str,
        callback: Callable[[Any], None],
        *,
        deliver_latest: bool = False,
    ) -> Subscription:
        """Register a callback to fire for every future ``frame`` on
        ``topic``. Returns a :class:`Subscription` handle.

        Args:
            topic: ROS-style topic name.
            callback: callable invoked with the frame.
            deliver_latest: if True and a message has already been
                published on this topic, fire the callback once
                immediately with the cached latest value. Mirrors
                ROS2 ``transient_local`` durability.
        """
        sub = Subscription(topic=topic, callback=callback, _bus=self)
        with self._lock:
            self._subs[topic].append(sub)
            cached = self._latest.get(topic) if deliver_latest else None
        if cached is not None:
            try:
                callback(cached)
            except Exception:
                import logging

                logging.getLogger(__name__).exception("deliver_latest callback raised on topic %r", topic)
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        with self._lock, contextlib.suppress(ValueError):
            self._subs[sub.topic].remove(sub)

    def latest(self, topic: str) -> Any:
        """Return the most recent frame published on ``topic``, or
        ``None`` if nothing has been published yet."""
        with self._lock:
            return self._latest.get(topic)

    def topics(self) -> list[str]:
        """Return all topics that currently have at least one message
        cached. Useful for observability."""
        with self._lock:
            return sorted(self._latest.keys())

    def stats(self) -> dict[str, int]:
        """Return per-topic publish counts."""
        with self._lock:
            return dict(self._counts)

    def clear(self) -> None:
        """Drop all subscriptions and cached values. For tests only."""
        with self._lock:
            self._subs.clear()
            self._latest.clear()
            self._counts.clear()


__all__ = ["Bus", "Subscription"]
