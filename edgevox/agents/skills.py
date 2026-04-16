"""Skills — cancellable, possibly long-running robot actions.

A **Tool** (``@tool``) is a fast, synchronous, pure function. A **Skill**
(``@skill``) is the robotics-layer equivalent: it returns a
``GoalHandle`` immediately while the real work happens on a worker
thread, and the dispatcher polls + cancels it based on safety signals.

The shape is deliberately modeled on ROS2 Actions: goal → feedback →
result → cancellation. Any ``SimEnvironment`` or ROS2 action client can
back a skill without the agent code noticing.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from queue import Empty, Queue
from typing import TYPE_CHECKING, Any, Literal

from edgevox.llm.tools import Tool as _ToolDescriptor
from edgevox.llm.tools import tool as _tool_decorator

if TYPE_CHECKING:
    from edgevox.agents.base import AgentContext

log = logging.getLogger(__name__)


class GoalStatus(Enum):
    """ROS2-action-style lifecycle of a skill invocation."""

    PENDING = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass
class GoalHandle:
    """Handle to an in-flight skill invocation.

    The dispatcher calls :meth:`poll` in a loop and :meth:`cancel` when
    the pipeline's ``ctx.stop`` fires. Skill implementations write to the
    handle through :meth:`set_feedback`, :meth:`succeed`, :meth:`fail`.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: GoalStatus = GoalStatus.PENDING
    result: Any = None
    error: str | None = None
    _cancel_event: threading.Event = field(default_factory=threading.Event)
    _feedback_q: Queue = field(default_factory=Queue)
    _done_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    # ----- dispatcher-side API -----

    def poll(self, timeout: float | None = None) -> GoalStatus:
        """Wait up to ``timeout`` seconds for the goal to reach a
        terminal state. Returns the current status.

        Returns immediately if already terminal.
        """
        if self.status in (GoalStatus.SUCCEEDED, GoalStatus.CANCELLED, GoalStatus.FAILED):
            return self.status
        self._done_event.wait(timeout=timeout)
        return self.status

    def cancel(self) -> None:
        """Request cancellation. The worker is expected to observe
        ``should_cancel()`` and exit cleanly within ~100 ms."""
        self._cancel_event.set()

    def feedback(self) -> Iterator[Any]:
        """Drain any pending feedback. Non-blocking."""
        while True:
            try:
                yield self._feedback_q.get_nowait()
            except Empty:
                return

    # ----- worker-side API -----

    def should_cancel(self) -> bool:
        return self._cancel_event.is_set()

    def set_feedback(self, fb: Any) -> None:
        self._feedback_q.put(fb)

    def succeed(self, result: Any) -> None:
        self.status = GoalStatus.SUCCEEDED
        self.result = result
        self._done_event.set()

    def fail(self, error: str) -> None:
        self.status = GoalStatus.FAILED
        self.error = error
        self._done_event.set()

    def mark_cancelled(self) -> None:
        self.status = GoalStatus.CANCELLED
        self._done_event.set()


# --------- Skill base ---------


class Skill:
    """A cancellable action. Built by the ``@skill`` decorator.

    Each skill carries:

    - ``name`` / ``description`` — exposed to the LLM like a tool.
    - ``latency_class`` — ``"fast"`` (sub-second, runs inline on the
      dispatcher thread) or ``"slow"`` (runs on a worker thread, can be
      cancelled mid-flight).
    - ``timeout_s`` — how long the dispatcher will wait before
      cancelling.
    - ``_func`` — the underlying Python callable.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        latency_class: Literal["fast", "slow"] = "slow",
        timeout_s: float = 30.0,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.latency_class = latency_class
        self.timeout_s = timeout_s
        self._func = func

    def start(self, ctx: AgentContext, **kwargs: Any) -> GoalHandle:
        """Kick the skill off and return a ``GoalHandle``.

        Two body shapes are supported:

        1. **Delegating body** — the skill function calls into a sim
           backend (``ctx.deps.apply_action(...)``) that itself returns
           a ``GoalHandle``. We adopt that handle directly so there are
           no nested lifecycles. This is the common path for ``@skill``
           functions that thinly wrap a ``SimEnvironment`` action.

        2. **Worker body** — the skill function has a long loop in
           Python. It returns a plain value (or None) when done. We
           run it on a worker thread so the dispatcher can cancel it
           via ``handle.should_cancel()``. Fast skills run inline.
        """

        # Case 1: delegating body. Try the call inline first. If it
        # returns a GoalHandle, that *is* the lifecycle — adopt it.
        # This gives tight sim-to-real parity: the sim's handle is the
        # skill's handle.
        if self.latency_class != "fast":
            try:
                maybe_handle = self._invoke(ctx, GoalHandle(), kwargs)
            except Exception as e:
                log.exception("Skill %r raised during start", self.name)
                failed = GoalHandle()
                failed.fail(f"{type(e).__name__}: {e}")
                return failed

            if isinstance(maybe_handle, GoalHandle):
                # Attach a timeout watchdog to the adopted handle.
                if self.timeout_s and self.timeout_s > 0:
                    self._install_timeout_watchdog(maybe_handle)
                return maybe_handle

            # Body returned a plain value — wrap it as a terminal
            # success handle (the work was actually synchronous).
            handle = GoalHandle()
            handle.succeed(maybe_handle)
            return handle

        # Case 2: fast synchronous skill. Run inline, return terminal.
        handle = GoalHandle()
        handle.status = GoalStatus.RUNNING
        try:
            result = self._invoke(ctx, handle, kwargs)
            if handle.should_cancel():
                handle.mark_cancelled()
            elif handle.status not in (
                GoalStatus.SUCCEEDED,
                GoalStatus.FAILED,
                GoalStatus.CANCELLED,
            ):
                handle.succeed(result)
        except Exception as e:
            log.exception("Skill %r raised", self.name)
            handle.fail(f"{type(e).__name__}: {e}")
        return handle

    def _install_timeout_watchdog(self, handle: GoalHandle) -> None:
        """Background watchdog that cancels a long-running handle once
        ``timeout_s`` elapses without completion."""

        def watchdog() -> None:
            if not handle._done_event.wait(timeout=self.timeout_s):
                handle.cancel()
                time.sleep(0.1)
                if not handle._done_event.is_set():
                    handle.fail(f"timeout after {self.timeout_s:.1f}s")

        threading.Thread(target=watchdog, name=f"skill-wd-{self.name}", daemon=True).start()

    def _invoke(self, ctx: AgentContext, handle: GoalHandle, kwargs: dict[str, Any]) -> Any:
        import inspect

        sig = inspect.signature(self._func)
        call_kwargs = dict(kwargs)
        if "ctx" in sig.parameters:
            call_kwargs["ctx"] = ctx
        if "handle" in sig.parameters:
            call_kwargs["handle"] = handle
        return self._func(**call_kwargs)

    def as_tool_descriptor(self) -> _ToolDescriptor:
        """Return a ``Tool`` descriptor suitable for stuffing into a
        ``ToolRegistry`` so the LLM sees this skill in its JSON schema
        list. The actual call is intercepted by the LLMAgent dispatcher;
        the body here is a no-op fallback for tests."""

        def _fallback(**kwargs: Any) -> str:  # pragma: no cover
            return f"skill {self.name!r} must be invoked via LLMAgent dispatch"

        return _ToolDescriptor(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            func=_fallback,
        )


# --------- @skill decorator ---------


def skill(
    _func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    latency_class: Literal["fast", "slow"] = "slow",
    timeout_s: float = 30.0,
) -> Any:
    """Mark a function as a cancellable :class:`Skill`.

    Reuses the ``@tool`` schema machinery so parameters are auto-derived
    from the function signature + Google-style docstring. The wrapped
    object is a ``Skill`` — not the bare function — because the agent
    dispatcher needs the lifecycle methods.

    Skill authors can optionally declare ``ctx: AgentContext`` and/or
    ``handle: GoalHandle`` in the signature. Both are stripped from the
    JSON schema and injected at call time.
    """

    def wrap(func: Callable[..., Any]) -> Skill:
        # Build parameter schema by reusing the existing @tool machinery.
        # We call the decorator directly, then extract the descriptor
        # and drop the ``ctx`` / ``handle`` parameters from the schema
        # the model sees.
        descriptor_carrier = _tool_decorator(
            name=name or func.__name__,
            description=description or "",
        )(func)
        descriptor: _ToolDescriptor = descriptor_carrier.__edgevox_tool__  # type: ignore[attr-defined]

        params = dict(descriptor.parameters)
        props = dict(params.get("properties", {}))
        required = list(params.get("required", []))
        for hidden in ("ctx", "handle"):
            props.pop(hidden, None)
            if hidden in required:
                required.remove(hidden)
        params["properties"] = props
        if required:
            params["required"] = required
        elif "required" in params:
            del params["required"]

        return Skill(
            func=func,
            name=descriptor.name,
            description=descriptor.description,
            parameters=params,
            latency_class=latency_class,
            timeout_s=timeout_s,
        )

    if _func is not None:
        return wrap(_func)
    return wrap
