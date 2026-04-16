"""Agent foundations: Session, AgentContext, AgentEvent, Handoff, LLMAgent.

The design points inherited from the plan (``docs/plan.md``):

- ``Agent`` is a minimal Protocol so workflows can polymorphically drive
  any mix of LLM-backed agents and composite workflows.
- ``LLMAgent`` wraps an ``edgevox.llm.LLM`` and swaps its history, tools,
  and persona per run so multiple agents can share one GGUF (memory
  budget is the hard constraint on edge devices).
- Delegation follows the **OpenAI Agents SDK "handoff-as-return-value"**
  pattern: a synthetic handoff tool returns a ``Handoff`` sentinel, and
  the agent loop short-circuits, transferring control to the target
  agent without another LLM hop on the router. That's 2 LLM calls
  (router + specialist) instead of smolagents-style 3.
- ``AgentContext.stop`` is a ``threading.Event`` the pipeline can set to
  preempt in-flight skills; the agent loop checks it between hops.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from edgevox.agents.bus import EventBus
from edgevox.llm.tools import Tool, ToolCallResult, ToolRegistry
from edgevox.llm.tools import tool as tool_decorator

if TYPE_CHECKING:
    from edgevox.agents.skills import GoalHandle, Skill
    from edgevox.llm.llamacpp import LLM

log = logging.getLogger(__name__)


# --------- Session / events / results ---------


@dataclass
class Session:
    """Per-run conversation state — messages and a scratchpad dict.

    Replaces raw ``LLM._history`` access so tests and workflows can
    inspect and mutate state without reaching into the LLM internals.
    """

    messages: list[dict] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear the conversation history and the scratchpad."""
        self.messages.clear()
        self.state.clear()


@dataclass
class AgentEvent:
    """Unified observability. The TUI chat log subscribes and renders
    each kind with its own styling.
    """

    kind: Literal[
        "agent_start",
        "agent_end",
        "tool_call",
        "skill_goal",
        "skill_feedback",
        "skill_cancelled",
        "handoff",
        "safety_preempt",
    ]
    agent_name: str
    payload: Any = None


EventCallback = Callable[[AgentEvent], None]


@dataclass
class AgentContext:
    """Dependency-injection + runtime plumbing passed through a run.

    - ``deps`` is the user-supplied dependency object — a ``ToyWorld``,
      ``SimEnvironment``, ROS2 Node, etc.
    - ``bus`` is the :class:`EventBus` every agent in this turn
      publishes to. Subscribers render UI, drive main-thread GUIs,
      collect metrics. Thread-safe.
    - ``on_event`` is kept for back-compat; if set, it's registered
      as a wildcard subscriber during ``__post_init__``.
    - ``stop`` is the safety-preempt ``threading.Event``. Skills poll
      it between feedbacks; the agent loop checks between tool hops.
    """

    session: Session = field(default_factory=Session)
    deps: Any = None
    bus: EventBus = field(default_factory=EventBus)
    on_event: EventCallback | None = None
    stop: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        if self.on_event is not None:
            self.bus.subscribe_all(self.on_event)

    def emit(self, kind: str, agent_name: str, payload: Any = None) -> None:
        """Publish an :class:`AgentEvent` to the bus."""
        self.bus.publish(
            AgentEvent(kind=kind, agent_name=agent_name, payload=payload)  # type: ignore[arg-type]
        )


@dataclass
class AgentResult:
    """Outcome of a single ``Agent.run()`` call."""

    reply: str
    agent_name: str
    elapsed: float = 0.0
    tool_calls: list[ToolCallResult] = field(default_factory=list)
    skill_goals: list[GoalHandle] = field(default_factory=list)
    handed_off_to: str | None = None
    preempted: bool = False


# --------- Handoff ---------


@dataclass
class Handoff:
    """Sentinel return value from a handoff tool.

    When ``LLMAgent._run_agent`` sees a tool whose result is a ``Handoff``,
    it stops calling the current LLM and invokes ``target.run(task, ctx)``.
    ``task`` defaults to the original user task if omitted — the common
    case for a Router.
    """

    target: Agent
    task: str | None = None
    reason: str = ""


# --------- Agent protocol ---------


@runtime_checkable
class Agent(Protocol):
    """Minimal polymorphic interface. Workflows accept this type."""

    name: str
    description: str

    def run(self, task: str, ctx: AgentContext) -> AgentResult: ...
    def run_stream(self, task: str, ctx: AgentContext) -> Iterator[str]: ...


# --------- LLMAgent ---------


ToolsArg = Iterable[Callable[..., object] | Tool] | ToolRegistry | None
SkillsArg = Iterable["Skill"] | None


class LLMAgent:
    """Concrete LLM-backed agent.

    The LLM itself may be shared across agents: pass ``llm=None`` here
    and inject a single ``LLM`` before the first ``run()`` via
    :meth:`bind_llm`. Workflows and ``AgentApp`` handle this for the
    user so examples never need to think about it.
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        *,
        tools: ToolsArg = None,
        skills: SkillsArg = None,
        llm: LLM | None = None,
        handoffs: list[Agent] | None = None,
        max_tool_hops: int = 3,
    ) -> None:
        self.name = name
        self.description = description
        self.instructions = instructions
        self._llm: LLM | None = llm
        self._max_tool_hops = max_tool_hops

        # Skills are stored alongside tools. We synthesize a synthetic
        # @tool for each skill and handoff target so the model sees a
        # single flat tool list. Dispatch then routes each call based on
        # its source (real tool / skill / handoff).
        self._tool_registry: ToolRegistry = ToolRegistry()
        self._skill_registry: dict[str, Skill] = {}
        self._handoff_registry: dict[str, Agent] = {}

        if tools is not None:
            self._register_tools(tools)
        if skills is not None:
            self._register_skills(skills)
        if handoffs:
            self._register_handoffs(handoffs)

    # ----- registration helpers -----

    def _register_tools(self, tools: ToolsArg) -> None:
        if isinstance(tools, ToolRegistry):
            for t in tools:
                self._tool_registry.register(t)
            return
        for t in tools:  # type: ignore[union-attr]
            self._tool_registry.register(t)

    def _register_skills(self, skills: SkillsArg) -> None:
        from edgevox.agents.skills import Skill as SkillBase

        for s in skills or []:
            if not isinstance(s, SkillBase):
                raise TypeError(f"{s!r} is not a Skill — did you forget @skill?")
            if s.name in self._skill_registry:
                log.warning("Skill %r already registered — overwriting.", s.name)
            self._skill_registry[s.name] = s
            # Synthesize a tool descriptor the model can call. Dispatch
            # routes to the skill registry by name, never to the tool.
            self._tool_registry.tools[s.name] = s.as_tool_descriptor()

    def _register_handoffs(self, handoffs: list[Agent]) -> None:
        for agent in handoffs:
            tool_name = f"handoff_to_{agent.name}"
            if tool_name in self._handoff_registry:
                log.warning("Handoff target %r already registered — overwriting.", agent.name)
            self._handoff_registry[tool_name] = agent
            synthetic = _make_handoff_tool(agent)
            self._tool_registry.register(synthetic)

    def bind_llm(self, llm: LLM) -> None:
        """Attach a shared ``LLM`` instance.

        Workflows call this before running a sub-agent so multiple agents
        share one GGUF in memory. Can be called repeatedly.
        """
        self._llm = llm

    @property
    def llm(self) -> LLM | None:
        return self._llm

    @property
    def tools(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def skills(self) -> dict[str, Skill]:
        return dict(self._skill_registry)

    # ----- the run() path -----

    def _ensure_llm(self) -> LLM:
        if self._llm is None:
            raise RuntimeError(
                f"LLMAgent {self.name!r} has no LLM bound. "
                "Either construct with llm=<instance> or call bind_llm() "
                "before run(). When using AgentApp, the framework does this "
                "for you."
            )
        return self._llm

    def _build_system_message(self) -> dict:
        """Build the system prompt for this agent. Each run() builds a
        fresh one so multiple agents sharing an LLM don't clobber each
        other's persona."""
        llm = self._ensure_llm()
        from edgevox.llm.llamacpp import get_system_prompt  # lazy

        system = get_system_prompt(
            language=llm._language,
            has_tools=bool(self._tool_registry),
            persona=self.instructions,
        )
        return {"role": "system", "content": system}

    def run(self, task: str, ctx: AgentContext | None = None) -> AgentResult:
        """Run one turn. Returns an :class:`AgentResult`.

        On handoff, returns the *target's* result but annotates
        ``handed_off_to`` so the caller can trace delegation.
        """
        ctx = ctx or AgentContext()
        ctx.emit("agent_start", self.name, {"task": task})

        if ctx.stop.is_set():
            ctx.emit("safety_preempt", self.name, "stopped before start")
            return AgentResult(
                reply="Stopped.",
                agent_name=self.name,
                preempted=True,
            )

        llm = self._ensure_llm()

        # History isolation — each run() owns its own messages list.
        # Seeds from the caller's Session if any, otherwise starts
        # fresh. This is what makes concurrent LLMAgent.run() calls
        # thread-safe: they never touch each other's history.
        if ctx.session.messages:
            messages = list(ctx.session.messages)
            # Ensure first message is this agent's system prompt — it
            # may have been installed by a different agent in the same
            # session previously.
            messages[0] = self._build_system_message()
        else:
            messages = [self._build_system_message()]

        t0 = time.perf_counter()
        reply, handoff, captured_tools = self._drive(llm, messages, task, ctx)
        # Persist the updated messages back into the session so the next
        # run() against the same context sees them.
        ctx.session.messages = messages
        elapsed = time.perf_counter() - t0

        if handoff is not None:
            ctx.emit("handoff", self.name, {"target": handoff.target.name, "reason": handoff.reason})
            target_task = handoff.task or task
            # Sub-agent runs with a fresh Session so its tool history
            # doesn't pollute the router's. deps / stop / on_event flow
            # through unchanged.
            sub_ctx = AgentContext(
                session=Session(),
                deps=ctx.deps,
                on_event=ctx.on_event,
                stop=ctx.stop,
            )
            if isinstance(handoff.target, LLMAgent) and handoff.target._llm is None:
                handoff.target.bind_llm(llm)
            sub_result = handoff.target.run(target_task, sub_ctx)
            sub_result.handed_off_to = handoff.target.name
            sub_result.elapsed = time.perf_counter() - t0
            ctx.emit("agent_end", self.name, {"reply": sub_result.reply, "via_handoff": True})
            return sub_result

        preempted = ctx.stop.is_set()
        result = AgentResult(
            reply=reply if not preempted else (reply or "Stopped."),
            agent_name=self.name,
            elapsed=elapsed,
            tool_calls=captured_tools,
            preempted=preempted,
        )
        ctx.emit("agent_end", self.name, {"reply": result.reply, "preempted": preempted})
        return result

    def _drive(
        self,
        llm: LLM,
        messages: list[dict],
        task: str,
        ctx: AgentContext,
    ) -> tuple[str, Handoff | None, list[ToolCallResult]]:
        """Drive the LLM loop against a caller-owned messages list.

        The ``messages`` list is mutated in place to append the user
        turn, any tool calls, and the assistant replies — but it is
        **not** shared with the LLM's internal state. Concurrent calls
        from parallel agents are isolated because each has its own
        list. The LLM itself is still thread-unsafe and is guarded by
        ``LLM.complete()``'s inference lock.
        """
        import json as _json

        messages.append({"role": "user", "content": task})
        captured: list[ToolCallResult] = []
        tool_schemas = self._tool_registry.openai_schemas() if self._tool_registry else None

        for hop in range(self._max_tool_hops + 1):
            if ctx.stop.is_set():
                return ("Stopped.", None, captured)

            result = llm.complete(messages, tools=tool_schemas, stream=False)
            message = result["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            raw_content = message.get("content") or ""
            content = raw_content.strip()
            fallback_mode = False

            if not tool_calls:
                from edgevox.llm.llamacpp import (
                    _parse_gemma_inline_tool_calls,
                )

                known = set(self._tool_registry.tools.keys())
                recovered = _parse_gemma_inline_tool_calls(raw_content, known_tools=known)
                if recovered:
                    tool_calls = recovered
                    fallback_mode = True
                    cut = raw_content.find("<|tool_call>")
                    content = raw_content[:cut].strip() if cut >= 0 else ""

            if not tool_calls:
                messages.append({"role": "assistant", "content": content})
                return (content, None, captured)

            if hop == self._max_tool_hops:
                log.warning(
                    "Tool-call budget exhausted for %s after %d hops",
                    self.name,
                    self._max_tool_hops,
                )
                fallback = content or "Sorry, I couldn't finish that request."
                messages.append({"role": "assistant", "content": fallback})
                return (fallback, None, captured)

            # ----- dispatch this batch of tool calls (parallel) -----
            #
            # Handoffs short-circuit before any tool runs — we split the
            # batch into handoff vs non-handoff and, if any handoff is
            # present, honour the first one immediately.
            handoff_hit: Handoff | None = None
            real_calls: list[dict] = []
            for call in tool_calls:
                name = (call.get("function") or {}).get("name", "")
                if name in self._handoff_registry and handoff_hit is None:
                    handoff_hit = Handoff(target=self._handoff_registry[name])
                    captured.append(
                        ToolCallResult(
                            name=name,
                            arguments={},
                            result=f"handoff -> {handoff_hit.target.name}",
                        )
                    )
                else:
                    real_calls.append(call)

            if handoff_hit is not None:
                return ("", handoff_hit, captured)

            # Run each tool/skill in parallel. Tools finish in milliseconds;
            # skills run on their own worker threads but the poll loop
            # here still blocks, so we parallelize the *dispatch* so one
            # slow skill doesn't serialize the others.
            dispatched = self._dispatch_batch(real_calls, ctx)
            captured.extend(out for _, _, out in dispatched)

            if ctx.stop.is_set():
                return ("Stopped.", None, captured)

            if fallback_mode:
                summary = "; ".join(f"{name} -> {_json.dumps(payload, default=str)}" for name, payload, _ in dispatched)
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"(system: tool results — {summary}. "
                            "Now answer the previous request in one short sentence.)"
                        ),
                    }
                )
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": real_calls,
                    }
                )
                for call, (name, payload, _) in zip(real_calls, dispatched, strict=False):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", f"call_{hop}_{name}"),
                            "name": name,
                            "content": _json.dumps(payload, default=str),
                        }
                    )

        return ("", None, captured)

    def _dispatch_batch(
        self,
        calls: list[dict],
        ctx: AgentContext,
    ) -> list[tuple[str, dict, ToolCallResult]]:
        """Dispatch a batch of tool/skill calls, parallelized when >1.

        The LLM can emit multiple tool_calls in a single response; each
        one runs in its own worker so a slow skill doesn't serialize
        the rest. Order of the returned list matches ``calls`` order
        so the LLM sees a deterministic tool-result sequence.
        """
        from concurrent.futures import ThreadPoolExecutor

        results: list[tuple[str, dict, ToolCallResult]] = [None] * len(calls)  # type: ignore[list-item]

        def run_one(idx: int, call: dict) -> None:
            if ctx.stop.is_set():
                return
            fn = call.get("function", {})
            name = fn.get("name", "")
            args_raw = fn.get("arguments", "{}")
            if name in self._skill_registry:
                outcome = self._dispatch_skill(name, args_raw, ctx)
            else:
                outcome = self._tool_registry.dispatch(name, args_raw, ctx=ctx)
            payload = {"ok": True, "result": outcome.result} if outcome.ok else {"ok": False, "error": outcome.error}
            results[idx] = (name, payload, outcome)
            ctx.emit("tool_call", self.name, outcome)

        if len(calls) > 1:
            with ThreadPoolExecutor(max_workers=min(len(calls), 8)) as ex:
                for f in [ex.submit(run_one, i, c) for i, c in enumerate(calls)]:
                    f.result()
        else:
            for i, c in enumerate(calls):
                run_one(i, c)
        return [r for r in results if r is not None]

    def _dispatch_skill(self, name: str, arguments: str | dict[str, Any], ctx: AgentContext) -> ToolCallResult:
        """Dispatch a skill call through its ``GoalHandle`` lifecycle.

        Blocks until the goal completes, fails, is cancelled, or hits
        its timeout. The worker thread inside the skill is where the
        actual work happens; this method polls so ``ctx.stop`` can
        preempt immediately.
        """
        import json as _json

        from edgevox.agents.skills import GoalStatus

        skill_obj = self._skill_registry[name]

        if isinstance(arguments, str):
            try:
                args = _json.loads(arguments) if arguments else {}
            except _json.JSONDecodeError as e:
                return ToolCallResult(name=name, arguments={}, error=f"invalid JSON arguments: {e}")
        else:
            args = arguments or {}

        try:
            handle = skill_obj.start(ctx, **args)
        except Exception as e:
            log.exception("Skill %r failed to start", name)
            return ToolCallResult(name=name, arguments=args, error=f"start failed: {e}")

        ctx.emit("skill_goal", self.name, {"skill": name, "handle_id": handle.id, "args": args})

        # Poll the handle. Physics (for sims like IrSimEnvironment) runs
        # on a dedicated background thread in the deps, so we just wait
        # on the handle's terminal event with a short timeout and check
        # ``ctx.stop`` in between for safety preemption.
        poll_interval = 0.05
        while True:
            if ctx.stop.is_set():
                handle.cancel()
                ctx.emit(
                    "skill_cancelled",
                    self.name,
                    {"skill": name, "reason": "safety_preempt"},
                )
                return ToolCallResult(name=name, arguments=args, error="cancelled (safety_preempt)")
            status = handle.poll(timeout=poll_interval)
            if status in (GoalStatus.SUCCEEDED, GoalStatus.FAILED, GoalStatus.CANCELLED):
                break

        if handle.status is GoalStatus.SUCCEEDED:
            return ToolCallResult(name=name, arguments=args, result=handle.result)
        if handle.status is GoalStatus.CANCELLED:
            return ToolCallResult(name=name, arguments=args, error="cancelled")
        return ToolCallResult(
            name=name,
            arguments=args,
            error=handle.error or "skill failed",
        )

    # ----- streaming is not supported on the multi-hop path; fall back -----

    def run_stream(self, task: str, ctx: AgentContext | None = None) -> Iterator[str]:
        """Yield the final reply as one chunk.

        Multi-step / multi-agent flows can't cleanly stream tokens while
        also intercepting tool calls, so we degrade to single-chunk.
        Callers that sentence-split (e.g. the TTS pipeline) still work
        naturally. Pure chitchat paths could stream but for simplicity
        this uses the non-streaming path uniformly.
        """
        result = self.run(task, ctx)
        if result.reply:
            yield result.reply


# --------- synthetic handoff tool helper ---------


def _make_handoff_tool(target: Agent) -> Callable[..., Any]:
    """Synthesize a ``@tool``-decorated function for a handoff target.

    The returned callable is only used for its ``__edgevox_tool__``
    descriptor — the LLMAgent dispatch loop intercepts calls to this
    tool's name before executing it, so the body here is just a
    fallback so the function is callable for tests.
    """
    tool_name = f"handoff_to_{target.name}"
    description = f"Transfer this request to the {target.name!r} specialist agent. {target.description}"

    def _handoff(reason: str = "") -> Handoff:  # body only runs in tests
        return Handoff(target=target, reason=reason)

    _handoff.__name__ = tool_name
    _handoff.__doc__ = (
        f"{description}\n\n    Args:\n        reason: one short phrase explaining why you're handing off.\n"
    )
    return tool_decorator(name=tool_name, description=description)(_handoff)
