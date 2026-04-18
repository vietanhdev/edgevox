"""Workflow primitives — Behavior-Tree-shaped agent composition.

Every workflow implements the :class:`Agent` protocol so workflows
nest transparently inside each other. The semantics are a direct
subset of Behavior Tree node types (Sequence, Fallback, Loop, Retry,
Timeout) plus a convenience ``Router`` helper that builds a single
handoff-only LLMAgent.

No ``Parallel`` in v1: a single GGUF is GIL-bound during generation,
and real parallelism would need multiple LLMs loaded (memory budget
on edge devices). Defer to v2 when multi-LLM becomes viable.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from edgevox.agents.base import Agent, AgentContext, AgentResult, LLMAgent

if TYPE_CHECKING:
    from edgevox.llm.llamacpp import LLM

log = logging.getLogger(__name__)


def _bind_llm_recursive(agent: Agent, llm: LLM) -> None:
    """Walk a composite agent tree and bind the shared LLM into every
    LLMAgent leaf. Workflows call this as a one-time wiring step so
    examples only need to pass the LLM to the top-level agent."""
    if isinstance(agent, LLMAgent):
        if agent.llm is None:
            agent.bind_llm(llm)
        for sub in agent._handoff_registry.values():
            _bind_llm_recursive(sub, llm)
        return
    for attr_name in ("_children", "_agent", "_wrapped", "_routes"):
        attr = getattr(agent, attr_name, None)
        if attr is None:
            continue
        if isinstance(attr, list | tuple):
            for child in attr:
                _bind_llm_recursive(child, llm)
        elif isinstance(attr, dict):
            for child in attr.values():
                _bind_llm_recursive(child, llm)
        else:
            _bind_llm_recursive(attr, llm)


# --------- Sequence ---------


class Sequence:
    """BT Sequence: run children in order, each child's reply becomes
    the next child's task. Stops early on failure (empty reply or
    preempted). Returns the *last successful* child's reply.
    """

    def __init__(self, name: str, agents: list[Agent], *, description: str = "") -> None:
        if not agents:
            raise ValueError("Sequence needs at least one agent")
        self.name = name
        self.description = description or f"Sequence of {len(agents)} agents"
        self._children = agents

    def run(self, task: str, ctx: AgentContext | None = None) -> AgentResult:
        ctx = ctx or AgentContext()
        ctx.emit("agent_start", self.name, {"task": task})
        t0 = time.perf_counter()
        current_task = task
        last_result: AgentResult | None = None
        for child in self._children:
            if ctx.stop.is_set():
                break
            result = child.run(current_task, ctx)
            last_result = result
            if result.preempted:
                break
            if not result.reply.strip():
                break
            current_task = result.reply
        elapsed = time.perf_counter() - t0
        if last_result is None:
            final = AgentResult(reply="", agent_name=self.name, elapsed=elapsed)
        else:
            final = AgentResult(
                reply=last_result.reply,
                agent_name=self.name,
                elapsed=elapsed,
                preempted=last_result.preempted,
            )
        ctx.emit("agent_end", self.name, {"reply": final.reply})
        return final

    def run_stream(self, task: str, ctx: AgentContext | None = None) -> Iterator[str]:
        result = self.run(task, ctx)
        if result.reply:
            yield result.reply


# --------- Fallback / Selector ---------


class Fallback:
    """BT Fallback (Selector): try each child in order; return the
    first one whose reply is non-empty and not preempted. If all fail,
    return the last child's result.
    """

    def __init__(self, name: str, agents: list[Agent], *, description: str = "") -> None:
        if not agents:
            raise ValueError("Fallback needs at least one agent")
        self.name = name
        self.description = description or f"Fallback of {len(agents)} agents"
        self._children = agents

    def run(self, task: str, ctx: AgentContext | None = None) -> AgentResult:
        ctx = ctx or AgentContext()
        ctx.emit("agent_start", self.name, {"task": task})
        t0 = time.perf_counter()
        last: AgentResult | None = None
        for child in self._children:
            if ctx.stop.is_set():
                break
            result = child.run(task, ctx)
            last = result
            if result.reply.strip() and not result.preempted:
                ctx.emit("agent_end", self.name, {"reply": result.reply, "via": child.name})
                return AgentResult(
                    reply=result.reply,
                    agent_name=self.name,
                    elapsed=time.perf_counter() - t0,
                )
        final = AgentResult(
            reply=last.reply if last else "",
            agent_name=self.name,
            elapsed=time.perf_counter() - t0,
            preempted=bool(last and last.preempted),
        )
        ctx.emit("agent_end", self.name, {"reply": final.reply, "all_failed": True})
        return final

    def run_stream(self, task: str, ctx: AgentContext | None = None) -> Iterator[str]:
        result = self.run(task, ctx)
        if result.reply:
            yield result.reply


# --------- Loop ---------


class Loop:
    """BT Loop: run a single child repeatedly until ``until(state)`` is
    truthy or ``max_iterations`` is reached. ``until`` receives the
    ``ctx.session.state`` dict each iteration so the child can write
    progress markers.
    """

    def __init__(
        self,
        name: str,
        agent: Agent,
        *,
        until: Callable[[dict[str, Any]], bool],
        max_iterations: int = 5,
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description or f"Loop over {agent.name}"
        self._agent = agent
        self._until = until
        self._max = max_iterations

    def run(self, task: str, ctx: AgentContext | None = None) -> AgentResult:
        ctx = ctx or AgentContext()
        ctx.emit("agent_start", self.name, {"task": task})
        t0 = time.perf_counter()
        last: AgentResult | None = None
        for _ in range(self._max):
            if ctx.stop.is_set():
                break
            last = self._agent.run(task if last is None else last.reply, ctx)
            if last.preempted:
                break
            try:
                done = bool(self._until(ctx.session.state))
            except Exception:
                log.exception("Loop until() raised")
                done = True
            if done:
                break
        final = AgentResult(
            reply=last.reply if last else "",
            agent_name=self.name,
            elapsed=time.perf_counter() - t0,
            preempted=bool(last and last.preempted),
        )
        ctx.emit("agent_end", self.name, {"reply": final.reply})
        return final

    def run_stream(self, task: str, ctx: AgentContext | None = None) -> Iterator[str]:
        result = self.run(task, ctx)
        if result.reply:
            yield result.reply


# --------- Retry decorator ---------


class Retry:
    """Wrap an agent with bounded retries. Re-runs on preempted=False
    and empty reply, up to ``max_attempts`` times.
    """

    def __init__(self, agent: Agent, *, max_attempts: int = 3, name: str | None = None) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._wrapped = agent
        self._max = max_attempts
        self.name = name or f"retry_{agent.name}"
        self.description = f"Retry wrapper (max {max_attempts}) around {agent.name}"

    def run(self, task: str, ctx: AgentContext | None = None) -> AgentResult:
        ctx = ctx or AgentContext()
        last: AgentResult | None = None
        for _ in range(self._max):
            if ctx.stop.is_set():
                break
            last = self._wrapped.run(task, ctx)
            if last.preempted:
                break
            if last.reply.strip():
                return last
        return last or AgentResult(reply="", agent_name=self.name)

    def run_stream(self, task: str, ctx: AgentContext | None = None) -> Iterator[str]:
        result = self.run(task, ctx)
        if result.reply:
            yield result.reply


# --------- Timeout decorator ---------


class Timeout:
    """Wrap an agent with a wall-clock deadline. On expiry, sets
    ``ctx.stop`` to cancel any in-flight skills and returns a failure
    result.
    """

    def __init__(self, agent: Agent, *, seconds: float, name: str | None = None) -> None:
        if seconds <= 0:
            raise ValueError("seconds must be positive")
        self._wrapped = agent
        self._seconds = seconds
        self.name = name or f"timeout_{agent.name}"
        self.description = f"Timeout wrapper ({seconds:.1f}s) around {agent.name}"

    def run(self, task: str, ctx: AgentContext | None = None) -> AgentResult:
        ctx = ctx or AgentContext()
        result_box: dict[str, AgentResult | None] = {"r": None}

        def body() -> None:
            result_box["r"] = self._wrapped.run(task, ctx)

        t = threading.Thread(target=body, name=f"timeout-{self._wrapped.name}", daemon=True)
        t.start()
        t.join(timeout=self._seconds)
        if t.is_alive():
            ctx.stop.set()
            t.join(timeout=0.5)
            return AgentResult(
                reply="Timed out.",
                agent_name=self.name,
                preempted=True,
            )
        return result_box["r"] or AgentResult(reply="", agent_name=self.name)

    def run_stream(self, task: str, ctx: AgentContext | None = None) -> Iterator[str]:
        result = self.run(task, ctx)
        if result.reply:
            yield result.reply


# --------- Router ---------


class Parallel:
    """Run N sub-agents concurrently and reduce their replies.

    Every sub-agent runs on its own worker thread with its own
    :class:`Session` so their tool histories don't interfere. The
    shared ``LLM`` serializes inference via its internal lock — the
    speedup comes from overlapping non-LLM work (tool dispatch,
    skill worker threads, I/O waits).

    Args:
        name: workflow display name.
        agents: list of sub-agents to fan out to.
        reduce: callable that receives the list of :class:`AgentResult`
            and returns the merged reply string. Defaults to
            concatenating replies with newlines.
        description: shown in composite workflows.
    """

    def __init__(
        self,
        name: str,
        agents: list[Agent],
        *,
        reduce: Callable[[list[AgentResult]], str] | None = None,
        description: str = "",
    ) -> None:
        if not agents:
            raise ValueError("Parallel needs at least one agent")
        self.name = name
        self.description = description or f"Parallel of {len(agents)} agents"
        self._children = agents
        self._reduce = reduce or (lambda results: "\n".join(r.reply for r in results if r.reply))

    def run(self, task: str, ctx: AgentContext | None = None) -> AgentResult:
        from concurrent.futures import ThreadPoolExecutor

        from edgevox.agents.base import Session

        ctx = ctx or AgentContext()
        ctx.emit("agent_start", self.name, {"task": task})
        t0 = time.perf_counter()

        # Each sub-agent gets its own Session so their histories stay
        # isolated. Deps / stop / bus flow through unchanged so sinks
        # observe every child's events.
        def _run_child(child: Agent) -> AgentResult:
            child_ctx = AgentContext(
                session=Session(),
                deps=ctx.deps,
                bus=ctx.bus,
                stop=ctx.stop,
            )
            return child.run(task, child_ctx)

        with ThreadPoolExecutor(max_workers=min(len(self._children), 16)) as ex:
            results = list(ex.map(_run_child, self._children))

        merged = self._reduce(results)
        final = AgentResult(
            reply=merged,
            agent_name=self.name,
            elapsed=time.perf_counter() - t0,
            preempted=any(r.preempted for r in results),
        )
        ctx.emit("agent_end", self.name, {"reply": merged})
        return final

    def run_stream(self, task: str, ctx: AgentContext | None = None) -> Iterator[str]:
        result = self.run(task, ctx)
        if result.reply:
            yield result.reply


class Router:
    """Convenience builder for a tiny handoff-only LLMAgent.

    A ``Router`` is the voice-optimized multi-agent primitive: one
    router LLM call that picks which specialist handles the turn,
    then the specialist runs. Total cost is 2 LLM calls (router +
    leaf), or 3 if the leaf itself uses a tool — never more.
    """

    @classmethod
    def build(
        cls,
        name: str,
        instructions: str,
        routes: dict[str, Agent],
        *,
        description: str = "",
    ) -> LLMAgent:
        """Construct the router as a plain ``LLMAgent`` with only
        handoff targets in its tool list."""
        return LLMAgent(
            name=name,
            description=description or f"Router over {len(routes)} specialists",
            instructions=instructions,
            handoffs=list(routes.values()),
        )


# --------- Supervisor (LangGraph supervisor pattern) ---------


class Supervisor:
    """Convenience builder for a LangGraph-style supervisor agent.

    A supervisor is the same wire-shape as :class:`Router` — one LLM
    call routes to a worker agent via a synthetic ``handoff_to_<name>``
    tool — but the API explicitly models the (name → worker) graph and
    forces the supervisor's :class:`LLMAgent` to use the
    ``required_first_hop`` tool-choice policy so it *must* dispatch
    rather than ramble. This is the canonical SLM loop-break for
    multi-agent dispatch.

    Use :class:`Router` when you want a router that can also reply
    plainly (e.g. small-talk fallback). Use :class:`Supervisor` when
    every turn must end up at a worker.
    """

    @classmethod
    def build(
        cls,
        name: str,
        instructions: str,
        workers: dict[str, Agent],
        *,
        description: str = "",
    ) -> LLMAgent:
        """Construct the supervisor as an :class:`LLMAgent` with only
        worker agents in its tool list and a forced first-hop tool
        call."""
        if not workers:
            raise ValueError("Supervisor needs at least one worker")
        return LLMAgent(
            name=name,
            description=description or f"Supervisor over {len(workers)} workers",
            instructions=instructions,
            handoffs=list(workers.values()),
            tool_choice_policy="required_first_hop",
        )


# --------- Orchestrator-Worker (Anthropic Claude Research pattern) ---------


class Orchestrator:
    """Anthropic-style orchestrator-worker: a lead LLM emits a JSON
    plan; for each subtask we ``spawn_subagent`` with a *scoped*
    subset of the registered tools; the lead then synthesises a final
    answer from the workers' replies.

    The lead and worker can share one GGUF in memory thanks to
    :meth:`LLMAgent.spawn_subagent` — and each subagent gets a fresh
    :class:`~edgevox.agents.base.Session` so its tool history doesn't
    pollute the lead's. Matches the "orchestrator-worker" pattern
    documented in the Anthropic engineering writeup of Claude
    Research; ~90% of single-agent quality at a fraction of the
    context cost on long fan-out tasks.

    The plan format is intentionally minimal — a JSON array of
    ``{"tools": [...], "objective": "...", "output_format": "..."}``
    objects — so any 1-3B SLM can emit it under
    ``tool_choice_policy="required_first_hop"`` (a strict
    ``emit_plan`` tool whose argument schema is the plan dict).
    Defer richer plan grammars (DAG dependencies, retries) to v2.

    Usage::

        orch = Orchestrator(
            name="research",
            lead_instructions="Decompose the user request into 1-3 sub-questions.",
            synth_instructions="Combine the worker answers into one short reply.",
            tools=[search, lookup, calc],
        )
        result = orch.run("Plan a kitchen remodel under $5k", ctx)
    """

    def __init__(
        self,
        name: str,
        lead_instructions: str,
        synth_instructions: str,
        *,
        tools: list | None = None,
        max_subtasks: int = 4,
        worker_max_tool_hops: int = 3,
        description: str = "",
    ) -> None:
        from edgevox.llm.tools import tool as tool_decorator

        if max_subtasks < 1:
            raise ValueError("max_subtasks must be >= 1")
        self.name = name
        self.description = description or f"Orchestrator-worker over {len(tools or [])} tools"
        self._tools = list(tools or [])
        self._max_subtasks = max_subtasks
        self._worker_max_tool_hops = worker_max_tool_hops

        # The lead emits a single ``emit_plan`` tool call whose
        # arguments are the full plan dict. Forcing that via
        # ``tool_choice_policy="required_first_hop"`` + a GBNF grammar
        # eliminates the malformed-JSON failure mode SLMs otherwise
        # hit on plan-shaped output.
        self._captured_plan: list[dict] = []

        @tool_decorator
        def emit_plan(subtasks: list) -> str:
            """Submit the orchestrator's decomposition.

            Args:
                subtasks: ordered list, each item ``{"objective": str,
                    "tools": [str], "output_format": str}``. Workers
                    run in parallel; their outputs are synthesised
                    into one final answer.
            """
            # Capture into the orchestrator's per-run buffer; the
            # actual fan-out happens after the lead returns.
            self._captured_plan.append({"subtasks": subtasks})
            return "plan accepted"

        self._lead = LLMAgent(
            name=f"{name}.lead",
            description=f"Plan emitter for {name}",
            instructions=lead_instructions,
            tools=[emit_plan],
            tool_choice_policy="required_first_hop",
        )
        self._synth = LLMAgent(
            name=f"{name}.synth",
            description=f"Result synthesiser for {name}",
            instructions=synth_instructions,
        )

    def run(self, task: str, ctx: AgentContext | None = None) -> AgentResult:
        ctx = ctx or AgentContext()
        ctx.emit("agent_start", self.name, {"task": task})
        t0 = time.perf_counter()

        # Phase 1: lead emits a plan via the ``emit_plan`` tool. Reset
        # the plan buffer so concurrent runs don't leak.
        self._captured_plan.clear()
        self._lead.run(task, ctx)
        plan = self._captured_plan[-1] if self._captured_plan else {"subtasks": []}
        subtasks = plan.get("subtasks") or []
        if not isinstance(subtasks, list):
            subtasks = []
        subtasks = subtasks[: self._max_subtasks]

        # Phase 2: spawn one subagent per subtask, each with a tool
        # filter so only the tools the plan named are visible. Run
        # synchronously — the LLM's inference lock would serialise
        # parallel subagents anyway on a single-GGUF deployment.
        worker_replies: list[tuple[str, str]] = []
        tool_index = {self._tool_name(t): t for t in self._tools}
        for st in subtasks:
            if ctx.stop.is_set() or ctx.should_stop():
                break
            objective = (st or {}).get("objective", "")
            wanted = (st or {}).get("tools", []) or []
            output_format = (st or {}).get("output_format", "concise text")
            if not isinstance(objective, str) or not objective.strip():
                continue
            scoped = [tool_index[n] for n in wanted if isinstance(n, str) and n in tool_index]
            sub_result = self._lead.spawn_subagent(
                f"{objective}\n\nReply format: {output_format}",
                parent_ctx=ctx,
                tools=scoped or None,
                max_tool_hops=self._worker_max_tool_hops,
            )
            worker_replies.append((objective, sub_result.reply))

        # Phase 3: synth folds the worker outputs into one reply.
        if not worker_replies:
            final_reply = ""
        elif len(worker_replies) == 1:
            # No synthesis needed when there's only one worker.
            final_reply = worker_replies[0][1]
        else:
            synth_input = "\n\n".join(f"### {obj}\n{reply}" for obj, reply in worker_replies if reply)
            synth_result = self._synth.run(f"Original request: {task}\n\nWorker outputs:\n{synth_input}", ctx)
            final_reply = synth_result.reply

        elapsed = time.perf_counter() - t0
        out = AgentResult(reply=final_reply, agent_name=self.name, elapsed=elapsed)
        ctx.emit("agent_end", self.name, {"reply": final_reply})
        return out

    def run_stream(self, task: str, ctx: AgentContext | None = None) -> Iterator[str]:
        result = self.run(task, ctx)
        if result.reply:
            yield result.reply

    @staticmethod
    def _tool_name(t: Any) -> str:
        # @tool decorator stamps __edgevox_tool__ on the function;
        # the tool's ``.name`` attribute is the canonical name.
        descriptor = getattr(t, "__edgevox_tool__", None)
        if descriptor is not None:
            return getattr(descriptor, "name", "")
        return getattr(t, "name", getattr(t, "__name__", ""))
