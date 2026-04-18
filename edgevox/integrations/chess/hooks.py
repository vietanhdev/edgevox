"""Chess-specific agent hooks.

Two hooks, both per-agent (attach via ``LLMAgent(hooks=[...])``):

- :class:`BoardStateInjectionHook` — fires at ``on_run_start``, prepends
  a compact board summary to the user's task so the LLM never invents
  positions. Priority 80, same tier as :class:`MemoryInjectionHook`.
- :class:`MoveCommentaryHook` — fires at ``after_tool`` when the tool
  was ``engine_move`` or ``play_user_move``; stashes the resulting
  state so the follow-up assistant message can draw from it. Priority
  60 (detection tier).

State is keyed by ``id(self)`` under ``ctx.hook_state`` per ADR-002 so
two hook instances don't share buffers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from edgevox.agents.hooks import AFTER_TOOL, ON_RUN_START, HookResult

if TYPE_CHECKING:
    from edgevox.agents.base import AgentContext
    from edgevox.integrations.chess.environment import ChessEnvironment
    from edgevox.llm.tools import ToolCallResult


class BoardStateInjectionHook:
    """Prepend a compact board summary to every user task.

    Pulls the state off ``ctx.deps`` (a :class:`ChessEnvironment`) and
    injects ``FEN | side-to-move | last move | eval`` so the LLM always
    reasons from the true position, even when the user's utterance is
    short ("your move", "what now?"). Idempotent across tool hops:
    only runs at ``on_run_start``.
    """

    points = frozenset({ON_RUN_START})
    priority = 80

    def __init__(self, *, include_history_plies: int = 4) -> None:
        self.include_history_plies = max(0, include_history_plies)

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        env = _chess_env(ctx)
        if env is None:
            return None
        if not isinstance(payload, dict):
            return None
        task = payload.get("task", "")
        prefix = self._render_prefix(env)
        if not prefix:
            return None
        new_payload = dict(payload)
        new_payload["task"] = f"{prefix}\n\n{task}"
        # Snapshot the rendered prefix under id(self)-keyed hook state
        # so observability / downstream hooks can see what we injected
        # without re-rendering.
        ctx.hook_state.setdefault(id(self), {})["last_prefix"] = prefix
        return HookResult.replace(new_payload, reason="board state injected")

    def _render_prefix(self, env: ChessEnvironment) -> str:
        state = env.snapshot()
        lines = [
            f"[board] fen: {state.fen}",
            f"[board] side to move: {state.turn} · user plays {env.user_plays} · engine plays {env.engine_plays}",
        ]
        if state.last_move_san:
            classification = state.last_move_classification.value if state.last_move_classification else "unclassified"
            lines.append(f"[board] last move: {state.last_move_san} ({classification})")
        if state.eval_cp is not None:
            lines.append(f"[board] eval: {state.eval_cp:+d} cp (white pov)")
        elif state.mate_in is not None:
            lines.append(f"[board] mate in {state.mate_in}")
        if state.opening:
            lines.append(f"[board] opening: {state.opening}")
        if self.include_history_plies and state.san_history:
            tail = state.san_history[-self.include_history_plies :]
            lines.append(f"[board] recent moves: {' '.join(tail)}")
        if state.is_game_over:
            lines.append(f"[board] GAME OVER: {state.game_over_reason} · winner: {state.winner or 'draw'}")
        return "\n".join(lines)


class MoveCommentaryHook:
    """Capture chess tool outcomes for commentary shaping.

    Doesn't rewrite anything directly — it stores the last move outcome
    under ``ctx.hook_state[id(self)]`` so other hooks / the agent's
    system prompt can reference a fresh eval + classification without
    re-querying the environment. Lets the persona's system prompt
    ("comment naturally on the last move") stay generic.
    """

    points = frozenset({AFTER_TOOL})
    priority = 60
    TRACKED_TOOLS = frozenset({"engine_move", "play_user_move"})

    def __call__(self, point: str, ctx: AgentContext, payload: ToolCallResult) -> HookResult | None:
        if payload.name not in self.TRACKED_TOOLS or not payload.ok:
            return None
        env = _chess_env(ctx)
        if env is None:
            return None
        state = env.snapshot()
        bucket = ctx.hook_state.setdefault(id(self), {})
        bucket["last_tool"] = payload.name
        bucket["last_state"] = state.to_json()
        return None


def _chess_env(ctx: AgentContext) -> ChessEnvironment | None:
    """Return ``ctx.deps`` if it looks like a :class:`ChessEnvironment`.

    Duck-typed so tests can swap a stub with the same surface. We look
    for ``snapshot()`` + the user/engine side attributes — the minimum
    the hooks need.
    """
    deps = getattr(ctx, "deps", None)
    if deps is None:
        return None
    if hasattr(deps, "snapshot") and hasattr(deps, "user_plays") and hasattr(deps, "engine_plays"):
        return deps  # type: ignore[return-value]
    return None


__all__ = ["BoardStateInjectionHook", "MoveCommentaryHook"]
