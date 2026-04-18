"""BoardHintHook — inject top engine candidates + legal-move hints.

Sits alongside :class:`BoardStateInjectionHook` to give the LLM a
curated slice of the position instead of forcing it to recall chess
theory on its own. Two shapes of enrichment:

- **Engine candidates** — always. After every turn, the engine's top
  1-3 moves (from a shallow search) land in the prompt as SAN strings.
  Small models pick better when they see what a stronger engine thinks;
  they don't have to invent candidates out of thin air.
- **Legal moves** — only when the user asks a meta question like
  "what are my options?" or "what can I do?". Full legal-move lists
  are expensive in tokens, so we gate them on intent.

Injection format piggybacks the existing ``[board] …`` lines so it
reads as one coherent block when the model consumes the system prompt.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from edgevox.agents.hooks import ON_RUN_START, HookResult

if TYPE_CHECKING:
    from edgevox.agents.base import AgentContext
    from edgevox.integrations.chess.environment import ChessEnvironment

log = logging.getLogger(__name__)


# Meta questions — user wants options/advice, not to play a move.
_META_RE = re.compile(
    r"\b(what (can|should|could|do) i|what are my|options|help me think|what do you (think|suggest)|any (ideas|suggestions)|hint|advice)\b",
    re.IGNORECASE,
)


class BoardHintHook:
    """Augment the prompt with top engine lines + optional legal-move list.

    Runs at :data:`ON_RUN_START` after :class:`MoveInterceptHook` (which
    may have applied a move and rewritten the task), and before
    :class:`BoardStateInjectionHook` (which does the main board
    summary). Priority 85 places us in that slot.

    The hook mutates ``payload['task']`` by prepending a short
    ``[board] …`` line. Keeping it as one extra line means the board
    summary from :class:`BoardStateInjectionHook` still reads coherently
    — nothing gets split.
    """

    points = frozenset({ON_RUN_START})
    priority = 85  # between MoveInterceptHook (90) and BoardStateInjectionHook (80)

    def __init__(self, *, candidate_depth: int = 6, max_candidates: int = 3) -> None:
        self.candidate_depth = candidate_depth
        self.max_candidates = max_candidates

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        if not isinstance(payload, dict):
            return None
        env = _chess_env(ctx)
        if env is None:
            return None
        task = payload.get("task", "")
        if not task:
            return None

        lines: list[str] = []

        # Engine candidates — always.
        try:
            analysis = env.analyse(depth=self.candidate_depth)
        except Exception:
            log.debug("BoardHintHook: analyse failed", exc_info=True)
            analysis = None
        if analysis and analysis.pv:
            pv = analysis.pv[: self.max_candidates]
            eval_str = ""
            if analysis.score_from_white is not None:
                eval_str = f" (eval {analysis.score_from_white:+d} cp)"
            lines.append(f"[board] top engine line: {' '.join(pv)}{eval_str}")

        # Legal-move listing only when the user is asking for options.
        if _META_RE.search(task):
            try:
                legal = env.list_legal_moves()
            except Exception:
                log.debug("BoardHintHook: list_legal_moves failed", exc_info=True)
                legal = []
            if legal:
                # Show up to 12 moves. Beyond that, list is noise for a 2B model.
                snippet = ", ".join(legal[:12])
                suffix = f" (+{len(legal) - 12} more)" if len(legal) > 12 else ""
                lines.append(f"[board] legal moves in UCI: {snippet}{suffix}")

        if not lines:
            return None

        prefix = "\n".join(lines)
        return HookResult.replace(
            {**payload, "task": f"{prefix}\n\n{task}"},
            reason="board hints injected",
        )


def _chess_env(ctx: AgentContext) -> ChessEnvironment | None:
    deps = getattr(ctx, "deps", None)
    if deps is None:
        return None
    if hasattr(deps, "analyse") and hasattr(deps, "list_legal_moves"):
        return deps  # type: ignore[return-value]
    return None


__all__ = ["BoardHintHook"]
