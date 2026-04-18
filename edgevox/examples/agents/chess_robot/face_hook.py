"""RobotFaceHook — publish ``robot_face`` events from chess state transitions.

The hook fires at four points:

- :data:`ON_RUN_START` — set initial face to ``thinking`` so the robot
  looks attentive the moment the user starts talking.
- :data:`AFTER_TOOL` — a move just applied; recompute mood + gaze so
  the face reacts *before* the agent's next LLM hop (i.e. before any
  commentary is generated).
- :data:`AFTER_LLM` — the agent produced its reply; switch tempo to
  ``speaking`` so the UI starts animating the mouth from the TTS
  stream.
- :data:`ON_RUN_END` — the run is done; return to ``calm`` so the face
  doesn't freeze mid-reaction if the TTS finishes before the next turn.

Each fire emits a ``robot_face`` :class:`AgentEvent` via
:meth:`AgentContext.emit` — the WebSocket server is the subscriber,
forwarding the payload to the React client as a ``robot_face`` message.

State for the hook (last-emitted mood, last gaze) lives under
``ctx.hook_state[id(self)]`` per ADR-002 so two hook instances (e.g. a
second agent in a handoff) don't share buffers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from edgevox.agents.hooks import (
    AFTER_LLM,
    AFTER_TOOL,
    ON_RUN_END,
    ON_RUN_START,
    HookResult,
)
from edgevox.examples.agents.chess_robot.mood import derive_mood, gaze_from_uci

if TYPE_CHECKING:
    from edgevox.agents.base import AgentContext
    from edgevox.integrations.chess.environment import ChessEnvironment


# Tools whose outcome should trigger a face update. Kept in sync with
# :class:`MoveCommentaryHook.TRACKED_TOOLS` because any move-applying
# tool is exactly what the face needs to react to.
MOVE_TOOLS = frozenset({"engine_move", "play_user_move", "new_game", "undo_last_move"})


@dataclass
class RobotFaceEvent:
    """Payload of a ``robot_face`` :class:`AgentEvent`.

    Kept as a dataclass (not a plain dict) so the type flows through the
    server's ``_on_event`` forwarder unchanged and mis-spelled fields
    fail at construction time instead of silently reaching the client.
    """

    mood: str  # Mood.value
    gaze_x: float  # -1..1, file (a=-1, h=+1)
    gaze_y: float  # -1..1, rank (1=-1, 8=+1)
    persona: str
    tempo: str  # "idle" | "thinking" | "speaking"
    last_move_san: str | None = None
    is_game_over: bool = False


class RobotFaceHook:
    """Translate chess state into robot face expression signals.

    Per-agent hook: attach via ``LLMAgent(hooks=[RobotFaceHook(...)])``.
    Idempotent-safe — emitting the same face twice is cheap and the UI
    tweens on changes only.
    """

    points = frozenset({ON_RUN_START, AFTER_TOOL, AFTER_LLM, ON_RUN_END})
    # Run after :class:`MoveCommentaryHook` (priority 60) so the latest
    # classification is already settled under ``ctx.hook_state`` when
    # we read the environment's snapshot. Priority 50 places us in the
    # observability tier.
    priority = 50

    def __init__(self, persona: str = "casual") -> None:
        self.persona = persona

    def __call__(self, point: str, ctx: AgentContext, payload: Any) -> HookResult | None:
        if point == ON_RUN_START:
            self._emit(ctx, tempo="thinking")
            return None

        if point == AFTER_TOOL:
            # Only react to move-applying tools. ``get_board_state`` /
            # ``list_legal_moves`` / ``analyze_position`` shouldn't
            # trigger a face change — they're the agent thinking, not
            # the position shifting.
            if not _is_move_tool(payload):
                return None
            self._emit(ctx, tempo="thinking")
            return None

        if point == AFTER_LLM:
            # The agent's reply is about to stream to TTS. Switch to
            # ``speaking`` so the client's lip-sync starts at the same
            # moment the audio does — otherwise the mouth is closed for
            # the first word.
            self._emit(ctx, tempo="speaking")
            return None

        if point == ON_RUN_END:
            # Return to rest. The UI uses this to kill any speaking
            # animation still in flight and fall back to idle blinks.
            self._emit(ctx, tempo="idle")
            return None

        return None

    # ----- internals -----

    def _emit(self, ctx: AgentContext, *, tempo: str) -> None:
        """Compute the face event from the current env + emit it."""
        env = _chess_env(ctx)
        if env is None:
            return
        state = env.snapshot()
        mood = derive_mood(state, engine_plays=env.engine_plays, persona=self.persona)

        # Gaze logic: user is the audience; when it's ``speaking`` or
        # ``idle`` the robot looks back at the user (0, 0). When
        # ``thinking``, it looks at the destination of the last move —
        # the board — which sells "processing your move".
        if tempo == "thinking":
            gx, gy = gaze_from_uci(state.last_move_uci)
        else:
            gx, gy = 0.0, 0.0

        event = RobotFaceEvent(
            mood=mood.value,
            gaze_x=gx,
            gaze_y=gy,
            persona=self.persona,
            tempo=tempo,
            last_move_san=state.last_move_san,
            is_game_over=state.is_game_over,
        )

        # Cache the last emission so tests / downstream hooks can
        # inspect without re-reading the env.
        ctx.hook_state.setdefault(id(self), {})["last"] = event

        ctx.emit("robot_face", ctx.agent_name or "chess_robot", asdict(event))


def _chess_env(ctx: AgentContext) -> ChessEnvironment | None:
    """Duck-typed :class:`ChessEnvironment` pull from ``ctx.deps``.

    Mirrors :func:`edgevox.integrations.chess.hooks._chess_env` — we
    check for ``snapshot`` + ``engine_plays`` rather than importing the
    class so a test stub with the same surface works unchanged.
    """
    deps = getattr(ctx, "deps", None)
    if deps is None:
        return None
    if hasattr(deps, "snapshot") and hasattr(deps, "engine_plays"):
        return deps  # type: ignore[return-value]
    return None


def _is_move_tool(payload: Any) -> bool:
    """``AFTER_TOOL`` payload is a :class:`ToolCallResult`. Be defensive
    in case an alternative hook upstream replaces it — we return False
    on anything we can't read so we fail closed (no face emission)
    rather than emitting a stale event."""
    from edgevox.llm.tools import ToolCallResult

    if not isinstance(payload, ToolCallResult):
        return False
    return payload.name in MOVE_TOOLS and payload.ok


__all__ = ["MOVE_TOOLS", "RobotFaceEvent", "RobotFaceHook"]
