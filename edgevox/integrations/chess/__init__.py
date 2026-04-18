"""Chess integration — pluggable engines + environment + hooks + personas.

Brings a chess-playing voice agent to EdgeVox by plugging into the same
``ctx.deps`` + tool + hook surfaces as the robot examples. The engine
layer is a Protocol (:class:`ChessEngine`) with two shipped backends:

- :class:`StockfishEngine` — wraps the Stockfish UCI binary. Strong,
  skill-level-tunable.
- :class:`MaiaEngine` — wraps LC0 + Maia weights for human-like play at
  a target ELO.

Both are invoked as UCI subprocesses via ``python-chess``'s
:class:`chess.engine.SimpleEngine`; no GPL Python bindings are linked.
Users install the binary separately (``apt install stockfish`` /
``brew install lc0``) and the wrapper finds it on ``$PATH`` or at an
explicit path.

:class:`ChessEnvironment` owns the live :class:`chess.Board` and
routes user moves + engine moves through the selected engine. It
implements the :class:`~edgevox.agents.sim.SimEnvironment` protocol
so the agent harness can hand it to ``ctx.deps`` identically to
:class:`~edgevox.integrations.sim.mujoco_arm.MujocoArmEnvironment`.
"""

from __future__ import annotations

from edgevox.integrations.chess.analytics import (
    MoveClassification,
    classify_move,
    opening_name,
    win_probability,
)
from edgevox.integrations.chess.engine import (
    ChessEngine,
    EngineMove,
    EngineUnavailable,
    MaiaEngine,
    StockfishEngine,
    build_engine,
)
from edgevox.integrations.chess.environment import ChessEnvironment, ChessState
from edgevox.integrations.chess.hooks import BoardStateInjectionHook, MoveCommentaryHook
from edgevox.integrations.chess.personas import PERSONAS, Persona, resolve_persona

__all__ = [
    "PERSONAS",
    "BoardStateInjectionHook",
    "ChessEngine",
    "ChessEnvironment",
    "ChessState",
    "EngineMove",
    "EngineUnavailable",
    "MaiaEngine",
    "MoveClassification",
    "MoveCommentaryHook",
    "Persona",
    "StockfishEngine",
    "build_engine",
    "classify_move",
    "opening_name",
    "resolve_persona",
    "win_probability",
]
