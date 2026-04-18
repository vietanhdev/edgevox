"""Pluggable chess-engine backends for the chess agent.

Every backend implements the :class:`ChessEngine` Protocol — ``bestmove``
plus ``analyse``. Agent code calls the Protocol; swapping Stockfish ↔
Maia never touches the tool or hook layer.

Both shipped backends delegate to ``python-chess``'s UCI driver
(``chess.engine.SimpleEngine``), so the actual engine binary
(Stockfish, LC0) runs out-of-process. That's the clean MIT-vs-GPL
boundary: EdgeVox stays MIT; the GPL engine binary is a user-installed
subprocess invoked over stdio.
"""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import chess
import chess.engine

log = logging.getLogger(__name__)


class EngineUnavailable(RuntimeError):
    """Raised when the requested engine binary can't be located or launched.

    Agent code catches this to fall back to a simpler backend or to
    surface a friendly "install stockfish" message to the user.
    """


@dataclass
class EngineMove:
    """One engine recommendation plus the evaluation that produced it."""

    uci: str
    san: str
    eval_cp: int | None = None
    mate_in: int | None = None
    pv: list[str] = field(default_factory=list)
    depth: int | None = None

    @property
    def score_from_white(self) -> int | None:
        """Centipawn score from white's perspective; ``None`` if mate-only.

        Agent UIs (eval bar) want a signed score that doesn't flip with
        the side-to-move, so we normalise here rather than leaking
        python-chess's POV semantics.
        """
        if self.mate_in is not None:
            return 10_000 if self.mate_in > 0 else -10_000
        return self.eval_cp


@runtime_checkable
class ChessEngine(Protocol):
    """Minimum surface every backend implements."""

    name: str

    def bestmove(self, board: chess.Board, *, time_limit: float = 1.0) -> EngineMove: ...
    def analyse(self, board: chess.Board, *, depth: int = 12) -> EngineMove: ...
    def close(self) -> None: ...


class _SimpleEngineWrapper:
    """Shared plumbing for any UCI engine driven via python-chess.

    Subclasses set ``name`` + how to locate the binary + any UCI
    options. This base class handles:

    - lazy subprocess launch on first use (construction is cheap, so
      tests can skip if no binary is available without paying startup
      cost),
    - a re-entrant lock around every call so a single engine instance
      is safe to share across threads (the chess agent may have
      ``analyze_position`` and ``engine_move`` issued in parallel from
      the LLM loop's parallel dispatcher),
    - clean shutdown via :meth:`close`.
    """

    name: str = "uci-engine"
    _binary_hint: ClassVar[str | Path] = ""
    _uci_options: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        *,
        binary: str | Path | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self._binary = Path(binary) if binary else self._find_binary()
        if not self._binary:
            raise EngineUnavailable(
                f"{self.name}: no binary found. Set binary=... or install the "
                f"engine and place it on $PATH (hint: {self._binary_hint!r})."
            )
        # Validate at construction so callers fail fast with a clear error
        # instead of a deferred subprocess-launch failure three calls deep.
        if not self._binary.is_file():
            raise EngineUnavailable(f"{self.name}: binary {self._binary} is not a file")
        self._options = {**self._uci_options, **(options or {})}
        self._engine: chess.engine.SimpleEngine | None = None
        self._lock = threading.RLock()

    def _find_binary(self) -> Path | None:
        """Resolve the engine binary via ``$PATH``.

        Subclasses override to look under common install prefixes first.
        Returns ``None`` when nothing is found so the caller can raise a
        helpful ``EngineUnavailable``.
        """
        if self._binary_hint:
            found = shutil.which(str(self._binary_hint))
            if found:
                return Path(found)
        return None

    # ----- lifecycle -----

    def _ensure_started(self) -> chess.engine.SimpleEngine:
        with self._lock:
            if self._engine is None:
                try:
                    self._engine = chess.engine.SimpleEngine.popen_uci(str(self._binary))
                except Exception as e:
                    raise EngineUnavailable(f"{self.name}: failed to launch {self._binary}: {e}") from e
                if self._options:
                    try:
                        self._engine.configure(self._options)
                    except chess.engine.EngineError as e:
                        log.warning("%s: configure(%r) rejected: %s", self.name, self._options, e)
            return self._engine

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                try:
                    self._engine.quit()
                except Exception:
                    log.exception("%s: quit() failed", self.name)
                self._engine = None

    # Context-manager sugar — useful for one-shot test runs.
    def __enter__(self) -> _SimpleEngineWrapper:
        self._ensure_started()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ----- ChessEngine protocol -----

    def bestmove(self, board: chess.Board, *, time_limit: float = 1.0) -> EngineMove:
        engine = self._ensure_started()
        with self._lock:
            result = engine.play(
                board,
                chess.engine.Limit(time=time_limit),
                info=chess.engine.INFO_SCORE | chess.engine.INFO_PV,
            )
        move = result.move
        if move is None:
            raise EngineUnavailable(f"{self.name}: engine returned no move (board may be terminal)")
        san = board.san(move)
        info = getattr(result, "info", None) or {}
        return _info_to_move(move, san, info, board)

    def analyse(self, board: chess.Board, *, depth: int = 12) -> EngineMove:
        engine = self._ensure_started()
        with self._lock:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
        if isinstance(info, list):
            info = info[0] if info else {}
        pv = info.get("pv") or []
        if not pv:
            raise EngineUnavailable(f"{self.name}: analysis returned no principal variation")
        move = pv[0]
        san = board.san(move)
        return _info_to_move(move, san, info, board, depth=depth)


def _info_to_move(
    move: chess.Move,
    san: str,
    info: dict[str, Any],
    board: chess.Board,
    *,
    depth: int | None = None,
) -> EngineMove:
    score = info.get("score")
    eval_cp: int | None = None
    mate_in: int | None = None
    if score is not None:
        # python-chess reports score POV side-to-move; flip to white to
        # match the eval-bar semantics agent code wants. ``white()`` on
        # a :class:`PovScore` handles the flip for us.
        try:
            white_score = score.white()
            eval_cp = white_score.score(mate_score=10_000)
            mate_in = white_score.mate()
        except Exception:
            log.debug("Could not decode engine score %r", score, exc_info=True)
    pv_list = info.get("pv") or [move]
    try:
        # SAN-render a copy of the board so we don't mutate the caller's.
        board_copy = board.copy(stack=False)
        pv_san: list[str] = []
        for m in pv_list[:6]:
            pv_san.append(board_copy.san(m))
            board_copy.push(m)
    except Exception:
        pv_san = []
    return EngineMove(
        uci=move.uci(),
        san=san,
        eval_cp=eval_cp,
        mate_in=mate_in,
        pv=pv_san,
        depth=depth if depth is not None else info.get("depth"),
    )


# ---------------------------------------------------------------------------
# Stockfish
# ---------------------------------------------------------------------------


class StockfishEngine(_SimpleEngineWrapper):
    """Stockfish backend. Strong, skill-level tunable (0-20).

    Skill level is the main knob for difficulty. Threads and Hash lift
    raw strength but don't change style; skill level 0 plays visibly
    weaker. See the Stockfish docs for the full UCI option list.
    """

    name = "stockfish"
    _binary_hint = "stockfish"

    def __init__(
        self,
        *,
        binary: str | Path | None = None,
        skill: int = 20,
        threads: int = 1,
        hash_mb: int = 64,
        options: dict[str, Any] | None = None,
    ) -> None:
        skill = max(0, min(20, int(skill)))
        merged: dict[str, Any] = {
            "Skill Level": skill,
            "Threads": max(1, int(threads)),
            "Hash": max(16, int(hash_mb)),
        }
        if options:
            merged.update(options)
        super().__init__(binary=binary, options=merged)
        self.skill = skill

    def _find_binary(self) -> Path | None:
        for candidate in ("stockfish", "stockfish-bin"):
            found = shutil.which(candidate)
            if found:
                return Path(found)
        # Common install prefixes on desktop Linux / macOS.
        for explicit in ("/usr/games/stockfish", "/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish"):
            p = Path(explicit)
            if p.is_file():
                return p
        return None


# ---------------------------------------------------------------------------
# Maia (LC0 + human-training weights)
# ---------------------------------------------------------------------------


class MaiaEngine(_SimpleEngineWrapper):
    """Maia backend — LC0 loaded with a Maia weight file.

    Maia plays like a human at a specific ELO (1100-1900 for the
    original, 600-2600 for Maia-2). The model is selected by pointing
    LC0 at the matching ``.pb.gz`` / ``.onnx`` weight file via the
    ``WeightsFile`` UCI option.

    Args:
        binary: path to the ``lc0`` executable. Defaults to ``$PATH`` lookup.
        weights: path to the Maia ``.pb.gz`` for the target ELO. Required.
        elo: advisory rating label used for telemetry / reporting only —
            LC0 always plays whatever the weights file was trained for.
        options: extra UCI options merged last.
    """

    name = "maia"
    _binary_hint = "lc0"

    def __init__(
        self,
        *,
        binary: str | Path | None = None,
        weights: str | Path,
        elo: int = 1500,
        options: dict[str, Any] | None = None,
    ) -> None:
        weights_path = Path(weights).expanduser()
        if not weights_path.is_file():
            raise EngineUnavailable(
                f"maia: weights file {weights_path} not found. Download the matching "
                f"maia-{elo}.pb.gz from https://maiachess.com/ or the CSSLab repo."
            )
        merged: dict[str, Any] = {
            # LC0's one-shot "play without search" mode is what makes
            # Maia mimic humans — searching amplifies the policy away
            # from the training distribution. One node = pure policy.
            "WeightsFile": str(weights_path),
            "Nodes": 1,
        }
        if options:
            merged.update(options)
        super().__init__(binary=binary, options=merged)
        self.elo = elo
        self.weights_path = weights_path

    def _find_binary(self) -> Path | None:
        for candidate in ("lc0", "lczero"):
            found = shutil.which(candidate)
            if found:
                return Path(found)
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_engine(kind: str, **kwargs: Any) -> ChessEngine:
    """Resolve a ``kind`` string to a configured :class:`ChessEngine`.

    Centralising this here keeps the ``--engine`` CLI flag on
    ``chess_partner.py`` a thin lookup instead of a sprawl of if-trees.
    Unknown kinds raise :class:`EngineUnavailable` so the CLI can print
    a clear error and list what was expected.
    """
    key = kind.lower().strip()
    if key == "stockfish":
        return StockfishEngine(**kwargs)
    if key == "maia":
        return MaiaEngine(**kwargs)
    raise EngineUnavailable(f"unknown chess engine {kind!r} (expected: stockfish, maia)")


__all__ = [
    "ChessEngine",
    "EngineMove",
    "EngineUnavailable",
    "MaiaEngine",
    "StockfishEngine",
    "build_engine",
]
