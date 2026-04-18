"""Pure functions: ChessState → robot face signals.

Keeping the reductions pure (no I/O, no framework imports) means they
are trivially unit-testable and can be called from anywhere — the hook
today, a PGN-replay tool tomorrow, a MuJoCo arm expression driver next.

The face looks out at the user; the *robot* is the one playing. That
means mood is biased towards the engine side: engine winning → ``amused``
/ ``triumphant``; engine losing → ``worried`` / ``defeated``. A face
that smiled when the user was crushing it would break the illusion.
"""

from __future__ import annotations

from enum import Enum


class Mood(str, Enum):
    """Six-state expression alphabet.

    Deliberately small. Six states is enough for a face to feel alive
    without devolving into "here's a slightly different eyebrow angle"
    micro-management. The UI is responsible for tweening between them.
    """

    CALM = "calm"
    CURIOUS = "curious"
    AMUSED = "amused"
    WORRIED = "worried"
    TRIUMPHANT = "triumphant"
    DEFEATED = "defeated"


# ---------------------------------------------------------------------------
# Thresholds — single source of truth so tests can assert on the boundaries
# ---------------------------------------------------------------------------

# Centipawn eval thresholds from the *engine's* perspective (so a
# positive number always means "robot is winning"). The steps are
# deliberately wide — 200/500 cp is a full pawn / two pawns — so the
# face only shifts on decisive swings, not every mid-game wobble.
AMUSED_CP = 200
TRIUMPHANT_CP = 500
WORRIED_CP = -200
DEFEATED_CP = -500


def _engine_perspective_cp(eval_cp: int | None, engine_plays: str) -> int | None:
    """Flip the white-POV eval to the engine's perspective.

    :class:`ChessState.eval_cp` is white-POV (see
    :attr:`EngineMove.score_from_white`). The face belongs to the robot,
    so we normalise by the side the engine is playing before bucketing.
    """
    if eval_cp is None:
        return None
    return eval_cp if engine_plays == "white" else -eval_cp


def derive_mood(
    state: dict | object,
    *,
    engine_plays: str,
    persona: str = "casual",
) -> Mood:
    """Reduce a chess state snapshot into a single :class:`Mood`.

    Accepts either a :class:`~edgevox.integrations.chess.environment.ChessState`
    (dataclass) or its ``to_json()`` dict form — whichever is cheaper at
    the call site. Both shapes are used in this codebase.

    Args:
        state: :class:`ChessState` or its dict form.
        engine_plays: ``"white"`` or ``"black"`` — which side the robot
            is playing. Needed to flip the white-POV eval.
        persona: Influences edge cases (e.g. ``trash_talker`` stays
            ``amused`` even when winning decisively — humility isn't the
            character). Defaults to ``casual``.
    """
    get = _getter(state)

    if get("is_game_over"):
        winner = get("winner")
        if winner == engine_plays:
            return Mood.TRIUMPHANT
        if winner is None:
            return Mood.CALM  # draw — composed
        return Mood.DEFEATED

    eval_cp = _engine_perspective_cp(get("eval_cp"), engine_plays)
    mate_in = get("mate_in")
    if mate_in is not None:
        if engine_plays == "white":
            return Mood.TRIUMPHANT if mate_in > 0 else Mood.DEFEATED
        return Mood.TRIUMPHANT if mate_in < 0 else Mood.DEFEATED

    if eval_cp is not None:
        if eval_cp >= TRIUMPHANT_CP:
            return Mood.TRIUMPHANT if persona != "trash_talker" else Mood.AMUSED
        if eval_cp >= AMUSED_CP:
            return Mood.AMUSED
        if eval_cp <= DEFEATED_CP:
            return Mood.DEFEATED
        if eval_cp <= WORRIED_CP:
            return Mood.WORRIED

    # No decisive eval: a blunder classification from the *user* is still
    # worth reacting to ("oho, interesting move"). We don't flip mood for
    # the engine's own moves here — the eval shift will carry that.
    classification = get("last_move_classification")
    turn = get("turn")  # side to move AFTER the last move
    # last move was played by the *other* side, so:
    last_mover_is_user = turn == engine_plays
    if last_mover_is_user and classification in {"mistake", "blunder"}:
        return Mood.CURIOUS if persona == "grandmaster" else Mood.AMUSED

    return Mood.CALM


def gaze_from_uci(uci: str | None) -> tuple[float, float]:
    """Convert a last-move UCI (``e2e4``) into a normalised gaze vector.

    The robot *looks at the destination square* of the last move — where
    the piece just landed, which is where a human eye would naturally
    snap. Returns ``(x, y)`` in ``[-1, 1]``:

    - ``x`` → file, ``-1`` = a-file (left), ``+1`` = h-file (right)
    - ``y`` → rank, ``-1`` = rank 1 (bottom, towards us), ``+1`` = rank 8

    The SVG transform expects screen coordinates where +y is down, so
    callers may negate ``y`` before applying — but keeping board
    semantics pure here means this helper is reusable for the TUI
    widget or a future physical eye-servo driver.
    """
    if not uci or len(uci) < 4:
        return (0.0, 0.0)
    dst = uci[2:4].lower()
    file_ch, rank_ch = dst[0], dst[1]
    if not ("a" <= file_ch <= "h") or not ("1" <= rank_ch <= "8"):
        return (0.0, 0.0)
    file_idx = ord(file_ch) - ord("a")  # 0..7
    rank_idx = int(rank_ch) - 1  # 0..7
    x = (file_idx - 3.5) / 3.5  # -1..1
    y = (rank_idx - 3.5) / 3.5
    return (round(x, 3), round(y, 3))


def _getter(state: dict | object):
    """Field accessor that works for both dataclass + dict snapshots.

    ``ChessState.to_json()`` serialises the ``MoveClassification`` enum
    to its ``.value``; the raw dataclass carries the enum itself. We
    coerce to a string so downstream comparisons don't have to branch.
    """
    if isinstance(state, dict):

        def _get(key: str):
            v = state.get(key)
            return getattr(v, "value", v)

        return _get

    def _get(key: str):
        v = getattr(state, key, None)
        return getattr(v, "value", v)

    return _get


__all__ = [
    "AMUSED_CP",
    "DEFEATED_CP",
    "TRIUMPHANT_CP",
    "WORRIED_CP",
    "Mood",
    "derive_mood",
    "gaze_from_uci",
]
