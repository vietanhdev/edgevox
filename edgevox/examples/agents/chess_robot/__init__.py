"""Rook — face-first voice chess robot.

A character-driven chess app where the star is a stylised SVG robot
face you play across from, not a board with a bot attached. The
existing :mod:`~edgevox.integrations.chess` stack (engine, environment,
tools, hooks, personas) does the chess work; everything in this
package is additive — a :class:`RobotFaceHook` that translates
:class:`ChessState` transitions into face-level signals (mood, gaze,
tempo) the web UI renders as the robot's expression.

The hook is plug-and-play — it composes with
:class:`~edgevox.integrations.chess.hooks.BoardStateInjectionHook` and
:class:`~edgevox.integrations.chess.hooks.MoveCommentaryHook` with no
coupling. Nothing in :mod:`edgevox.integrations.chess` needs to change.
"""

from __future__ import annotations

from edgevox.examples.agents.chess_robot.face_hook import (
    RobotFaceEvent,
    RobotFaceHook,
)
from edgevox.examples.agents.chess_robot.mood import (
    Mood,
    derive_mood,
    gaze_from_uci,
)

__all__ = [
    "Mood",
    "RobotFaceEvent",
    "RobotFaceHook",
    "derive_mood",
    "gaze_from_uci",
]
