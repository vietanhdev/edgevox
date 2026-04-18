"""Lottie-backed robot face — full animation fidelity.

Renders one of the JSON animations in ``assets/lottie/`` via
``rlottie-python`` (LGPL — safe as a dynamic-link library inside an
MIT app). The same mood / tempo vocabulary that drove the old SVG
face maps each state to a specific animation file; crossfading is
handled by restarting playback from frame 0 when the target changes.

If ``rlottie-python`` is not installed, callers fall back to the
pure-Qt :class:`RobotFaceWidget`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

log = logging.getLogger(__name__)

_LOTTIE_DIR = Path(__file__).resolve().parent / "assets" / "lottie"


# Mood/tempo → animation filename (minus ``.json``). Tempo overrides
# mood because the user's attention is on "what's happening right
# now", not the ambient emotional state. Matches the React RookApp
# mapping exactly so moving back and forth feels consistent.
TEMPO_ANIM: dict[str, str] = {
    "thinking": "robot_thinking",
    "speaking": "robot_speaking",
}

MOOD_ANIM: dict[str, str] = {
    "calm": "robot_idle",
    "curious": "robot_searching",
    "amused": "robot_happy",
    "worried": "robot_worried",
    "triumphant": "robot_celebrating",
    "defeated": "robot_sad",
}


def is_available() -> bool:
    """Whether rlottie-python can be imported + assets are present."""
    try:
        import rlottie_python  # noqa: F401
    except Exception:
        return False
    return _LOTTIE_DIR.is_dir() and any(_LOTTIE_DIR.glob("*.json"))


class LottieFaceWidget(QLabel):
    """A QLabel that plays a Lottie animation via rlottie.

    Single-animation at a time: changing mood/tempo picks the matching
    JSON, rebuilds the LottieAnimation, and restarts the frame timer.
    Renders at widget size using rlottie's native buffer → QImage.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(160, 160)
        self.setStyleSheet("background: transparent;")
        self._anim = None
        self._frame = 0
        self._frame_count = 1
        self._frame_rate = 30.0
        self._current_file: str | None = None
        self._mood = "calm"
        self._tempo = "idle"
        self._persona = "casual"

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        self._load_for_state()

    # ----- public API (mirrors RobotFaceWidget) -----

    def set_mood(self, mood: str) -> None:
        if mood == self._mood:
            return
        self._mood = mood
        self._load_for_state()

    def set_tempo(self, tempo: str) -> None:
        if tempo == self._tempo:
            return
        self._tempo = tempo
        self._load_for_state()

    def set_persona(self, persona: str) -> None:
        # Current animation pack is persona-agnostic; keep the setter
        # for API parity so the window can call it unconditionally.
        self._persona = persona

    # ----- loading + playback -----

    def _pick_file(self) -> str | None:
        """Resolve the animation file for the current mood/tempo."""
        name = TEMPO_ANIM.get(self._tempo) or MOOD_ANIM.get(self._mood) or "robot_idle"
        path = _LOTTIE_DIR / f"{name}.json"
        if path.is_file():
            return str(path)
        # Fall back to idle if the specific animation is missing.
        idle = _LOTTIE_DIR / "robot_idle.json"
        return str(idle) if idle.is_file() else None

    def _load_for_state(self) -> None:
        path = self._pick_file()
        if path is None or path == self._current_file:
            return
        try:
            from rlottie_python import LottieAnimation

            self._anim = LottieAnimation.from_file(path)
            self._frame_count = max(1, self._anim.lottie_animation_get_totalframe())
            self._frame_rate = max(1.0, self._anim.lottie_animation_get_framerate())
            self._frame = 0
            self._current_file = path
            self._timer.start(int(1000 / self._frame_rate))
            self._advance()
        except Exception:
            log.exception("Failed to load Lottie animation at %s", path)
            self._anim = None
            self._timer.stop()

    def _advance(self) -> None:
        if self._anim is None:
            return
        size = min(self.width(), self.height())
        if size <= 0:
            return
        try:
            pil = self._anim.render_pillow_frame(frame_num=self._frame, width=size, height=size)
        except Exception:
            log.exception("Lottie render failed; stopping animation")
            self._timer.stop()
            return
        img = QImage(
            pil.tobytes("raw", "RGBA"),
            pil.width,
            pil.height,
            QImage.Format.Format_RGBA8888,
        )
        self.setPixmap(QPixmap.fromImage(img))
        self._frame = (self._frame + 1) % self._frame_count

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._advance()


__all__ = ["LottieFaceWidget", "is_available"]
