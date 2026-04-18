"""User preferences: piece set, board colours, persona.

Persisted via :class:`QSettings` so changes survive app restarts.
Exposed as a lightweight :class:`Settings` dataclass the UI reads and
the :class:`SettingsDialog` writes.

All shipped piece sets are MIT-licensed (Maurizio Monge's fantasy /
celtic / spatial). Board themes are pure hex colour pairs — free
to tweak, no asset licensing concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QSettings

# ----- piece sets -----

PIECE_SETS: dict[str, str] = {
    # key → human-readable label. Asset dir is "assets/pieces-<key>".
    "fantasy": "Fantasy (Staunton)",
    "celtic": "Celtic",
    "spatial": "Spatial (line art)",
}


def piece_set_dir(key: str) -> Path:
    root = Path(__file__).resolve().parent / "assets"
    if (root / f"pieces-{key}").is_dir():
        return root / f"pieces-{key}"
    return root / "pieces-fantasy"


# ----- board themes — light / dark square colour pairs -----

BOARD_THEMES: dict[str, tuple[str, str]] = {
    "wood": ("#ead8ba", "#a87d5a"),  # default, lichess-ish brown
    "green": ("#eeeed2", "#769656"),  # chess.com green
    "blue": ("#e0e6ee", "#7396bd"),  # lichess blue
    "gray": ("#dddddd", "#888888"),  # neutral
    "brown-dark": ("#e4c194", "#7f5338"),  # darker wood
    "night": ("#4a5b6e", "#1e2b3a"),  # dark on dark for late-night play
}

BOARD_THEME_LABELS: dict[str, str] = {
    "wood": "Wood (default)",
    "green": "Green",
    "blue": "Blue",
    "gray": "Gray",
    "brown-dark": "Dark wood",
    "night": "Night",
}


# ----- persona -----

PERSONAS = ("casual", "grandmaster", "trash_talker")
PERSONA_LABELS = {
    "casual": "Casual Club Player",
    "grandmaster": "Grandmaster",
    "trash_talker": "Trash-Talking Coach",
}


@dataclass
class Settings:
    """Current persisted preferences."""

    piece_set: str = "fantasy"
    board_theme: str = "wood"
    persona: str = "casual"
    voice_enabled: bool = True
    sfx_muted: bool = False

    @classmethod
    def load(cls) -> Settings:
        q = QSettings("EdgeVox", "RookApp")
        return cls(
            piece_set=str(q.value("piece_set", "fantasy")),
            board_theme=str(q.value("board_theme", "wood")),
            persona=str(q.value("persona", "casual")),
            voice_enabled=_bool(q.value("voice_enabled", True)),
            sfx_muted=_bool(q.value("sfx_muted", False)),
        )

    def save(self) -> None:
        q = QSettings("EdgeVox", "RookApp")
        q.setValue("piece_set", self.piece_set)
        q.setValue("board_theme", self.board_theme)
        q.setValue("persona", self.persona)
        q.setValue("voice_enabled", self.voice_enabled)
        q.setValue("sfx_muted", self.sfx_muted)


def _bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "on")
    return bool(v)


__all__ = [
    "BOARD_THEMES",
    "BOARD_THEME_LABELS",
    "PERSONAS",
    "PERSONA_LABELS",
    "PIECE_SETS",
    "Settings",
    "piece_set_dir",
]
