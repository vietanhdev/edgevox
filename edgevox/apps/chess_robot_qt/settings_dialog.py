"""Settings dialog — piece set, board theme, persona, voice, SFX."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QLabel,
    QVBoxLayout,
)

from edgevox.apps.chess_robot_qt.settings import (
    BOARD_THEME_LABELS,
    BOARD_THEMES,
    PERSONA_LABELS,
    PERSONAS,
    PIECE_SETS,
    Settings,
)


class SettingsDialog(QDialog):
    """Modal preferences panel — edit :class:`Settings` in place.

    Emits :attr:`changed` with the new :class:`Settings` instance on OK;
    the window applies what it can live (piece set, theme, SFX mute)
    and flags a restart hint for things that need a fresh agent
    (persona, voice pipeline).
    """

    changed = Signal(Settings)

    def __init__(self, current: Settings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(360)
        self.setStyleSheet(
            "QDialog { background: #0b111a; color: #dbe4ef; } "
            "QLabel { color: #dbe4ef; } "
            "QComboBox, QCheckBox { color: #dbe4ef; } "
            "QComboBox { background: #10161f; border: 1px solid #1e2b3a; "
            "padding: 4px 8px; border-radius: 5px; } "
            "QComboBox QAbstractItemView { background: #10161f; color: #dbe4ef; "
            "selection-background-color: #2b3c58; }"
        )

        self._current = current

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(14)

        header = QLabel("Preferences")
        header.setStyleSheet("font-size: 15px; font-weight: 600; color: #dbe4ef;")
        layout.addWidget(header)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

        self._piece_combo = _combo(PIECE_SETS, current.piece_set)
        form.addRow("Piece set", self._piece_combo)

        self._theme_combo = _combo(BOARD_THEME_LABELS, current.board_theme)
        form.addRow("Board theme", self._theme_combo)

        persona_items = {slug: PERSONA_LABELS[slug] for slug in PERSONAS}
        self._persona_combo = _combo(persona_items, current.persona)
        form.addRow("Persona", self._persona_combo)

        self._voice_checkbox = QCheckBox("Enable voice input")
        self._voice_checkbox.setChecked(current.voice_enabled)
        form.addRow("", self._voice_checkbox)

        self._sfx_checkbox = QCheckBox("Mute sound effects")
        self._sfx_checkbox.setChecked(current.sfx_muted)
        form.addRow("", self._sfx_checkbox)

        layout.addLayout(form)

        # Hint: some settings require a restart (persona, voice).
        hint = QLabel(
            "Piece set + theme + mute apply instantly. Persona / voice changes take effect on the next launch."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8796a8; font-size: 11px; padding-top: 4px;")
        layout.addWidget(hint)

        # Theme preview swatches — tiny row under the theme combo.
        self._preview = _ThemePreview()
        self._theme_combo.currentTextChanged.connect(lambda: self._preview.set_theme(self._selected_theme()))
        self._preview.set_theme(current.board_theme)
        layout.addWidget(self._preview)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.setStyleSheet(
            "QPushButton { background: #1e2b3a; color: #dbe4ef; padding: 6px 14px; "
            "border: none; border-radius: 5px; } "
            "QPushButton:hover { background: #2b3c58; } "
            "QPushButton:default { background: #34d399; color: #0a0e14; }"
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _selected_theme(self) -> str:
        return _selected_key(self._theme_combo, BOARD_THEME_LABELS)

    def _on_accept(self) -> None:
        new = Settings(
            piece_set=_selected_key(self._piece_combo, PIECE_SETS),
            board_theme=_selected_key(self._theme_combo, BOARD_THEME_LABELS),
            persona=_selected_key(self._persona_combo, {s: PERSONA_LABELS[s] for s in PERSONAS}),
            voice_enabled=self._voice_checkbox.isChecked(),
            sfx_muted=self._sfx_checkbox.isChecked(),
        )
        new.save()
        self.changed.emit(new)
        self.accept()


def _combo(items: dict[str, str], current_key: str) -> QComboBox:
    combo = QComboBox()
    for key, label in items.items():
        combo.addItem(label, key)
    # Prefer userData match; fall back to the first item.
    idx = next((i for i, k in enumerate(items) if k == current_key), 0)
    combo.setCurrentIndex(idx)
    return combo


def _selected_key(combo: QComboBox, items: dict[str, str]) -> str:
    text = combo.currentText()
    for key, label in items.items():
        if label == text:
            return key
    return next(iter(items))


class _ThemePreview(QFrame):
    """Eight-square stripe showing the currently-selected board colours."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(24)
        self._theme = "wood"

    def set_theme(self, theme: str) -> None:
        self._theme = theme
        self.update()

    def paintEvent(self, _event) -> None:
        from PySide6.QtGui import QBrush, QColor, QPainter

        light_hex, dark_hex = BOARD_THEMES.get(self._theme, BOARD_THEMES["wood"])
        p = QPainter(self)
        try:
            sz = self.width() / 8
            for i in range(8):
                colour = QColor(light_hex) if i % 2 == 0 else QColor(dark_hex)
                p.fillRect(int(i * sz), 0, int(sz) + 1, self.height(), QBrush(colour))
        finally:
            p.end()


__all__ = ["SettingsDialog"]
