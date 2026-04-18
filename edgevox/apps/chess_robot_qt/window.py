"""Main window — custom titlebar + board/face/chat layout.

Matches the Tauri/React UI 1:1 where it makes sense:

    ┌─────────────────────────────────────── RookApp  — ◻ ✕ ┐
    │ ● RookApp · online     your turn   🎤  ↻  ☰        │
    ├──────────────────────────────────┬──────────────────┤
    │                                  │  [ robot face ]  │
    │      [ chess board ]             │                  │
    │                                  │  [ chat log    ] │
    │   captured pieces strip          │                  │
    │   move history                   │  [ input ▶ ]     │
    └──────────────────────────────────┴──────────────────┘

Frameless + our own title bar, same pattern as the Tauri app. No
browser, no webview — pure Qt widgets.
"""

from __future__ import annotations

import logging

import qtawesome as qta
from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from edgevox.apps.chess_robot_qt.board import ChessBoardView
from edgevox.apps.chess_robot_qt.bridge import RookBridge
from edgevox.apps.chess_robot_qt.chat import ChatView
from edgevox.apps.chess_robot_qt.face import RobotFaceWidget
from edgevox.apps.chess_robot_qt.lottie_face import LottieFaceWidget
from edgevox.apps.chess_robot_qt.lottie_face import is_available as lottie_available
from edgevox.apps.chess_robot_qt.settings import Settings
from edgevox.apps.chess_robot_qt.settings_dialog import SettingsDialog
from edgevox.apps.chess_robot_qt.tts import TTSWorker
from edgevox.apps.chess_robot_qt.voice import VoiceWorker

log = logging.getLogger(__name__)


_PERSONA_ACCENT = {
    "grandmaster": "#7aa8ff",
    "casual": "#ffb066",
    "trash_talker": "#ff5ad1",
}
_DEFAULT_ACCENT = "#34d399"


class RookWindow(QMainWindow):
    """Frameless main window with custom titlebar."""

    def __init__(self, bridge: RookBridge) -> None:
        super().__init__()
        self._bridge = bridge
        self._settings = Settings.load()
        # Sync the loaded settings into the bridge config when present —
        # e.g. restore the previously-chosen persona at launch.
        if self._settings.persona != bridge.config.persona:
            bridge.config.persona = self._settings.persona
        self._accent = QColor(_PERSONA_ACCENT.get(bridge.config.persona, _DEFAULT_ACCENT))
        self.setWindowTitle("RookApp")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.resize(1280, 820)
        self.setMinimumSize(980, 640)
        self.setStyleSheet("QMainWindow { background: #060a12; } QLabel { color: #dbe4ef; }")

        root = QWidget()
        self.setCentralWidget(root)
        col = QVBoxLayout(root)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)

        # One unified top bar: brand + drag region + status + turn pill
        # + action buttons + window controls. Replaces the old
        # titlebar/appbar pair (which duplicated the brand label).
        self._app_bar = _AppBar(self._accent, self)
        self._app_bar.new_game_clicked.connect(self._on_new_game)
        self._app_bar.mic_clicked.connect(self._on_mic_clicked)
        self._app_bar.menu_triggered.connect(self._on_menu_action)
        self._app_bar.close_clicked.connect(self.close)
        self._app_bar.minimize_clicked.connect(self.showMinimized)
        self._app_bar.toggle_maximize_clicked.connect(self._toggle_max)
        col.addWidget(self._app_bar)

        # Main area.
        main = QWidget()
        main_row = QHBoxLayout(main)
        main_row.setContentsMargins(14, 10, 14, 10)
        main_row.setSpacing(14)

        # Left column — framed board + history strip.
        left = QVBoxLayout()
        left.setSpacing(6)
        board_frame = QFrame()
        board_frame.setStyleSheet(
            "QFrame { background: #0b111a; border: 1px solid #1b2634; border-radius: 10px; padding: 8px; }"
        )
        bf_col = QVBoxLayout(board_frame)
        bf_col.setContentsMargins(10, 10, 10, 10)
        self._board = ChessBoardView()
        self._board.move_requested.connect(self._on_board_move)
        self._board.setMinimumSize(420, 420)
        self._board.set_orientation(bridge.config.user_plays)
        self._board.set_piece_set(self._settings.piece_set)
        self._board.set_theme(self._settings.board_theme)
        bf_col.addWidget(self._board)
        left.addWidget(board_frame, stretch=1)
        self._history = QLabel("no moves yet")
        self._history.setStyleSheet("color: #9aa7b9; font-family: monospace; font-size: 11px; padding: 2px 8px;")
        self._history.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(self._history)
        main_row.addLayout(left, stretch=60)

        # Right column — face, chat, input.
        right = QVBoxLayout()
        right.setSpacing(10)
        face_frame = QFrame()
        face_frame.setStyleSheet("QFrame { background: #0b111a; border: 1px solid #1b2634; border-radius: 10px; }")
        face_col = QVBoxLayout(face_frame)
        face_col.setContentsMargins(10, 10, 10, 6)
        face_col.setSpacing(4)
        self._face = LottieFaceWidget() if lottie_available() else RobotFaceWidget()
        self._face.setFixedHeight(220)
        self._face.set_persona(bridge.config.persona)
        face_col.addWidget(self._face, stretch=0)
        self._persona_label = QLabel(bridge.config.persona.replace("_", " ").title())
        self._persona_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._persona_label.setStyleSheet(
            f"color: {self._accent.name()}; font-family: monospace; font-size: 11px; "
            "letter-spacing: 2px; text-transform: uppercase;"
        )
        face_col.addWidget(self._persona_label, stretch=0)
        right.addWidget(face_frame, stretch=0)

        self._reply_label = QLabel("")
        self._reply_label.setWordWrap(True)
        self._reply_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._reply_label.setStyleSheet("color: #9aa7b9; font-size: 13px; padding: 6px 12px;")
        self._reply_label.setMinimumHeight(34)
        right.addWidget(self._reply_label, stretch=0)

        self._chat = ChatView(self._accent)
        right.addWidget(self._chat, stretch=1)

        self._input = _InputBar(self._accent)
        self._input.submitted.connect(self._on_submit)
        right.addWidget(self._input, stretch=0)

        main_row.addLayout(right, stretch=40)
        col.addWidget(main, stretch=1)

        # Keyboard shortcuts live in the Menu dropdown now — no need
        # for a duplicate footer strip here.

        # Wire the bridge → UI signals.
        b = bridge.signals
        b.state_changed.connect(self._on_state)
        b.chess_state_changed.connect(self._on_chess_state)
        b.face_changed.connect(self._on_face)
        b.reply_finalised.connect(self._on_reply)
        b.user_echo.connect(self._on_user_echo)
        b.error.connect(self._on_error)
        b.ready.connect(self._on_ready)
        b.load_progress.connect(lambda t: self._app_bar.set_status(t))

        # Start the bridge's background load of LLM + engine.
        self._app_bar.set_status("loading…")
        self._input.set_enabled(False)
        bridge.start()

        # TTS — kokoro loads lazily on a worker when the first reply
        # comes in, so we don't pay the ~300 MB ONNX cost at launch.
        # Mute means don't even load the model.
        self._tts: TTSWorker | None = None
        if not self._settings.sfx_muted:
            self._tts = TTSWorker(parent=self)
            self._tts.error.connect(lambda msg: log.warning("TTS: %s", msg))
            self._tts.started.connect(self._on_tts_started)
            self._tts.finished.connect(self._on_tts_finished)
            # Warmup in the background — no UI status spam; failures
            # downgrade to text-only silently.
            self._tts.start()

    # ----- bridge signal handlers -----

    def _on_ready(self) -> None:
        self._app_bar.set_status("online")
        self._input.set_enabled(True)
        # Prime the board with the starting position.
        snap = self._bridge.snapshot()
        if snap is not None:
            self._board.set_state(snap)

    def _on_state(self, state: str) -> None:
        label = {
            "listening": "your turn",
            "thinking": "rook thinking…",
            "speaking": "rook replying…",
        }.get(state, state)
        self._app_bar.set_turn_label(label, highlight=state == "listening")
        self._face.set_tempo(state)

    def _on_chess_state(self, state) -> None:
        self._board.set_state(state)
        # Move-history strip.
        moves = getattr(state, "san_history", []) or []
        if not moves:
            self._history.setText("no moves yet")
        else:
            pairs: list[str] = []
            for i in range(0, len(moves), 2):
                w = moves[i]
                b = moves[i + 1] if i + 1 < len(moves) else ""
                pairs.append(f"{i // 2 + 1}. {w} {b}".rstrip())
            self._history.setText("   ".join(pairs[-8:]))
        # Engine-move chip.
        if (
            state.last_move_san
            and state.turn == self._bridge.config.user_plays
            and state.ply >= 2
            and state.ply % 2 == 0
        ):
            self._chat.add_move_chip("rook", state.last_move_san)

    def _on_face(self, payload: dict) -> None:
        mood = payload.get("mood", "calm")
        tempo = payload.get("tempo", "idle")
        persona = payload.get("persona", self._bridge.config.persona)
        self._face.set_mood(mood)
        self._face.set_tempo(tempo)
        self._face.set_persona(persona)
        self._persona_label.setText(persona.replace("_", " ").title())

    def _on_reply(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._reply_label.setText(text)
        self._chat.add_rook(text)
        if self._tts is not None:
            self._tts.speak(text)

    def _on_tts_started(self) -> None:
        # While TTS plays we mute mic input so Rook's own voice doesn't
        # loop back through STT. Keep the UI-level active flag intact
        # so the mic button visually represents what the user picked.
        voice = getattr(self, "_voice", None)
        if voice is not None and getattr(self, "_voice_active", False):
            voice.set_listening(False)
        self._face.set_tempo("speaking")

    def _on_tts_finished(self) -> None:
        voice = getattr(self, "_voice", None)
        if voice is not None and getattr(self, "_voice_active", False):
            voice.set_listening(True)
        self._face.set_tempo("idle")

    def _on_user_echo(self, text: str) -> None:
        self._reply_label.setText("")

    def _on_error(self, msg: str) -> None:
        self._app_bar.set_status(msg, error=True)
        QTimer.singleShot(4500, lambda: self._app_bar.set_status("online"))

    # ----- user actions -----

    def _on_submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._chat.add_user(text)
        self._bridge.submit_text(text)

    def _on_board_move(self, uci: str) -> None:
        self._chat.add_user(uci)
        self._bridge.submit_text(uci)

    def _on_new_game(self) -> None:
        self._chat.clear()
        self._bridge.submit_text("new game")

    def _on_mic_clicked(self) -> None:
        """Toggle voice input. Lazy-starts the VoiceWorker on first
        click so we don't pay Whisper's startup cost up-front — voice
        users opt in; text users never load STT."""
        if not hasattr(self, "_voice"):
            if not self._settings.voice_enabled:
                self._on_error("Voice is disabled in Settings — enable it there first.")
                return
            self._voice = VoiceWorker(language="en", parent=self)
            self._voice.transcript.connect(self._on_voice_transcript)
            self._voice.error.connect(self._on_error)
            self._voice.loading.connect(lambda on: self._app_bar.set_status("warming up mic…" if on else "online"))
            self._voice.ready.connect(lambda: self._voice_enable_listen())
            self._voice.start()
            self._voice_active = True
            self._app_bar.set_mic_active(True)
            return
        # Subsequent clicks toggle the listening gate.
        self._voice_active = not getattr(self, "_voice_active", False)
        self._voice.set_listening(self._voice_active)
        self._app_bar.set_mic_active(self._voice_active)

    def _voice_enable_listen(self) -> None:
        self._voice.set_listening(getattr(self, "_voice_active", True))

    def _on_voice_transcript(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._chat.add_user(text)
        self._bridge.submit_text(text)

    def _on_menu_action(self, action: str) -> None:
        if action == "new_game":
            self._on_new_game()
        elif action == "settings":
            self._open_settings()
        elif action == "about":
            self._app_bar.set_status("RookApp — voice chess on EdgeVox.", error=False)
            QTimer.singleShot(4500, lambda: self._app_bar.set_status("online"))

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._settings, self)
        dlg.changed.connect(self._apply_settings)
        dlg.exec()

    def _apply_settings(self, new: Settings) -> None:
        """Apply what we can live; flag the rest as needing a restart."""
        self._board.set_piece_set(new.piece_set)
        self._board.set_theme(new.board_theme)
        if new.sfx_muted != self._settings.sfx_muted:
            # SFX are a future hook — flag for later.
            pass
        # Persona + voice require a restart; show a hint.
        needs_restart = new.persona != self._settings.persona or new.voice_enabled != self._settings.voice_enabled
        self._settings = new
        if needs_restart:
            self._on_error("Persona / voice change applies next launch.")

    # ----- window chrome -----

    def _toggle_max(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_N and event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        ):
            self._on_new_game()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self._tts is not None:
            self._tts.stop()
        voice = getattr(self, "_voice", None)
        if voice is not None:
            voice.stop()
        self._bridge.close()
        super().closeEvent(event)


# ----- sub-widgets -----


class _AppBar(QFrame):
    """Unified title + status bar. Drag the empty area to move the
    frameless window; click the brand area for the same effect."""

    close_clicked = Signal()
    minimize_clicked = Signal()
    toggle_maximize_clicked = Signal()
    new_game_clicked = Signal()
    mic_clicked = Signal()
    menu_triggered = Signal(str)

    def __init__(self, accent: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setStyleSheet("background: rgba(10, 14, 22, 0.85); border-bottom: 1px solid #141d2a;")
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 0, 0, 0)
        row.setSpacing(10)

        # Brand cluster: two rows stacked in a QVBoxLayout so the
        # status caption sits under the brand instead of running
        # alongside it (which read cluttered before).
        brand_col = QVBoxLayout()
        brand_col.setSpacing(0)
        brand_col.setContentsMargins(0, 4, 0, 4)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(6)
        brand_row.setContentsMargins(0, 0, 0, 0)
        self._status_dot = QLabel()
        self._status_dot.setFixedSize(8, 8)
        self._status_dot.setStyleSheet(
            f"background: {accent.name()}; border-radius: 4px; "
            "min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;"
        )
        self._status_dot.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        brand_row.addWidget(self._status_dot)
        brand_name = QLabel("RookApp")
        brand_name.setStyleSheet("color: #dbe4ef; font-weight: 600; font-size: 13px; letter-spacing: 0.4px;")
        brand_name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        brand_row.addWidget(brand_name)
        brand_row.addStretch(0)
        brand_col.addLayout(brand_row)

        self._status_text = QLabel("booting")
        self._status_text.setStyleSheet("color: #8796a8; font-family: monospace; font-size: 10px; padding-left: 14px;")
        self._status_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        brand_col.addWidget(self._status_text)

        row.addLayout(brand_col)
        row.addSpacing(6)

        # Middle area: pure drag region.
        row.addStretch(1)

        self._turn_pill = QLabel("Your turn")
        self._turn_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._turn_pill.setMinimumWidth(128)
        self._turn_pill.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        row.addWidget(self._turn_pill)

        self._mic_btn = _IconButton("fa5s.microphone", "Talk to Rook")
        self._mic_btn.clicked.connect(self.mic_clicked.emit)
        row.addWidget(self._mic_btn)

        new_btn = _IconButton("fa5s.redo-alt", "New game")
        new_btn.clicked.connect(self.new_game_clicked.emit)
        row.addWidget(new_btn)

        self._menu_btn = _IconButton("fa5s.bars", "Menu")
        self._menu_btn.clicked.connect(self._open_menu)
        row.addWidget(self._menu_btn)

        # Window controls on the far right — minimise / maximise / close.
        for icon_name, signal in [
            ("mdi.window-minimize", self.minimize_clicked),
            ("mdi.window-maximize", self.toggle_maximize_clicked),
            ("mdi.close", self.close_clicked),
        ]:
            btn = QToolButton()
            btn.setIcon(qta.icon(icon_name, color="#9aa7b9"))
            btn.setFixedSize(46, 40)
            btn.setAutoRaise(True)
            close_hover = "#e53935" if icon_name == "mdi.close" else "rgba(255, 255, 255, 0.07)"
            btn.setStyleSheet(
                "QToolButton { background: transparent; border: none; } "
                f"QToolButton:hover {{ background: {close_hover}; }}"
            )
            btn.clicked.connect(signal.emit)
            row.addWidget(btn)

        self._accent = accent
        self._drag_offset: QPoint | None = None
        self.set_status("booting", error=False)

    # ----- drag / maximise-on-double-click -----

    def _child_at(self, event: QMouseEvent) -> QWidget | None:
        """Return the child widget directly under the cursor (or None
        if the click landed on the bare QFrame background). We use this
        so clicks on buttons / the turn pill / labels DON'T trigger a
        window drag — they handle their own clicks."""
        return self.childAt(event.position().toPoint())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # QToolButton + QPushButton handle their own clicks; dragging
        # from them should do nothing. QLabels / stretch / the bare
        # frame allow drag.
        if event.button() == Qt.MouseButton.LeftButton and not isinstance(
            self._child_at(event), (QToolButton, QPushButton)
        ):
            top = self.window()
            # Prefer compositor-owned system move — the only drag path
            # that works on Wayland (where QWidget.move is a no-op) and
            # keeps drag smooth on X11. Falls through to a manual
            # offset-based drag if the platform can't service it.
            handle = top.windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
            self._drag_offset = event.globalPosition().toPoint() - top.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            top = self.window()
            top.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            target = self._child_at(event)
            if not isinstance(target, (QToolButton, QPushButton)):
                self.toggle_maximize_clicked.emit()

    def set_status(self, text: str, *, error: bool = False) -> None:
        colour = "#ef4444" if error else self._accent.name()
        text_colour = "#ef4444" if error else "#8796a8"
        self._status_dot.setStyleSheet(
            f"background: {colour}; border-radius: 4px; "
            "min-width: 8px; max-width: 8px; min-height: 8px; max-height: 8px;"
        )
        # Elide long status messages at a reasonable width so they don't
        # push the action buttons off the bar.
        self._status_text.setText(text)
        self._status_text.setStyleSheet(
            f"color: {text_colour}; font-family: monospace; font-size: 11px; padding-left: 4px;"
        )

    def set_turn_label(self, label: str, *, highlight: bool) -> None:
        self._turn_pill.setText(label)
        if highlight:
            r, g, b = self._accent.red(), self._accent.green(), self._accent.blue()
            self._turn_pill.setStyleSheet(
                f"background: rgba({r}, {g}, {b}, 0.18); "
                f"border: 1px solid {self._accent.name()}; "
                f"color: {self._accent.name()}; "
                "border-radius: 13px; padding: 4px 14px; "
                "font-size: 12px; font-weight: 600; letter-spacing: 0.3px;"
            )
        else:
            self._turn_pill.setStyleSheet(
                "background: rgba(255, 255, 255, 0.03); border: 1px solid #1e2b3a; "
                "color: #9aa7b9; border-radius: 13px; padding: 4px 14px; "
                "font-size: 12px; font-weight: 500; letter-spacing: 0.3px;"
            )

    def set_mic_active(self, active: bool) -> None:
        """Highlight the mic button while recording."""
        colour = self._accent.name() if active else "#9aa7b9"
        border = self._accent.name() if active else "#1e2b3a"
        self._mic_btn.setStyleSheet(
            "QToolButton { background: transparent; border: 1px solid "
            f"{border}; border-radius: 6px; color: {colour}; "
            "} QToolButton:hover { background: rgba(255, 255, 255, 0.05); }"
        )
        self._mic_btn.setIcon(qta.icon("fa5s.microphone", color=colour))

    def _open_menu(self) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background: #10161f; color: #dbe4ef; border: 1px solid #1e2b3a; } "
            "QMenu::item:selected { background: rgba(52, 211, 153, 0.15); }"
        )
        new_action = menu.addAction(qta.icon("fa5s.redo-alt", color="#9aa7b9"), "New game")
        settings_action = menu.addAction(qta.icon("fa5s.cog", color="#9aa7b9"), "Settings…")
        menu.addSeparator()
        about_action = menu.addAction(qta.icon("fa5s.info-circle", color="#9aa7b9"), "About RookApp")
        btn_rect = self._menu_btn.rect()
        pos = self._menu_btn.mapToGlobal(btn_rect.bottomRight())
        chosen = menu.exec(pos)
        if chosen is new_action:
            self.menu_triggered.emit("new_game")
        elif chosen is settings_action:
            self.menu_triggered.emit("settings")
        elif chosen is about_action:
            self.menu_triggered.emit("about")


class _InputBar(QWidget):
    submitted = Signal(str)

    def __init__(self, accent: QColor, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._field = QLineEdit()
        self._field.setPlaceholderText("type a move or question — e.g. e4, Nf3, castle kingside")
        self._field.setStyleSheet(
            "QLineEdit { background: #0d1520; border: 1px solid #1e2b3a; "
            "border-radius: 6px; color: #dbe4ef; padding: 8px 12px; "
            "font-family: monospace; font-size: 13px; } "
            f"QLineEdit:focus {{ border-color: {accent.name()}; }}"
        )
        self._field.returnPressed.connect(self._emit)
        row.addWidget(self._field, stretch=1)

        self._send = QPushButton()
        self._send.setIcon(qta.icon("fa5s.paper-plane", color="#0a0e14"))
        self._send.setFixedSize(40, 36)
        self._send.setStyleSheet(
            f"QPushButton {{ background: {accent.name()}; border: none; border-radius: 6px; }} "
            "QPushButton:disabled { background: #1e2b3a; }"
        )
        self._send.clicked.connect(self._emit)
        row.addWidget(self._send)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def set_enabled(self, enabled: bool) -> None:
        self._field.setEnabled(enabled)
        self._send.setEnabled(enabled)

    def _emit(self) -> None:
        text = self._field.text().strip()
        if not text:
            return
        self.submitted.emit(text)
        self._field.clear()


class _IconButton(QToolButton):
    def __init__(self, icon_name: str, tip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setIcon(qta.icon(icon_name, color="#9aa7b9"))
        self.setToolTip(tip)
        self.setFixedSize(30, 30)
        self.setAutoRaise(True)
        self.setStyleSheet(
            "QToolButton { background: transparent; border: 1px solid #1e2b3a; border-radius: 6px; } "
            "QToolButton:hover { background: rgba(255, 255, 255, 0.05); border-color: #2b3c58; }"
        )


# Silence unused-import lints.
_ = QEvent


__all__ = ["RookWindow"]
