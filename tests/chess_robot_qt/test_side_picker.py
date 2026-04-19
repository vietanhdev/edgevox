"""Tests for the New-Game side-picker flow: dialog, settings persistence,
and the bridge's user_plays-aware reset_game."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("qtawesome")

from PySide6.QtCore import QSettings

from edgevox.apps.chess_robot_qt.settings import Settings, _side_slug
from edgevox.apps.chess_robot_qt.settings_dialog import SidePickerDialog


@pytest.fixture(autouse=True)
def isolate_qsettings(tmp_path, monkeypatch):
    """Point ``QSettings("EdgeVox", "RookApp")`` at a throwaway ini file
    so tests never read or clobber the developer's real
    ``~/.config/EdgeVox/RookApp.conf``. Runs for every test in this
    module — applies at the QSettings global level (it's a Qt
    singleton-by-key), so the isolation is total."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path),
    )
    yield
    # Reset Qt's path so the next test module starts clean. Using an
    # empty string restores the default lookup chain.
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, "")


class TestSideSlugNormalisation:
    def test_white_and_black_pass_through(self):
        assert _side_slug("white") == "white"
        assert _side_slug("black") == "black"

    def test_case_is_normalised(self):
        assert _side_slug("WHITE") == "white"
        assert _side_slug("Black") == "black"

    def test_random_is_rejected(self):
        # ``random`` must never reach Settings — the picker resolves it
        # to a concrete side before persisting. A stray ``random`` in
        # QSettings (from a partial migration) falls back to white.
        assert _side_slug("random") == "white"

    def test_garbage_falls_back_to_white(self):
        assert _side_slug("") == "white"
        assert _side_slug(None) == "white"
        assert _side_slug("cyan") == "white"


class TestSettingsUserSideRoundTrip:
    """The ``isolate_qsettings`` autouse fixture gives each test its
    own fresh QSettings backing file, so these tests don't need their
    own setup/teardown — the fixture handles isolation."""

    def test_default_is_white(self):
        assert Settings.load().user_side == "white"

    def test_save_and_load_black(self):
        s = Settings.load()
        s.user_side = "black"
        s.save()
        assert Settings.load().user_side == "black"

    def test_load_sanitises_unexpected_value(self):
        # Simulate a corrupted or partially-migrated QSettings blob.
        q = QSettings("EdgeVox", "RookApp")
        q.setValue("user_side", "random")
        # ``random`` must never leak out as a persisted side.
        assert Settings.load().user_side == "white"


class TestSidePickerDialog:
    def test_white_choice_accepts_and_returns_white(self, qtbot):
        dlg = SidePickerDialog(current="white")
        qtbot.addWidget(dlg)
        dlg._select("white")
        assert dlg.choice == "white"

    def test_black_choice_accepts_and_returns_black(self, qtbot):
        dlg = SidePickerDialog(current="white")
        qtbot.addWidget(dlg)
        dlg._select("black")
        assert dlg.choice == "black"

    def test_random_is_resolved_by_pick(self, monkeypatch):
        # ``pick`` opens ``exec`` modally, which stalls headlessly. Stub
        # ``exec`` to simulate the user clicking the Random button and
        # the dialog accepting, then verify ``random.choice`` resolves
        # it to a concrete side.
        def fake_exec(self):
            self._select("random")
            return self.DialogCode.Accepted

        monkeypatch.setattr(SidePickerDialog, "exec", fake_exec)
        monkeypatch.setattr(
            "edgevox.apps.chess_robot_qt.settings_dialog.random.choice",
            lambda _choices: "black",
        )
        assert SidePickerDialog.pick() == "black"

    def test_cancel_returns_none(self, monkeypatch):
        def fake_exec(self):
            return self.DialogCode.Rejected

        monkeypatch.setattr(SidePickerDialog, "exec", fake_exec)
        assert SidePickerDialog.pick() is None


class TestBridgeResetGameUserPlays:
    """``RookBridge.reset_game(user_plays=...)`` must:
    - update ``config.user_plays`` so downstream hooks pick up the new
      side without rebuilding the agent,
    - call ``env.new_game(user_plays=...)`` to reset the board, and
    - when the user is Black, also call ``env.engine_move()`` so the
      welcome turn has a move to narrate.

    We mock the env entirely — the real :class:`ChessEnvironment`
    needs a subprocess engine we don't want in unit tests.
    """

    def _make_bridge_with_mock_env(self):
        from edgevox.apps.chess_robot_qt.bridge import RookBridge, RookConfig

        bridge = RookBridge(RookConfig())
        bridge._env = MagicMock()
        bridge._env.engine_move.return_value = (MagicMock(), MagicMock())
        return bridge

    def test_white_reset_updates_config_and_skips_engine_move(self):
        bridge = self._make_bridge_with_mock_env()
        opened = bridge.reset_game(user_plays="white")
        assert opened is False
        assert bridge.config.user_plays == "white"
        bridge._env.new_game.assert_called_once_with(user_plays="white")
        bridge._env.engine_move.assert_not_called()

    def test_black_reset_triggers_engine_opening(self):
        bridge = self._make_bridge_with_mock_env()
        opened = bridge.reset_game(user_plays="black")
        assert opened is True
        assert bridge.config.user_plays == "black"
        bridge._env.new_game.assert_called_once_with(user_plays="black")
        bridge._env.engine_move.assert_called_once_with()

    def test_engine_failure_on_black_is_non_fatal(self):
        bridge = self._make_bridge_with_mock_env()
        bridge._env.engine_move.side_effect = RuntimeError("engine died")
        opened = bridge.reset_game(user_plays="black")
        # Reset still happened; caller sees engine_opened=False so it
        # picks the welcome-user prompt instead of narrate-my-opening.
        assert opened is False
        assert bridge.config.user_plays == "black"

    def test_no_sidearg_preserves_config_and_skips_engine(self):
        bridge = self._make_bridge_with_mock_env()
        bridge.config.user_plays = "black"
        opened = bridge.reset_game()
        assert opened is False
        # Without user_plays we don't touch config; env.new_game() runs
        # with no kwargs — its internal ``_user_plays`` is preserved.
        assert bridge.config.user_plays == "black"
        bridge._env.new_game.assert_called_once_with()
        bridge._env.engine_move.assert_not_called()

    def test_no_env_returns_false_silently(self):
        from edgevox.apps.chess_robot_qt.bridge import RookBridge, RookConfig

        bridge = RookBridge(RookConfig())
        # Bridge not started yet — _env is None. Must not raise.
        assert bridge.reset_game(user_plays="black") is False


class TestEngineJustMovedDetection:
    """The window's engine-reveal delay + engine-move chip both trigger
    when "my turn starts AND a move just landed". Prior versions keyed
    on ``ply % 2 == 0`` which quietly broke when the user plays Black —
    engine moves land on odd plies then. These tests lock in the
    side-agnostic behaviour.
    """

    def _make_state(self, *, turn: str, ply: int, last_san: str | None):
        from edgevox.integrations.chess.environment import ChessState

        return ChessState(
            fen="placeholder",
            ply=ply,
            turn=turn,
            last_move_uci="e2e4" if last_san else None,
            last_move_san=last_san,
            san_history=[last_san] if last_san else [],
        )

    def _build_window(self, monkeypatch, user_plays: str):
        from edgevox.apps.chess_robot_qt.bridge import RookBridge, RookConfig

        monkeypatch.setattr(RookBridge, "start", lambda self: None)
        # Persist the side first — ``RookWindow.__init__`` loads
        # :class:`Settings` and overrides ``bridge.config.user_plays``
        # from ``Settings.user_side``. Without this write, the window
        # would flip the side back to Settings' default (``"white"``)
        # every time.
        q = QSettings("EdgeVox", "RookApp")
        q.setValue("user_side", user_plays)
        q.sync()

        from edgevox.apps.chess_robot_qt.window import RookWindow

        cfg = RookConfig()
        cfg.user_plays = user_plays
        bridge = RookBridge(cfg)
        return RookWindow(bridge)

    def test_user_white_engine_on_even_ply_triggers_reveal(self, qtbot, monkeypatch):
        window = self._build_window(monkeypatch, user_plays="white")
        qtbot.addWidget(window)
        state = self._make_state(turn="white", ply=2, last_san="e5")
        window._on_chess_state(state)
        assert window._pending_engine_state is state

    def test_user_black_engine_opens_on_odd_ply_triggers_reveal(self, qtbot, monkeypatch):
        """Regression: user=black, engine opened with e4 on ply=1. Old
        parity check rejected this (ply%2==1) and the reveal fired
        instantly instead of holding for the animation delay."""
        window = self._build_window(monkeypatch, user_plays="black")
        qtbot.addWidget(window)
        state = self._make_state(turn="black", ply=1, last_san="e4")
        window._on_chess_state(state)
        assert window._pending_engine_state is state, (
            "engine's opening move on Black-side games must trigger the reveal delay"
        )

    def test_user_move_does_not_trigger_engine_reveal(self, qtbot, monkeypatch):
        window = self._build_window(monkeypatch, user_plays="white")
        qtbot.addWidget(window)
        # User=white just played e4 → ply=1, turn=black. Engine's turn
        # now, not user's — must not fire the reveal delay.
        state = self._make_state(turn="black", ply=1, last_san="e4")
        window._on_chess_state(state)
        assert window._pending_engine_state is None

    def test_fresh_board_no_moves_does_not_trigger_reveal(self, qtbot, monkeypatch):
        window = self._build_window(monkeypatch, user_plays="black")
        qtbot.addWidget(window)
        # ply=0, no moves — even though turn=white may match (for white)
        # / not match (for black), ``last_move_san`` is empty so no reveal.
        state = self._make_state(turn="white", ply=0, last_san=None)
        window._on_chess_state(state)
        assert window._pending_engine_state is None


class TestBoardOrientationAfterSideChange:
    """The board view owns its own orientation flag; the window calls
    ``set_orientation`` before ``reset_game`` so the first paint of the
    new match lands flipped. Round-trip the flag."""

    def test_flip_round_trip(self, qtbot):
        from edgevox.apps.chess_robot_qt.board import ChessBoardView

        view = ChessBoardView()
        qtbot.addWidget(view)
        view.set_orientation("white")
        assert view._orientation == "white"
        view.set_orientation("black")
        assert view._orientation == "black"
        # A mid-word prefix ("b" vs "black") must match — settings may
        # round-trip either form.
        view.set_orientation("b")
        assert view._orientation == "black"
        view.set_orientation("w")
        assert view._orientation == "white"

    def test_click_mapping_follows_orientation(self, qtbot):
        """When flipped to black, top-left click should hit h1 (white's
        kingside rook from black's POV), not a8. This is the test that
        would have caught a missed flip in ``_sq_at_point``."""
        from edgevox.apps.chess_robot_qt.board import ChessBoardView

        view = ChessBoardView()
        qtbot.addWidget(view)
        view.resize(400, 400)
        # Force an explicit size so _square_size returns something
        # deterministic. viewport() size depends on layout.
        view.setFixedSize(400, 400)
        view.show()
        qtbot.waitExposed(view)

        view.set_orientation("white")
        sz = view._square_size()
        # Top-left corner in white POV → a8.
        assert view._sq_at_point(sz * 0.5, sz * 0.5) == "a8"

        view.set_orientation("black")
        # Same pixel in black POV → h1 (board is flipped both axes).
        assert view._sq_at_point(sz * 0.5, sz * 0.5) == "h1"
