"""Rook server factory — wires chess tools, hooks, and RobotFaceHook together.

Doesn't launch uvicorn; we just assert the returned ``(agent, env)``
has the right shape so ``edgevox-serve --agent ...:build_rook_server_agent``
works out of the box.
"""

from __future__ import annotations

from unittest import mock

import pytest

from edgevox.examples.agents.chess_robot.app import (
    _build_argparser,
    _env_from_args,
    build_rook_server_agent,
)
from edgevox.examples.agents.chess_robot.face_hook import RobotFaceHook
from edgevox.integrations.chess import EngineUnavailable
from edgevox.integrations.chess.hooks import MoveCommentaryHook


@pytest.fixture
def stockfish_env(monkeypatch):
    """Default env for a stockfish-backed build. Forces engine=stockfish
    so the default 'casual' persona's ``maia`` preference doesn't
    demand weights in the test environment."""
    for k in (
        "EDGEVOX_CHESS_PERSONA",
        "EDGEVOX_CHESS_USER_PLAYS",
        "EDGEVOX_CHESS_STOCKFISH_SKILL",
        "EDGEVOX_CHESS_MAIA_WEIGHTS",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("EDGEVOX_CHESS_ENGINE", "stockfish")


class TestBuildRookServerAgent:
    def _patch_build_engine(self, monkeypatch):
        """Replace build_engine with FakeEngine so we don't require stockfish at test time."""
        from tests.chess.conftest import FakeEngine

        fake = FakeEngine()
        monkeypatch.setattr(
            "edgevox.examples.agents.chess_robot.app.build_engine",
            lambda _kind, **_kwargs: fake,
        )
        return fake

    def test_default_build(self, monkeypatch, stockfish_env):
        self._patch_build_engine(monkeypatch)
        agent, env = build_rook_server_agent(None)
        try:
            assert "Rook" in agent.name
            assert agent.tools is not None
            tool_names = {t.name for t in agent.tools}
            assert "play_user_move" in tool_names
            assert "engine_move" in tool_names

            hook_types = {type(h).__name__ for h in agent.hooks}
            assert {"RichChessAnalyticsHook", "MoveCommentaryHook", "RobotFaceHook"} <= hook_types

            assert env.user_plays == "white"
            assert env.engine_plays == "black"
        finally:
            env.close()

    def test_persona_env_picks_correct_face(self, monkeypatch, stockfish_env):
        monkeypatch.setenv("EDGEVOX_CHESS_PERSONA", "trash_talker")
        self._patch_build_engine(monkeypatch)
        agent, env = build_rook_server_agent(None)
        try:
            face = next(h for h in agent.hooks if isinstance(h, RobotFaceHook))
            assert face.persona == "trash_talker"
        finally:
            env.close()

    def test_explicit_maia_without_weights_raises(self, monkeypatch, stockfish_env):
        """If the user *explicitly* asked for maia via env, missing weights
        must be surfaced loudly — silent fallback would hide misconfigs."""
        monkeypatch.setenv("EDGEVOX_CHESS_ENGINE", "maia")
        with pytest.raises(EngineUnavailable, match="MAIA_WEIGHTS"):
            build_rook_server_agent(None)

    def test_persona_defaulting_to_maia_falls_back_to_stockfish(self, monkeypatch):
        """A persona (e.g. ``casual``) whose default engine is maia
        should fall back to stockfish when no weights are configured,
        not crash. Plug-and-play: the app should boot with whatever
        engine is available."""
        # No EDGEVOX_CHESS_ENGINE set — lets the persona default through.
        for k in (
            "EDGEVOX_CHESS_PERSONA",
            "EDGEVOX_CHESS_ENGINE",
            "EDGEVOX_CHESS_USER_PLAYS",
            "EDGEVOX_CHESS_STOCKFISH_SKILL",
            "EDGEVOX_CHESS_MAIA_WEIGHTS",
        ):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("EDGEVOX_CHESS_PERSONA", "casual")  # maia-default persona
        captured = {}

        def _fake_build_engine(kind, **kwargs):
            captured["kind"] = kind
            captured["kwargs"] = kwargs
            from tests.chess.conftest import FakeEngine

            return FakeEngine()

        monkeypatch.setattr("edgevox.examples.agents.chess_robot.app.build_engine", _fake_build_engine)
        _agent, env = build_rook_server_agent(None)
        try:
            assert captured["kind"] == "stockfish"
            assert "skill" in captured["kwargs"]
            # casual persona targets ELO 1400 → skill ~6 (rounded to (1400-800)/100)
            assert 4 <= captured["kwargs"]["skill"] <= 8
        finally:
            env.close()

    def test_hooks_include_full_stack(self, monkeypatch, stockfish_env):
        self._patch_build_engine(monkeypatch)
        agent, env = build_rook_server_agent(None)
        try:
            # Chess-specific hooks: rich analytics (replaces
            # BoardStateInjectionHook + BoardHintHook), move commentary,
            # face.
            from edgevox.examples.agents.chess_robot.rich_board import RichChessAnalyticsHook

            assert any(isinstance(h, RichChessAnalyticsHook) for h in agent.hooks)
            assert any(isinstance(h, MoveCommentaryHook) for h in agent.hooks)
            assert any(isinstance(h, RobotFaceHook) for h in agent.hooks)
            # TTS sanitation: <think> stripper + markdown cleanup.
            from edgevox.examples.agents.chess_robot.sanitize import (
                ThinkTagStripHook,
                VoiceCleanupHook,
            )

            assert any(isinstance(h, ThinkTagStripHook) for h in agent.hooks)
            assert any(isinstance(h, VoiceCleanupHook) for h in agent.hooks)
            # Default SLM hardening bundle is included (tested upstream).
            hook_names = {type(h).__name__ for h in agent.hooks}
            assert "LoopDetectorHook" in hook_names
            assert "EchoedPayloadHook" in hook_names
            assert "SchemaRetryHook" in hook_names
        finally:
            env.close()


class TestEnvFromArgs:
    def test_persona_flag_sets_env(self, monkeypatch):
        monkeypatch.delenv("EDGEVOX_CHESS_PERSONA", raising=False)
        parser = _build_argparser()
        args = parser.parse_args(["--persona", "trash_talker"])
        _env_from_args(args)
        import os

        assert os.environ["EDGEVOX_CHESS_PERSONA"] == "trash_talker"

    def test_user_plays_always_written(self, monkeypatch):
        monkeypatch.delenv("EDGEVOX_CHESS_USER_PLAYS", raising=False)
        parser = _build_argparser()
        args = parser.parse_args([])
        _env_from_args(args)
        import os

        assert os.environ["EDGEVOX_CHESS_USER_PLAYS"] == "white"

    def test_stockfish_skill_serialised_to_string(self, monkeypatch):
        monkeypatch.delenv("EDGEVOX_CHESS_STOCKFISH_SKILL", raising=False)
        parser = _build_argparser()
        args = parser.parse_args(["--stockfish-skill", "12"])
        _env_from_args(args)
        import os

        assert os.environ["EDGEVOX_CHESS_STOCKFISH_SKILL"] == "12"

    def test_maia_weights_passthrough(self, monkeypatch):
        monkeypatch.delenv("EDGEVOX_CHESS_MAIA_WEIGHTS", raising=False)
        parser = _build_argparser()
        args = parser.parse_args(["--maia-weights", "/tmp/maia-1500.pb.gz"])
        _env_from_args(args)
        import os

        assert os.environ["EDGEVOX_CHESS_MAIA_WEIGHTS"] == "/tmp/maia-1500.pb.gz"


class TestArgparseDefaults:
    def test_minimal_invocation_parses(self):
        parser = _build_argparser()
        args = parser.parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 8765
        assert args.user_plays == "white"
        assert args.persona is None  # unset → falls back to env / casual
        assert args.no_browser is False

    def test_invalid_persona_rejected(self):
        parser = _build_argparser()
        with pytest.raises(SystemExit), mock.patch("sys.stderr"):
            parser.parse_args(["--persona", "villain"])

    def test_invalid_engine_rejected(self):
        parser = _build_argparser()
        with pytest.raises(SystemExit), mock.patch("sys.stderr"):
            parser.parse_args(["--engine", "alphazero"])
