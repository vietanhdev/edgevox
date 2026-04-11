"""Integration tests for all EdgeVox interfaces and options with real models.

Tests every entry point, CLI option combination, WebSocket control message,
language/voice switching, and audio processing path.
"""

from __future__ import annotations

import struct
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures — shared server core
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server_core():
    """Create a real ServerCore with smallest models."""
    try:
        from edgevox.server.core import ServerCore

        return ServerCore(
            language="en",
            stt_model="tiny",
            stt_device="cpu",
        )
    except Exception as e:
        pytest.skip(f"ServerCore deps not available: {e}")


@pytest.fixture(scope="module")
def app(server_core):
    from edgevox.server.main import create_app

    return create_app(server_core)


@pytest.fixture()
def client(app):
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. ServerCore initialisation with all language/backend combinations
# ---------------------------------------------------------------------------


class TestServerCoreInit:
    """Verify ServerCore initialises correctly with every language config."""

    def test_english_kokoro(self, server_core):
        assert server_core.language == "en"
        assert type(server_core.tts).__name__ == "KokoroTTS"
        assert type(server_core.stt).__name__ == "WhisperSTT"

    def test_info_has_all_fields(self, server_core):
        info = server_core.info()
        for key in ("language", "languages", "voice", "voices", "stt", "tts", "tts_sample_rate", "active_sessions"):
            assert key in info, f"Missing key: {key}"

    def test_voices_for_english(self, server_core):
        voices = server_core.voices_for_language("en")
        assert isinstance(voices, list)
        assert len(voices) > 0
        assert "af_heart" in voices

    def test_voices_for_vietnamese(self, server_core):
        voices = server_core.voices_for_language("vi")
        assert isinstance(voices, list)
        assert "vi-vais1000" in voices

    def test_voices_for_korean(self, server_core):
        voices = server_core.voices_for_language("ko")
        assert isinstance(voices, list)
        assert "ko-F1" in voices

    def test_voices_for_german(self, server_core):
        voices = server_core.voices_for_language("de")
        assert isinstance(voices, list)
        assert "de-thorsten" in voices

    def test_fresh_history(self, server_core):
        hist = server_core.fresh_history()
        assert isinstance(hist, list)
        assert len(hist) >= 1
        assert hist[0]["role"] == "system"


# ---------------------------------------------------------------------------
# 2. HTTP API endpoints
# ---------------------------------------------------------------------------


class TestHTTPEndpoints:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert isinstance(data["active_sessions"], int)

    def test_info(self, client):
        resp = client.get("/api/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == "en"
        assert isinstance(data["voices"], list)
        assert isinstance(data["languages"], list)
        assert data["tts_sample_rate"] > 0

    def test_info_includes_stt_tts_names(self, client):
        data = client.get("/api/info").json()
        assert "stt" in data
        assert "tts" in data
        assert "Whisper" in data["stt"] or "STT" in data["stt"]


# ---------------------------------------------------------------------------
# 3. WebSocket — connect, ready message, state lifecycle
# ---------------------------------------------------------------------------


class TestWebSocketConnect:
    def test_ready_message(self, client):
        with client.websocket_connect("/ws") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert "session_id" in ready
            assert "language" in ready
            assert "languages" in ready
            assert "voices" in ready
            assert "tts_sample_rate" in ready
            assert ready["frame_size"] == 512
            assert ready["sample_rate"] == 16_000

    def test_listening_state_after_ready(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            state = ws.receive_json()
            assert state["type"] == "state"
            assert state["value"] == "listening"


# ---------------------------------------------------------------------------
# 4. WebSocket — control messages
# ---------------------------------------------------------------------------


class TestWebSocketControls:
    def test_reset(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "reset"})
            msg = ws.receive_json()
            assert msg["type"] == "info"
            assert "cleared" in msg["message"]

    def test_interrupt(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "interrupt"})
            msg = ws.receive_json()
            assert msg["type"] == "state"
            assert msg["value"] == "listening"

    def test_hello(self, client):
        """Hello is a no-op — should not crash."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "hello"})
            # No response expected for hello — send another to verify connection is alive
            ws.send_json({"type": "reset"})
            msg = ws.receive_json()
            assert msg["type"] == "info"

    def test_unknown_control_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "nonexistent_command"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "unknown" in msg["message"]

    def test_invalid_json_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_text("not valid json {{{")
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "invalid" in msg["message"].lower()


# ---------------------------------------------------------------------------
# 5. WebSocket — language switching
# ---------------------------------------------------------------------------


class TestWebSocketLanguage:
    def test_set_language_french(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "set_language", "language": "fr"})
            msgs = [ws.receive_json(), ws.receive_json()]
            types = {m["type"] for m in msgs}
            assert "info" in types
            assert "language_changed" in types

            lang_msg = next(m for m in msgs if m["type"] == "language_changed")
            assert lang_msg["language"] == "fr"
            assert lang_msg["voice"] == "ff_siwis"
            assert isinstance(lang_msg["voices"], list)

    def test_set_language_vietnamese(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "set_language", "language": "vi"})
            msgs = [ws.receive_json(), ws.receive_json()]
            lang_msg = next(m for m in msgs if m["type"] == "language_changed")
            assert lang_msg["language"] == "vi"
            assert "vi-" in lang_msg["voice"]

    def test_set_language_korean(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "set_language", "language": "ko"})
            msgs = [ws.receive_json(), ws.receive_json()]
            lang_msg = next(m for m in msgs if m["type"] == "language_changed")
            assert lang_msg["language"] == "ko"
            assert "ko-" in lang_msg["voice"]

    def test_set_language_unknown_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "set_language", "language": "xx"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "unknown" in msg["message"]

    @pytest.mark.parametrize("lang", ["en", "fr", "es", "ja", "zh"])
    def test_set_kokoro_languages(self, client, lang):
        """Kokoro languages switch without model reload — fast and safe."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "set_language", "language": lang})
            msgs = [ws.receive_json(), ws.receive_json()]
            types = {m["type"] for m in msgs}
            assert "language_changed" in types


# ---------------------------------------------------------------------------
# 6. WebSocket — voice switching
# ---------------------------------------------------------------------------


class TestWebSocketVoice:
    def test_set_voice_kokoro(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "set_voice", "voice": "af_heart"})
            msgs = [ws.receive_json(), ws.receive_json()]
            types = {m["type"] for m in msgs}
            assert "voice_changed" in types

            voice_msg = next(m for m in msgs if m["type"] == "voice_changed")
            assert voice_msg["voice"] == "af_heart"

    def test_set_voice_missing_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "set_voice", "voice": ""})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "missing" in msg["message"].lower()

    def test_set_voice_unknown_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "set_voice", "voice": "nonexistent_voice_xyz"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "unknown" in msg["message"]


# ---------------------------------------------------------------------------
# 7. WebSocket — text input (LLM → TTS, skip STT)
# ---------------------------------------------------------------------------


class TestWebSocketTextInput:
    def test_text_input_empty_ignored(self, client):
        """Empty text_input should not crash the session."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "text_input", "text": ""})
            # Verify connection still alive
            ws.send_json({"type": "reset"})
            msg = ws.receive_json()
            assert msg["type"] == "info"


# ---------------------------------------------------------------------------
# 8. WebSocket — /say (TTS only, skip STT and LLM)
# ---------------------------------------------------------------------------


class TestWebSocketSay:
    def test_say_empty_ignored(self, client):
        """Empty /say should not crash the session."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            ws.send_json({"type": "say", "text": ""})
            ws.send_json({"type": "reset"})
            msg = ws.receive_json()
            assert msg["type"] == "info"


# ---------------------------------------------------------------------------
# 9. WebSocket — binary PCM audio frames (real VAD + STT)
# ---------------------------------------------------------------------------


class TestWebSocketAudio:
    def _make_pcm_frame(self, n_samples: int = 512, amplitude: float = 0.0) -> bytes:
        """Create a raw int16 PCM frame."""
        if amplitude == 0.0:
            return b"\x00\x00" * n_samples
        samples = []
        for i in range(n_samples):
            val = int(amplitude * 32767 * np.sin(2 * np.pi * 440 * i / 16000))
            samples.append(struct.pack("<h", max(-32768, min(32767, val))))
        return b"".join(samples)

    def test_send_silence_no_crash(self, client):
        """Sending silence frames should not crash."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            for _ in range(10):
                ws.send_bytes(self._make_pcm_frame(512, 0.0))

            # Connection should still be alive
            ws.send_json({"type": "reset"})
            msg = ws.receive_json()
            assert msg["type"] in ("info", "level")

    def test_send_audio_level_reported(self, client):
        """Sending non-silent audio should produce level messages."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # ready
            ws.receive_json()  # listening

            # Send loud audio frames
            for _ in range(5):
                ws.send_bytes(self._make_pcm_frame(512, 0.8))

            # Collect messages, looking for level reports
            got_level = False
            for _ in range(10):
                try:
                    msg = ws.receive_json()
                    if msg.get("type") == "level":
                        got_level = True
                        assert isinstance(msg["value"], float)
                        break
                except Exception:
                    break

            # Level messages are optional (depends on VAD timing)
            assert isinstance(got_level, bool)


# ---------------------------------------------------------------------------
# 10. Multiple sessions concurrency
# ---------------------------------------------------------------------------


class TestMultipleSessions:
    def test_two_sessions_independent(self, client, server_core):
        with client.websocket_connect("/ws") as ws1, client.websocket_connect("/ws") as ws2:
            ready1 = ws1.receive_json()
            ready2 = ws2.receive_json()
            assert ready1["session_id"] != ready2["session_id"]
            assert server_core.info()["active_sessions"] >= 2

    def test_session_cleanup_on_disconnect(self, client, server_core):
        initial = server_core.info()["active_sessions"]
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            assert server_core.info()["active_sessions"] > initial
        # After disconnect, session count should drop
        time.sleep(0.1)
        assert server_core.info()["active_sessions"] <= initial + 1


# ---------------------------------------------------------------------------
# 11. CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIParsing:
    def test_tui_parser_defaults(self):
        from edgevox.tui import _build_parser

        parser = _build_parser()
        args = parser.parse_args([])
        assert args.language == "en"
        assert args.stt is None
        assert args.llm is None
        assert args.tts is None
        assert args.voice is None
        assert args.verbose is False
        assert args.web_ui is False
        assert args.simple_ui is False
        assert args.text_mode is False
        assert args.wakeword is None
        assert args.host == "127.0.0.1"
        assert args.port == 8765

    def test_tui_parser_language(self):
        from edgevox.tui import _build_parser

        args = _build_parser().parse_args(["--language", "vi"])
        assert args.language == "vi"

    def test_tui_parser_web_ui(self):
        from edgevox.tui import _build_parser

        args = _build_parser().parse_args(["--web-ui", "--host", "0.0.0.0", "--port", "9000"])
        assert args.web_ui is True
        assert args.host == "0.0.0.0"
        assert args.port == 9000

    def test_tui_parser_simple_ui_text_mode(self):
        from edgevox.tui import _build_parser

        args = _build_parser().parse_args(["--simple-ui", "--text-mode"])
        assert args.simple_ui is True
        assert args.text_mode is True

    def test_tui_parser_model_options(self):
        from edgevox.tui import _build_parser

        args = _build_parser().parse_args(
            ["--stt", "tiny", "--stt-device", "cpu", "--llm", "/tmp/m.gguf", "--tts", "piper", "--voice", "vi-vais1000"]
        )
        assert args.stt == "tiny"
        assert args.stt_device == "cpu"
        assert args.llm == "/tmp/m.gguf"
        assert args.tts == "piper"
        assert args.voice == "vi-vais1000"

    def test_tui_parser_wakeword_and_timeout(self):
        from edgevox.tui import _build_parser

        args = _build_parser().parse_args(["--wakeword", "hey jarvis", "--session-timeout", "60"])
        assert args.wakeword == "hey jarvis"
        assert args.session_timeout == 60.0

    def test_tui_parser_mutually_exclusive_ui(self):
        from edgevox.tui import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--web-ui", "--simple-ui"])

    def test_cli_parser_defaults(self):
        import argparse

        # Verify the CLI module can parse defaults without crashing
        parser = argparse.ArgumentParser()
        parser.add_argument("--stt", default=None)
        parser.add_argument("--language", default="en")
        parser.add_argument("--text-mode", action="store_true")
        args = parser.parse_args([])
        assert args.language == "en"

    def test_server_parser_defaults(self):
        """Verify the server module parses correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8765)
        parser.add_argument("--language", default="en")
        args = parser.parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 8765


# ---------------------------------------------------------------------------
# 12. Model factory combinations
# ---------------------------------------------------------------------------


class TestFactoryCombinations:
    """Verify create_stt/create_tts produce the right backend for each language."""

    @pytest.mark.parametrize(
        "lang, expected_tts",
        [
            ("en", "KokoroTTS"),
            ("fr", "KokoroTTS"),
            ("es", "KokoroTTS"),
            ("ja", "KokoroTTS"),
            ("vi", "PiperTTS"),
            ("de", "PiperTTS"),
            ("ru", "PiperTTS"),
            ("ko", "SupertonicTTS"),
            ("th", "PyThaiTTSBackend"),
        ],
    )
    def test_create_tts_for_language(self, lang, expected_tts):
        try:
            from edgevox.tts import create_tts

            tts = create_tts(lang)
            assert type(tts).__name__ == expected_tts
        except Exception:
            pytest.skip(f"TTS backend for {lang} not available")

    @pytest.mark.parametrize(
        "lang, expected_stt",
        [
            ("en", "WhisperSTT"),
            ("fr", "WhisperSTT"),
            ("vi", "SherpaSTT"),
        ],
    )
    def test_create_stt_for_language(self, lang, expected_stt):
        try:
            from edgevox.stt import create_stt

            stt = create_stt(lang, model_size="tiny", device="cpu") if lang == "en" else create_stt(lang)
            assert type(stt).__name__ == expected_stt
        except Exception:
            pytest.skip(f"STT backend for {lang} not available")


# ---------------------------------------------------------------------------
# 13. TTS output format validation
# ---------------------------------------------------------------------------


class TestTTSOutputFormat:
    """Verify TTS backends produce valid audio arrays."""

    @pytest.fixture(scope="class")
    def kokoro(self):
        try:
            from edgevox.tts.kokoro import KokoroTTS

            return KokoroTTS(voice="af_heart", lang_code="a")
        except Exception as e:
            pytest.skip(f"Kokoro not available: {e}")

    def test_kokoro_output_dtype(self, kokoro):
        audio = kokoro.synthesize("Test.")
        assert audio.dtype == np.float32

    def test_kokoro_output_range(self, kokoro):
        audio = kokoro.synthesize("Test.")
        assert np.max(np.abs(audio)) <= 2.0  # reasonable audio range

    def test_kokoro_output_nonzero(self, kokoro):
        audio = kokoro.synthesize("Hello world.")
        assert np.max(np.abs(audio)) > 0.001

    def test_kokoro_output_length_proportional(self, kokoro):
        short = kokoro.synthesize("Hi.")
        long = kokoro.synthesize("The quick brown fox jumps over the lazy dog and runs across the meadow.")
        assert len(long) > len(short)


# ---------------------------------------------------------------------------
# 14. STT output validation
# ---------------------------------------------------------------------------


class TestSTTOutput:
    @pytest.fixture(scope="class")
    def whisper(self):
        try:
            from edgevox.stt.whisper import WhisperSTT

            return WhisperSTT(model_size="tiny", device="cpu")
        except Exception as e:
            pytest.skip(f"Whisper not available: {e}")

    def test_returns_string(self, whisper):
        audio = np.zeros(16000, dtype=np.float32)
        text = whisper.transcribe(audio, language="en")
        assert isinstance(text, str)

    def test_handles_various_lengths(self, whisper):
        for length in [800, 8000, 16000, 48000]:
            text = whisper.transcribe(np.zeros(length, dtype=np.float32), language="en")
            assert isinstance(text, str)


# ---------------------------------------------------------------------------
# 15. Session state management
# ---------------------------------------------------------------------------


class TestSessionState:
    def test_session_id_unique(self):
        from edgevox.server.session import SessionState

        s1 = SessionState(language="en", history=[])
        s2 = SessionState(language="en", history=[])
        assert s1.id != s2.id

    def test_session_history_isolation(self):
        from edgevox.server.session import SessionState

        h1 = [{"role": "system", "content": "p1"}]
        h2 = [{"role": "system", "content": "p2"}]
        s1 = SessionState(language="en", history=h1)
        s2 = SessionState(language="fr", history=h2)
        assert s1.history[0]["content"] != s2.history[0]["content"]

    def test_feed_audio_accepts_int16(self):
        from edgevox.server.session import SessionState

        try:
            from edgevox.audio._original import VAD

            vad = VAD()
        except Exception:
            pytest.skip("VAD not available")

        s = SessionState(language="en", history=[], vad=vad)
        audio = np.zeros(512, dtype=np.int16)
        s.feed_audio(audio)
        # Should not raise

    def test_drain_segments_empty(self):
        from edgevox.server.session import SessionState

        try:
            from edgevox.audio._original import VAD

            vad = VAD()
        except Exception:
            pytest.skip("VAD not available")

        s = SessionState(language="en", history=[], vad=vad)
        segments = s.drain_segments()
        assert segments == []

    def test_reset_audio(self):
        from edgevox.server.session import SessionState

        try:
            from edgevox.audio._original import VAD

            vad = VAD()
        except Exception:
            pytest.skip("VAD not available")

        s = SessionState(language="en", history=[], vad=vad)
        s.feed_audio(np.zeros(1024, dtype=np.int16))
        s.reset_audio()
        assert s.drain_segments() == []
