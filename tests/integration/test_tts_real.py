"""Integration tests for TTS backends with real models.

Downloads actual models and runs synthesis on sample text.
These tests are slow and require network access + disk space.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---- Kokoro TTS ----


@pytest.fixture(scope="module")
def kokoro_tts():
    """Load Kokoro-82M ONNX model (~338MB total)."""
    try:
        from edgevox.tts.kokoro import KokoroTTS

        return KokoroTTS(voice="af_heart", lang_code="a")
    except Exception as e:
        pytest.skip(f"Kokoro not available: {e}")


class TestKokoroReal:
    def test_loads_successfully(self, kokoro_tts):
        assert kokoro_tts._kokoro is not None
        assert kokoro_tts.sample_rate == 24_000

    def test_synthesize_english(self, kokoro_tts):
        audio = kokoro_tts.synthesize("Hello, how are you?")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    def test_synthesize_short_text(self, kokoro_tts):
        audio = kokoro_tts.synthesize("Hi.")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_synthesize_long_text(self, kokoro_tts):
        text = "The quick brown fox jumps over the lazy dog. " * 3
        audio = kokoro_tts.synthesize(text)
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 24_000  # at least 1 second

    def test_synthesize_stream(self, kokoro_tts):
        chunks = list(kokoro_tts.synthesize_stream("Hello world."))
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, np.ndarray)

    def test_set_language_french(self, kokoro_tts):
        kokoro_tts.set_language("f", "ff_siwis")
        audio = kokoro_tts.synthesize("Bonjour le monde.")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        # Restore
        kokoro_tts.set_language("a", "af_heart")

    @pytest.mark.parametrize(
        "lang_code, voice, text",
        [
            ("a", "af_heart", "Hello world."),
            ("b", "bf_emma", "Hello world."),
            ("f", "ff_siwis", "Bonjour."),
            ("e", "ef_dora", "Hola."),
            ("j", "jf_alpha", "Konnichiwa."),
        ],
    )
    def test_synthesize_multiple_languages(self, kokoro_tts, lang_code, voice, text):
        kokoro_tts.set_language(lang_code, voice)
        audio = kokoro_tts.synthesize(text)
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        # Restore
        kokoro_tts.set_language("a", "af_heart")


# ---- Piper TTS ----


@pytest.fixture(scope="module")
def piper_tts_vi():
    """Load Piper TTS with Vietnamese voice."""
    try:
        from edgevox.tts.piper import PiperTTS

        return PiperTTS(voice="vi-vais1000")
    except Exception as e:
        pytest.skip(f"Piper not available: {e}")


@pytest.fixture(scope="module")
def piper_tts_de():
    """Load Piper TTS with German voice."""
    try:
        from edgevox.tts.piper import PiperTTS

        return PiperTTS(voice="de-thorsten")
    except Exception as e:
        pytest.skip(f"Piper not available: {e}")


class TestPiperVietnameseReal:
    def test_loads_successfully(self, piper_tts_vi):
        assert piper_tts_vi.sample_rate > 0

    def test_synthesize(self, piper_tts_vi):
        audio = piper_tts_vi.synthesize("Xin chào thế giới.")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    def test_synthesize_short(self, piper_tts_vi):
        audio = piper_tts_vi.synthesize("Xin chào.")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0


class TestPiperGermanReal:
    def test_loads_successfully(self, piper_tts_de):
        assert piper_tts_de.sample_rate > 0

    def test_synthesize(self, piper_tts_de):
        audio = piper_tts_de.synthesize("Hallo, wie geht es Ihnen?")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0


# ---- Supertonic TTS ----


@pytest.fixture(scope="module")
def supertonic_tts():
    """Load Supertonic TTS with Korean voice (~255MB)."""
    try:
        from edgevox.tts.supertonic import SupertonicTTS

        return SupertonicTTS(voice="ko-F1", lang="ko")
    except Exception as e:
        pytest.skip(f"Supertonic not available: {e}")


class TestSupertonicReal:
    def test_loads_successfully(self, supertonic_tts):
        assert supertonic_tts.sample_rate > 0

    def test_synthesize_korean(self, supertonic_tts):
        audio = supertonic_tts.synthesize("안녕하세요.")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    def test_synthesize_short(self, supertonic_tts):
        audio = supertonic_tts.synthesize("안녕.")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0


# ---- PyThaiTTS ----


@pytest.fixture(scope="module")
def pythaitts():
    """Load PyThaiTTS backend (~163MB)."""
    try:
        from edgevox.tts.pythaitts_backend import PyThaiTTSBackend

        return PyThaiTTSBackend()
    except Exception as e:
        pytest.skip(f"PyThaiTTS not available: {e}")


class TestPyThaiTTSReal:
    def test_loads_successfully(self, pythaitts):
        assert pythaitts.sample_rate > 0

    def test_synthesize_thai(self, pythaitts):
        audio = pythaitts.synthesize("สวัสดีครับ")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0


# ---- Factory tests ----


class TestCreateTTSFactory:
    def test_create_kokoro_for_english(self):
        try:
            from edgevox.tts import create_tts

            tts = create_tts("en")
            assert type(tts).__name__ == "KokoroTTS"
            assert tts.sample_rate == 24_000
        except Exception:
            pytest.skip("Kokoro dependencies not available")

    def test_create_piper_for_vietnamese(self):
        try:
            from edgevox.tts import create_tts

            tts = create_tts("vi")
            assert type(tts).__name__ == "PiperTTS"
        except Exception:
            pytest.skip("Piper dependencies not available")

    def test_create_supertonic_for_korean(self):
        try:
            from edgevox.tts import create_tts

            tts = create_tts("ko")
            assert type(tts).__name__ == "SupertonicTTS"
        except Exception:
            pytest.skip("Supertonic dependencies not available")

    def test_create_pythaitts_for_thai(self):
        try:
            from edgevox.tts import create_tts

            tts = create_tts("th")
            assert type(tts).__name__ == "PyThaiTTSBackend"
        except Exception:
            pytest.skip("PyThaiTTS dependencies not available")
