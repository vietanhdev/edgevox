"""Integration tests for STT backends with real models.

Downloads actual models and runs inference on synthetic audio.
These tests are slow (~30s each) and require network access.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def whisper_stt():
    """Load Whisper tiny model (smallest, ~150MB)."""
    from edgevox.stt.whisper import WhisperSTT

    return WhisperSTT(model_size="tiny", device="cpu")


@pytest.fixture(scope="module")
def sherpa_stt():
    """Load Sherpa-ONNX Zipformer Vietnamese model (~30MB)."""
    try:
        from edgevox.stt.sherpa_stt import SherpaSTT

        return SherpaSTT(device="cpu")
    except Exception as e:
        pytest.skip(f"Sherpa-ONNX not available: {e}")


class TestWhisperReal:
    def test_loads_successfully(self, whisper_stt):
        assert whisper_stt._model_size == "tiny"
        assert whisper_stt._device == "cpu"

    def test_transcribe_silence(self, whisper_stt):
        silence = np.zeros(16000, dtype=np.float32)
        text = whisper_stt.transcribe(silence, language="en")
        assert isinstance(text, str)

    def test_transcribe_noise(self, whisper_stt):
        rng = np.random.default_rng(42)
        noise = (rng.standard_normal(16000) * 0.01).astype(np.float32)
        text = whisper_stt.transcribe(noise, language="en")
        assert isinstance(text, str)

    def test_transcribe_short_audio(self, whisper_stt):
        short = np.zeros(1600, dtype=np.float32)  # 0.1s
        text = whisper_stt.transcribe(short, language="en")
        assert isinstance(text, str)

    def test_transcribe_long_audio(self, whisper_stt):
        long = np.zeros(16000 * 10, dtype=np.float32)  # 10s silence
        text = whisper_stt.transcribe(long, language="en")
        assert isinstance(text, str)

    @pytest.mark.parametrize("lang", ["en", "fr", "es", "de", "ja", "zh"])
    def test_transcribe_multiple_languages(self, whisper_stt, lang):
        silence = np.zeros(16000, dtype=np.float32)
        text = whisper_stt.transcribe(silence, language=lang)
        assert isinstance(text, str)


class TestSherpaReal:
    def test_loads_successfully(self, sherpa_stt):
        assert sherpa_stt._recognizer is not None

    def test_transcribe_silence(self, sherpa_stt):
        silence = np.zeros(16000, dtype=np.float32)
        text = sherpa_stt.transcribe(silence, language="vi")
        assert isinstance(text, str)

    def test_transcribe_noise(self, sherpa_stt):
        rng = np.random.default_rng(42)
        noise = (rng.standard_normal(16000) * 0.01).astype(np.float32)
        text = sherpa_stt.transcribe(noise, language="vi")
        assert isinstance(text, str)

    def test_transcribe_short_audio(self, sherpa_stt):
        short = np.zeros(1600, dtype=np.float32)
        text = sherpa_stt.transcribe(short, language="vi")
        assert isinstance(text, str)


class TestCreateSTTFactory:
    def test_create_whisper_for_english(self):
        from edgevox.stt import create_stt

        stt = create_stt("en", model_size="tiny", device="cpu")
        assert type(stt).__name__ == "WhisperSTT"
        assert stt._model_size == "tiny"

    def test_create_sherpa_for_vietnamese(self):
        from edgevox.stt import create_stt

        try:
            stt = create_stt("vi")
            assert type(stt).__name__ in ("SherpaSTT", "WhisperSTT")
        except Exception:
            pytest.skip("Sherpa-ONNX dependencies not available")
