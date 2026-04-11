"""Integration tests for the full STT → LLM → TTS pipeline with real models.

Downloads all required models and runs end-to-end inference.
This is the most comprehensive test — verifies the entire pipeline works.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def pipeline_components():
    """Load all pipeline components with smallest possible models."""
    components = {}

    # STT: Whisper tiny (smallest)
    try:
        from edgevox.stt.whisper import WhisperSTT

        components["stt"] = WhisperSTT(model_size="tiny", device="cpu")
    except Exception as e:
        pytest.skip(f"Whisper not available: {e}")

    # LLM
    try:
        from edgevox.llm import LLM

        components["llm"] = LLM(language="en")
    except Exception as e:
        pytest.skip(f"LLM not available: {e}")

    # TTS: Kokoro (default English)
    try:
        from edgevox.tts.kokoro import KokoroTTS

        components["tts"] = KokoroTTS(voice="af_heart", lang_code="a")
    except Exception as e:
        pytest.skip(f"Kokoro not available: {e}")

    return components


class TestFullPipelineReal:
    def test_stt_to_llm_to_tts(self, pipeline_components):
        """End-to-end: audio → text → response → speech."""
        stt = pipeline_components["stt"]
        llm = pipeline_components["llm"]
        tts = pipeline_components["tts"]

        # STT: transcribe silence (will return empty/noise text)
        audio_in = np.zeros(16000 * 2, dtype=np.float32)
        text = stt.transcribe(audio_in, language="en")
        assert isinstance(text, str)

        # LLM: generate a response
        llm.reset()
        reply = llm.chat("Say hello.")
        assert isinstance(reply, str)
        assert len(reply) > 0

        # TTS: synthesize the response
        audio_out = tts.synthesize(reply)
        assert isinstance(audio_out, np.ndarray)
        assert len(audio_out) > 0

    def test_streaming_pipeline(self, pipeline_components):
        """Test streaming: LLM tokens → sentence split → TTS per sentence."""
        from edgevox.core.pipeline import stream_sentences

        llm = pipeline_components["llm"]
        tts = pipeline_components["tts"]

        llm.reset()
        token_stream = llm.chat_stream("Tell me a very short joke.")
        sentences = list(stream_sentences(token_stream))
        assert len(sentences) >= 1

        # Synthesize each sentence
        for sentence in sentences:
            audio = tts.synthesize(sentence)
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0


class TestMultiLanguagePipeline:
    """Test the pipeline with different language configurations."""

    @pytest.fixture(scope="class")
    def whisper_stt(self):
        try:
            from edgevox.stt.whisper import WhisperSTT

            return WhisperSTT(model_size="tiny", device="cpu")
        except Exception as e:
            pytest.skip(f"Whisper not available: {e}")

    def test_french_pipeline(self, whisper_stt):
        """French: Whisper STT + Kokoro TTS."""
        try:
            from edgevox.tts.kokoro import KokoroTTS

            tts = KokoroTTS(voice="ff_siwis", lang_code="f")
        except Exception:
            pytest.skip("Kokoro not available")

        audio_in = np.zeros(16000, dtype=np.float32)
        text = whisper_stt.transcribe(audio_in, language="fr")
        assert isinstance(text, str)

        audio_out = tts.synthesize("Bonjour le monde.")
        assert isinstance(audio_out, np.ndarray)
        assert len(audio_out) > 0

    def test_vietnamese_pipeline(self):
        """Vietnamese: Sherpa STT + Piper TTS."""
        try:
            from edgevox.stt.sherpa_stt import SherpaSTT

            stt = SherpaSTT(device="cpu")
        except Exception:
            pytest.skip("Sherpa not available")

        try:
            from edgevox.tts.piper import PiperTTS

            tts = PiperTTS(voice="vi-vais1000")
        except Exception:
            pytest.skip("Piper not available")

        audio_in = np.zeros(16000, dtype=np.float32)
        text = stt.transcribe(audio_in, language="vi")
        assert isinstance(text, str)

        audio_out = tts.synthesize("Xin chào.")
        assert isinstance(audio_out, np.ndarray)
        assert len(audio_out) > 0

    def test_korean_pipeline(self, whisper_stt):
        """Korean: Whisper STT + Supertonic TTS."""
        try:
            from edgevox.tts.supertonic import SupertonicTTS

            tts = SupertonicTTS(voice="ko-F1", lang="ko")
        except Exception:
            pytest.skip("Supertonic not available")

        audio_in = np.zeros(16000, dtype=np.float32)
        text = whisper_stt.transcribe(audio_in, language="ko")
        assert isinstance(text, str)

        audio_out = tts.synthesize("안녕하세요.")
        assert isinstance(audio_out, np.ndarray)
        assert len(audio_out) > 0

    def test_german_pipeline(self, whisper_stt):
        """German: Whisper STT + Piper TTS."""
        try:
            from edgevox.tts.piper import PiperTTS

            tts = PiperTTS(voice="de-thorsten")
        except Exception:
            pytest.skip("Piper not available")

        audio_in = np.zeros(16000, dtype=np.float32)
        text = whisper_stt.transcribe(audio_in, language="de")
        assert isinstance(text, str)

        audio_out = tts.synthesize("Hallo Welt.")
        assert isinstance(audio_out, np.ndarray)
        assert len(audio_out) > 0

    def test_thai_pipeline(self, whisper_stt):
        """Thai: Whisper STT + PyThaiTTS."""
        try:
            from edgevox.tts.pythaitts_backend import PyThaiTTSBackend

            tts = PyThaiTTSBackend()
        except Exception:
            pytest.skip("PyThaiTTS not available")

        audio_in = np.zeros(16000, dtype=np.float32)
        text = whisper_stt.transcribe(audio_in, language="th")
        assert isinstance(text, str)

        audio_out = tts.synthesize("สวัสดีครับ")
        assert isinstance(audio_out, np.ndarray)
        assert len(audio_out) > 0
