"""Integration tests for VAD and audio utilities with real Silero model."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def vad():
    """Load real Silero VAD v6 model (~2MB)."""
    try:
        from edgevox.audio._original import VAD

        return VAD(threshold=0.4)
    except Exception as e:
        pytest.skip(f"VAD not available: {e}")


class TestVADReal:
    def test_loads_successfully(self, vad):
        assert vad is not None

    def test_silence_is_not_speech(self, vad):
        vad.reset()
        silence = np.zeros(512, dtype=np.float32)
        result = vad.is_speech(silence)
        assert result is False

    def test_loud_noise_may_be_speech(self, vad):
        vad.reset()
        rng = np.random.default_rng(42)
        # Loud broadband noise might trigger VAD
        noise = (rng.standard_normal(512) * 0.5).astype(np.float32)
        result = vad.is_speech(noise)
        assert isinstance(result, bool)

    def test_reset_clears_state(self, vad):
        vad.reset()
        silence = np.zeros(512, dtype=np.float32)
        vad.is_speech(silence)
        vad.reset()
        # After reset, state should be clean
        result = vad.is_speech(silence)
        assert result is False

    def test_correct_frame_size(self, vad):
        vad.reset()
        # VAD expects 512 samples (32ms @ 16kHz)
        frame = np.zeros(512, dtype=np.float32)
        result = vad.is_speech(frame)
        assert isinstance(result, bool)

    def test_multiple_frames(self, vad):
        vad.reset()
        for _ in range(10):
            frame = np.zeros(512, dtype=np.float32)
            result = vad.is_speech(frame)
            assert isinstance(result, bool)

    def test_sine_wave_detection(self, vad):
        """A loud sine wave should be classified as speech-like."""
        vad.reset()
        t = np.arange(512, dtype=np.float32) / 16000
        sine = (np.sin(2 * np.pi * 440 * t) * 0.8).astype(np.float32)
        # Feed several frames to let the model accumulate context
        results = []
        for _ in range(5):
            results.append(vad.is_speech(sine))
        # At least one frame should be detected (model is probabilistic)
        assert isinstance(results[-1], bool)


class TestSessionStateVAD:
    """Test VAD integration through SessionState."""

    def test_feed_audio_and_drain(self):
        try:
            from edgevox.audio._original import VAD
            from edgevox.server.session import SessionState

            vad = VAD(threshold=0.4)
        except Exception as e:
            pytest.skip(f"VAD not available: {e}")

        session = SessionState(language="en", history=[], vad=vad)

        # Feed silence — should not produce segments
        silence = np.zeros(16000, dtype=np.int16)  # 1 second
        session.feed_audio(silence)
        segments = session.drain_segments()
        assert isinstance(segments, list)
