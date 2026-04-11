"""Tests for edgevox.server.audio_utils — standalone audio conversion helpers."""

from __future__ import annotations

import io
import struct
import wave

import numpy as np

from edgevox.server.audio_utils import float32_to_wav_bytes, int16_bytes_to_float32


class TestInt16BytesToFloat32:
    def test_round_trip(self):
        original = np.array([0, 1000, -1000, 32767, -32768], dtype=np.int16)
        raw = original.tobytes()
        result = int16_bytes_to_float32(raw)
        assert result.dtype == np.float32
        assert len(result) == len(original)
        # Values should be normalized to [-1.0, 1.0]
        assert abs(result[0]) < 0.001
        assert abs(result[3] - 1.0) < 0.001

    def test_empty_input(self):
        result = int16_bytes_to_float32(b"")
        assert len(result) == 0

    def test_single_sample(self):
        sample = struct.pack("<h", 16384)  # half max
        result = int16_bytes_to_float32(sample)
        assert len(result) == 1
        assert abs(result[0] - 0.5) < 0.01


class TestFloat32ToWavBytes:
    def test_valid_wav(self):
        audio = np.random.randn(1000).astype(np.float32)
        wav_bytes = float32_to_wav_bytes(audio, sample_rate=16000)
        # Should be valid WAV
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 1000

    def test_different_sample_rate(self):
        audio = np.zeros(500, dtype=np.float32)
        wav_bytes = float32_to_wav_bytes(audio, sample_rate=24000)
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getframerate() == 24000

    def test_empty_audio(self):
        audio = np.array([], dtype=np.float32)
        wav_bytes = float32_to_wav_bytes(audio, sample_rate=16000)
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 0
