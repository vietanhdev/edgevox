"""Tests for edgevox.audio — _resample, InterruptiblePlayer, AudioRecorder, and helpers.

All hardware-dependent code (sounddevice, onnxruntime) is mocked.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# _resample
# ---------------------------------------------------------------------------


class TestResample:
    def test_same_rate_returns_same(self):
        from edgevox.audio._original import _resample

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _resample(data, 16000, 16000)
        np.testing.assert_array_equal(result, data)

    def test_downsample(self):
        from edgevox.audio._original import _resample

        data = np.arange(48000, dtype=np.float32)
        result = _resample(data, 48000, 16000)
        assert len(result) == 16000
        assert result.dtype == np.float32

    def test_upsample(self):
        from edgevox.audio._original import _resample

        data = np.arange(16000, dtype=np.float32)
        result = _resample(data, 16000, 48000)
        assert len(result) == 48000

    def test_preserves_endpoints(self):
        from edgevox.audio._original import _resample

        data = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = _resample(data, 16000, 48000)
        assert abs(result[0] - 0.0) < 0.01
        assert abs(result[-1] - 1.0) < 0.01

    def test_single_sample(self):
        from edgevox.audio._original import _resample

        data = np.array([0.5], dtype=np.float32)
        result = _resample(data, 16000, 48000)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# InterruptiblePlayer (with mocked sounddevice)
# ---------------------------------------------------------------------------


class TestInterruptiblePlayer:
    @patch("edgevox.audio._original._sd")
    def test_init_state(self, _mock_sd):
        from edgevox.audio._original import InterruptiblePlayer

        p = InterruptiblePlayer()
        assert p.is_playing is False

    @patch("edgevox.audio._original._sd")
    def test_interrupt_sets_stop(self, _mock_sd):
        from edgevox.audio._original import InterruptiblePlayer

        p = InterruptiblePlayer()
        p.interrupt()
        assert p._stop.is_set()

    @patch("edgevox.audio._original._sd")
    def test_link_recorder(self, _mock_sd):
        from edgevox.audio._original import InterruptiblePlayer

        p = InterruptiblePlayer()
        recorder = MagicMock()
        p.link_recorder(recorder)
        assert p._recorder is recorder
        p.link_recorder(None)
        assert p._recorder is None

    @patch("edgevox.audio._original._sd")
    def test_set_device_closes_stream_on_change(self, _mock_sd):
        from edgevox.audio._original import InterruptiblePlayer

        p = InterruptiblePlayer()
        mock_stream = MagicMock()
        p._stream = mock_stream
        p._device = 0
        p.set_device(1)
        assert p._device == 1
        mock_stream.stop.assert_called_once()

    @patch("edgevox.audio._original._sd")
    def test_set_device_noop_on_same(self, _mock_sd):
        from edgevox.audio._original import InterruptiblePlayer

        p = InterruptiblePlayer()
        p._device = 0
        p.set_device(0)
        # No stream close needed

    @patch("edgevox.audio._original._sd")
    def test_shutdown(self, _mock_sd):
        from edgevox.audio._original import InterruptiblePlayer

        p = InterruptiblePlayer()
        mock_stream = MagicMock()
        p._stream = mock_stream
        p.shutdown()
        assert p._stream is None
        mock_stream.stop.assert_called_once()

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_play_calls_stream_write(self, _mock_sd_fn, mock_sd_module):
        from edgevox.audio._original import InterruptiblePlayer

        mock_stream = MagicMock()
        mock_stream.active = True
        mock_sd_module.OutputStream.return_value = mock_stream
        mock_sd_module.query_devices.return_value = {"max_output_channels": 2}

        p = InterruptiblePlayer()
        p._stream = mock_stream
        p._stream_sr = 24000
        p._stream_device = None
        p._channels = 1
        audio = np.zeros(1200, dtype=np.float32)  # 50ms at 24kHz
        result = p.play(audio, sample_rate=24000)
        assert result is True
        assert mock_stream.write.called

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_play_interrupted_returns_false(self, _mock_sd_fn, mock_sd_module):
        import time as _time

        from edgevox.audio._original import InterruptiblePlayer

        mock_stream = MagicMock()
        mock_stream.active = True
        # Make write() slow so the interrupt thread has time to fire
        mock_stream.write.side_effect = lambda _chunk: _time.sleep(0.01)
        mock_sd_module.query_devices.return_value = {"max_output_channels": 1}

        p = InterruptiblePlayer()
        p._stream = mock_stream
        p._stream_sr = 24000
        p._stream_device = None
        p._channels = 1

        def interrupt_soon():
            _time.sleep(0.03)
            p._stop.set()

        t = threading.Thread(target=interrupt_soon)
        t.start()
        audio = np.zeros(24000 * 10, dtype=np.float32)  # very long audio
        result = p.play(audio, sample_rate=24000)
        t.join()
        assert result is False

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_play_pauses_linked_recorder(self, _mock_sd_fn, mock_sd_module):
        from edgevox.audio._original import InterruptiblePlayer

        mock_stream = MagicMock()
        mock_stream.active = True
        mock_sd_module.query_devices.return_value = {"max_output_channels": 1}

        recorder = MagicMock()
        p = InterruptiblePlayer()
        p._stream = mock_stream
        p._stream_sr = 24000
        p._stream_device = None
        p._channels = 1
        p.link_recorder(recorder)

        audio = np.zeros(100, dtype=np.float32)
        p.play(audio, sample_rate=24000)
        recorder.pause.assert_called_once()
        recorder.resume_after_cooldown.assert_called_once()


# ---------------------------------------------------------------------------
# AudioRecorder (with mocked sounddevice)
# ---------------------------------------------------------------------------


_DEVICE_INFO = {"default_samplerate": 48000, "max_input_channels": 2}


class TestAudioRecorder:
    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_init(self, _mock_sd_fn, mock_sd_module):
        from edgevox.audio._original import AudioRecorder

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        callback = MagicMock()
        rec = AudioRecorder(on_speech=callback)
        assert rec._on_speech is callback

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_pause_sets_suppressed(self, _mock_sd_fn, mock_sd_module):
        from edgevox.audio._original import AudioRecorder

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        rec = AudioRecorder(on_speech=MagicMock())
        rec.pause()
        assert rec._suppressed is True

    @patch("edgevox.audio._original.sd")
    @patch("edgevox.audio._original._sd")
    def test_force_resume(self, _mock_sd_fn, mock_sd_module):
        from edgevox.audio._original import AudioRecorder

        mock_sd_module.query_devices.return_value = _DEVICE_INFO
        rec = AudioRecorder(on_speech=MagicMock())
        rec.pause()
        assert rec._suppressed is True
        rec.force_resume(delay=0.0)
        import time

        time.sleep(0.05)
        assert rec._suppressed is False


# ---------------------------------------------------------------------------
# WakeWordDetector (with mocked pymicro-wakeword)
# ---------------------------------------------------------------------------


class TestWakeWordDetector:
    @patch("pymicro_wakeword.MicroWakeWordFeatures")
    @patch("pymicro_wakeword.MicroWakeWord")
    @patch("pymicro_wakeword.Model")
    def test_init(self, mock_model_enum, mock_mww, mock_features):
        from edgevox.audio.wakeword import WakeWordDetector

        mock_model_enum.HEY_JARVIS = "hey_jarvis_model"
        det = WakeWordDetector(wake_words=["hey jarvis"])
        assert det._detected_name == "hey jarvis"
        mock_mww.from_builtin.assert_called_once_with("hey_jarvis_model")

    @patch("pymicro_wakeword.MicroWakeWordFeatures")
    @patch("pymicro_wakeword.MicroWakeWord")
    @patch("pymicro_wakeword.Model")
    def test_unknown_wakeword_raises(self, _model, _mww, _features):
        from edgevox.audio.wakeword import WakeWordDetector

        with pytest.raises(ValueError, match="Unknown wake word"):
            WakeWordDetector(wake_words=["nonexistent"])

    @patch("pymicro_wakeword.MicroWakeWordFeatures")
    @patch("pymicro_wakeword.MicroWakeWord")
    @patch("pymicro_wakeword.Model")
    def test_detect_returns_none_on_silence(self, mock_model_enum, mock_mww, mock_features):
        from edgevox.audio.wakeword import WakeWordDetector

        mock_model_enum.HEY_JARVIS = "hj"
        mock_instance = MagicMock()
        mock_instance.process_streaming.return_value = False
        mock_mww.from_builtin.return_value = mock_instance

        feat_instance = MagicMock()
        feat_instance.process_streaming.return_value = [MagicMock()]
        mock_features.return_value = feat_instance

        det = WakeWordDetector()
        result = det.detect(np.zeros(512, dtype=np.float32))
        assert result is None

    @patch("pymicro_wakeword.MicroWakeWordFeatures")
    @patch("pymicro_wakeword.MicroWakeWord")
    @patch("pymicro_wakeword.Model")
    def test_detect_returns_name_on_detection(self, mock_model_enum, mock_mww, mock_features):
        from edgevox.audio.wakeword import WakeWordDetector

        mock_model_enum.HEY_JARVIS = "hj"
        mock_instance = MagicMock()
        mock_instance.process_streaming.return_value = True
        mock_mww.from_builtin.return_value = mock_instance

        feat_instance = MagicMock()
        feat_instance.process_streaming.return_value = [MagicMock()]
        mock_features.return_value = feat_instance

        det = WakeWordDetector()
        result = det.detect(np.zeros(512, dtype=np.float32))
        assert result == "hey jarvis"

    @patch("pymicro_wakeword.MicroWakeWordFeatures")
    @patch("pymicro_wakeword.MicroWakeWord")
    @patch("pymicro_wakeword.Model")
    def test_reset(self, mock_model_enum, mock_mww, mock_features):
        from edgevox.audio.wakeword import WakeWordDetector

        mock_model_enum.HEY_JARVIS = "hj"
        det = WakeWordDetector()
        old_features = det._features
        det.reset()
        assert det._features is not old_features


# ---------------------------------------------------------------------------
# StreamingPipeline (with mocked STT/LLM/TTS)
# ---------------------------------------------------------------------------


class TestStreamingPipeline:
    def _make_pipeline(self):
        from edgevox.core.pipeline import StreamingPipeline

        stt = MagicMock()
        stt.transcribe.return_value = "hello"

        llm = MagicMock()
        llm.chat_stream.return_value = iter(["Sure", ",", " I", " can", " help", "."])

        tts = MagicMock()
        tts.synthesize.return_value = np.zeros(1000, dtype=np.float32)

        callbacks = {
            "state_changes": [],
            "user_texts": [],
            "bot_texts": [],
            "metrics_list": [],
        }

        pipeline = StreamingPipeline(
            stt=stt,
            llm=llm,
            tts=tts,
            on_state_change=lambda s: callbacks["state_changes"].append(s),
            on_user_text=lambda t, d: callbacks["user_texts"].append(t),
            on_bot_text=lambda t, d: callbacks["bot_texts"].append(t),
            on_metrics=lambda m: callbacks["metrics_list"].append(m),
        )
        return pipeline, stt, llm, tts, callbacks

    @patch("edgevox.core.pipeline.play_audio")
    def test_process_full_turn(self, mock_play):
        pipeline, stt, llm, tts, cb = self._make_pipeline()
        audio = np.zeros(16000, dtype=np.float32)
        metrics = pipeline.process(audio, language="en")

        stt.transcribe.assert_called_once()
        llm.chat_stream.assert_called_once_with("hello")
        assert tts.synthesize.called
        assert mock_play.called
        assert "stt" in metrics
        assert "llm" in metrics
        assert "tts" in metrics
        assert "total" in metrics
        assert len(cb["state_changes"]) >= 3  # transcribing, thinking/speaking, listening
        assert cb["user_texts"] == ["hello"]

    @patch("edgevox.core.pipeline.play_audio")
    def test_process_empty_transcription_skips(self, mock_play):
        pipeline, stt, llm, tts, _cb = self._make_pipeline()
        stt.transcribe.return_value = "   "
        audio = np.zeros(16000, dtype=np.float32)
        metrics = pipeline.process(audio)

        assert metrics.get("skipped") is True
        llm.chat_stream.assert_not_called()
        tts.synthesize.assert_not_called()

    @patch("edgevox.core.pipeline.play_audio")
    def test_interrupt_stops_pipeline(self, mock_play):
        pipeline, _stt, llm, _tts, _cb = self._make_pipeline()

        def slow_tokens(msg):
            yield "Hello"
            pipeline.interrupt()
            yield " world."

        llm.chat_stream.side_effect = slow_tokens

        audio = np.zeros(16000, dtype=np.float32)
        pipeline.process(audio)
        # Should have been interrupted — play_audio may or may not be called


# ---------------------------------------------------------------------------
# SessionState.segment_duration
# ---------------------------------------------------------------------------


class TestSegmentDuration:
    def test_one_second(self):
        from edgevox.server.session import SessionState

        seg = np.zeros(16000, dtype=np.float32)
        assert abs(SessionState.segment_duration(seg) - 1.0) < 0.001

    def test_half_second(self):
        from edgevox.server.session import SessionState

        seg = np.zeros(8000, dtype=np.float32)
        assert abs(SessionState.segment_duration(seg) - 0.5) < 0.001

    def test_empty(self):
        from edgevox.server.session import SessionState

        seg = np.zeros(0, dtype=np.float32)
        assert SessionState.segment_duration(seg) == 0.0


# ---------------------------------------------------------------------------
# TUI helpers (pure functions, no GUI needed)
# ---------------------------------------------------------------------------


class TestSparkline:
    def test_empty_values(self):
        from edgevox.tui import _sparkline

        result = _sparkline([])
        assert len(result) == 24
        assert result == " " * 24

    def test_all_zeros(self):
        from edgevox.tui import _sparkline

        result = _sparkline([0.0] * 10, width=10)
        assert len(result) == 10
        assert all(c == " " for c in result)

    def test_all_ones(self):
        from edgevox.tui import _sparkline

        result = _sparkline([1.0] * 10, width=10)
        assert len(result) == 10
        assert all(c == "▇" for c in result)

    def test_mixed_values(self):
        from edgevox.tui import _sparkline

        result = _sparkline([0.0, 0.5, 1.0], width=3)
        assert len(result) == 3
        assert result[0] == " "
        assert result[2] == "▇"

    def test_padding_short_list(self):
        from edgevox.tui import _sparkline

        result = _sparkline([1.0], width=5)
        assert len(result) == 5
        assert result[-1] == "▇"
        assert result[0] == " "

    def test_clips_to_width(self):
        from edgevox.tui import _sparkline

        result = _sparkline([0.5] * 50, width=10)
        assert len(result) == 10


class TestDevicePrefs:
    def test_save_and_load(self, tmp_path, monkeypatch):
        from edgevox.tui import _load_device_prefs, _save_device_prefs

        prefs_file = tmp_path / "devices.json"
        monkeypatch.setattr("edgevox.tui._DEVICES_CFG", prefs_file)

        _save_device_prefs(mic=2, spk=4)
        prefs = _load_device_prefs()
        assert prefs["mic"] == 2
        assert prefs["spk"] == 4

    def test_load_missing_file(self, tmp_path, monkeypatch):
        from edgevox.tui import _load_device_prefs

        monkeypatch.setattr("edgevox.tui._DEVICES_CFG", tmp_path / "nonexistent.json")
        prefs = _load_device_prefs()
        assert prefs == {}

    def test_resolve_saved_device_found(self):
        from edgevox.tui import _resolve_saved_device

        available = [("Mic A", 0), ("Mic B", 2), ("Mic C", 5)]
        assert _resolve_saved_device(2, available) == 2

    def test_resolve_saved_device_not_found(self):
        from edgevox.tui import _resolve_saved_device

        available = [("Mic A", 0), ("Mic B", 2)]
        assert _resolve_saved_device(99, available) is None

    def test_resolve_saved_device_none(self):
        from edgevox.tui import _resolve_saved_device

        assert _resolve_saved_device(None, [("Mic A", 0)]) is None
