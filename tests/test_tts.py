"""Tests for edgevox.tts — TTS backends and factory, with mocked model loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from edgevox.tts import BaseTTS, get_piper_voices


class TestBaseTTS:
    def test_synthesize_raises(self):
        with pytest.raises(NotImplementedError):
            BaseTTS().synthesize("hi")

    def test_synthesize_stream_default(self):
        tts = BaseTTS()
        tts.synthesize = lambda text: np.zeros(100, dtype=np.float32)  # type: ignore[method-assign]
        chunks = list(tts.synthesize_stream("hi"))
        assert len(chunks) == 1
        assert len(chunks[0]) == 100

    def test_sample_rate_default(self):
        assert BaseTTS.sample_rate == 24_000


class TestGetPiperVoices:
    def test_returns_dict(self):
        voices = get_piper_voices()
        assert isinstance(voices, dict)
        assert len(voices) >= 15

    def test_contains_vi_voice(self):
        assert "vi-vais1000" in get_piper_voices()

    def test_contains_de_voice(self):
        assert "de-thorsten" in get_piper_voices()


class TestCreateTTSFactory:
    @patch("kokoro_onnx.Kokoro")
    @patch("edgevox.tts.kokoro.hf_hub_download", return_value="/tmp/fake")
    def test_english_uses_kokoro(self, _hf, mock_kokoro):
        from edgevox.tts import create_tts

        tts = create_tts("en")
        assert type(tts).__name__ == "KokoroTTS"

    @patch("piper.PiperVoice")
    @patch("edgevox.tts.piper.hf_hub_download", return_value="/tmp/fake")
    def test_vi_uses_piper(self, _hf, mock_piper):
        from edgevox.tts import create_tts

        mock_voice = MagicMock()
        mock_voice.config.sample_rate = 22050
        mock_piper.load.return_value = mock_voice

        tts = create_tts("vi")
        assert type(tts).__name__ == "PiperTTS"

    @patch("supertonic.TTS")
    @patch("edgevox.tts.supertonic.hf_hub_download", return_value="/tmp/fake")
    def test_ko_uses_supertonic(self, _hf, mock_super):
        from edgevox.tts import create_tts

        mock_tts = MagicMock()
        mock_tts.sample_rate = 44100
        mock_super.return_value = mock_tts

        tts = create_tts("ko")
        assert type(tts).__name__ == "SupertonicTTS"

    @patch("pythaitts.TTS")
    def test_th_uses_pythaitts(self, mock_pythai):
        from edgevox.tts import create_tts

        mock_tts = MagicMock()
        mock_pythai.return_value = mock_tts

        tts = create_tts("th")
        assert type(tts).__name__ == "PyThaiTTSBackend"


class TestKokoroTTS:
    @patch("kokoro_onnx.Kokoro")
    @patch("edgevox.tts.kokoro.hf_hub_download", return_value="/tmp/fake")
    def test_init(self, _hf, mock_kokoro):
        from edgevox.tts.kokoro import KokoroTTS

        tts = KokoroTTS(voice="af_heart", lang_code="a")
        assert tts._voice == "af_heart"
        assert tts._lang == "en-us"

    @patch("kokoro_onnx.Kokoro")
    @patch("edgevox.tts.kokoro.hf_hub_download", return_value="/tmp/fake")
    def test_synthesize_returns_ndarray(self, _hf, mock_kokoro_cls):
        from edgevox.tts.kokoro import KokoroTTS

        mock_instance = MagicMock()
        mock_instance.create.return_value = (np.zeros(1000, dtype=np.float32), 24000)
        mock_kokoro_cls.return_value = mock_instance

        tts = KokoroTTS()
        audio = tts.synthesize("Hello")
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 1000

    @patch("kokoro_onnx.Kokoro")
    @patch("edgevox.tts.kokoro.hf_hub_download", return_value="/tmp/fake")
    def test_set_language(self, _hf, mock_kokoro):
        from edgevox.tts.kokoro import KokoroTTS

        tts = KokoroTTS()
        tts.set_language("f", "ff_siwis")
        assert tts._lang == "fr-fr"
        assert tts._voice == "ff_siwis"

    @patch("kokoro_onnx.Kokoro")
    @patch("edgevox.tts.kokoro.hf_hub_download", return_value="/tmp/fake")
    def test_sample_rate(self, _hf, mock_kokoro):
        from edgevox.tts.kokoro import KokoroTTS

        tts = KokoroTTS()
        assert tts.sample_rate == 24_000


class TestPiperTTS:
    @patch("piper.PiperVoice")
    @patch("edgevox.tts.piper.hf_hub_download", return_value="/tmp/fake")
    def test_init_valid_voice(self, _hf, mock_piper):
        from edgevox.tts.piper import PiperTTS

        mock_voice = MagicMock()
        mock_voice.config.sample_rate = 22050
        mock_piper.load.return_value = mock_voice

        tts = PiperTTS(voice="vi-vais1000")
        assert tts.sample_rate == 22050

    def test_init_unknown_voice_raises(self):
        from edgevox.tts.piper import PiperTTS

        with pytest.raises(ValueError, match="Unknown Piper voice"):
            PiperTTS(voice="nonexistent")

    @patch("piper.PiperVoice")
    @patch("edgevox.tts.piper.hf_hub_download", return_value="/tmp/fake")
    def test_synthesize(self, _hf, mock_piper):
        from edgevox.tts.piper import PiperTTS

        mock_chunk = MagicMock()
        mock_chunk.audio_float_array = np.ones(500, dtype=np.float32)
        mock_voice = MagicMock()
        mock_voice.config.sample_rate = 22050
        mock_voice.synthesize.return_value = [mock_chunk]
        mock_piper.load.return_value = mock_voice

        tts = PiperTTS(voice="de-thorsten")
        audio = tts.synthesize("Hallo")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 500  # includes padding

    def test_voices_dict_has_expected_keys(self):
        from edgevox.tts.piper import PiperTTS

        assert "vi-vais1000" in PiperTTS._VOICES
        assert "de-thorsten" in PiperTTS._VOICES
        assert "ru-irina" in PiperTTS._VOICES
        assert "ar-kareem" in PiperTTS._VOICES
        assert "id-news" in PiperTTS._VOICES
        assert len(PiperTTS._VOICES) == 20


class TestSupertonicTTS:
    @patch("supertonic.TTS")
    @patch("edgevox.tts.supertonic.hf_hub_download", return_value="/tmp/fake")
    def test_init(self, _hf, mock_super_cls):
        from edgevox.tts.supertonic import SupertonicTTS

        mock_tts = MagicMock()
        mock_tts.sample_rate = 44100
        mock_super_cls.return_value = mock_tts

        tts = SupertonicTTS(voice="ko-F1")
        assert tts.sample_rate == 44100

    def test_invalid_voice_raises(self):
        from edgevox.tts.supertonic import SupertonicTTS

        with pytest.raises(ValueError, match="Unknown Supertonic voice"):
            SupertonicTTS(voice="ko-X9")

    def test_voices_constant(self):
        from edgevox.tts.supertonic import SUPERTONIC_VOICES

        assert len(SUPERTONIC_VOICES) == 10
        for key in SUPERTONIC_VOICES:
            assert key.startswith("ko-")

    @patch("supertonic.TTS")
    @patch("edgevox.tts.supertonic.hf_hub_download", return_value="/tmp/fake")
    def test_synthesize(self, _hf, mock_super_cls):
        from edgevox.tts.supertonic import SupertonicTTS

        mock_tts = MagicMock()
        mock_tts.sample_rate = 44100
        wav = np.zeros((1, 50000), dtype=np.float32)
        duration = np.array([1.0])
        mock_tts.synthesize.return_value = (wav, duration)
        mock_super_cls.return_value = mock_tts

        tts = SupertonicTTS(voice="ko-M2")
        audio = tts.synthesize("안녕하세요")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0


class TestPyThaiTTSBackend:
    @patch("pythaitts.TTS")
    def test_init(self, mock_pythai_cls):
        from edgevox.tts.pythaitts_backend import PyThaiTTSBackend

        mock_tts = MagicMock()
        mock_pythai_cls.return_value = mock_tts

        tts = PyThaiTTSBackend()
        assert tts.sample_rate == 22_050

    @patch("soundfile.read", return_value=(np.ones(5000, dtype=np.float32), 22050))
    @patch("pythaitts.TTS")
    def test_synthesize(self, mock_pythai_cls, mock_sf_read):
        from edgevox.tts.pythaitts_backend import PyThaiTTSBackend

        mock_tts = MagicMock()
        mock_tts.tts.return_value = "/tmp/output.wav"
        mock_pythai_cls.return_value = mock_tts

        tts = PyThaiTTSBackend()
        audio = tts.synthesize("สวัสดี")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 5000  # includes padding
