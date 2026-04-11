"""Tests for edgevox.core.config — language configuration, no mocks needed."""

from __future__ import annotations

import pytest

from edgevox.core.config import LANGUAGES, LanguageConfig, get_lang, lang_options, needs_stt_reload


class TestLanguagesDict:
    def test_populated(self):
        assert len(LANGUAGES) >= 15

    def test_keys_are_strings(self):
        for key in LANGUAGES:
            assert isinstance(key, str)

    def test_values_are_language_config(self):
        for cfg in LANGUAGES.values():
            assert isinstance(cfg, LanguageConfig)


class TestGetLang:
    def test_english_default_for_unknown(self):
        cfg = get_lang("xx")
        assert cfg.code == "en"
        assert cfg.name == "English"

    @pytest.mark.parametrize(
        "code, expected_name",
        [
            ("en", "English"),
            ("vi", "Vietnamese"),
            ("fr", "French"),
            ("ko", "Korean"),
            ("th", "Thai"),
            ("de", "German"),
            ("ja", "Japanese"),
            ("zh", "Chinese"),
        ],
    )
    def test_known_codes(self, code, expected_name):
        cfg = get_lang(code)
        assert cfg.name == expected_name
        assert cfg.code == code


class TestLanguageConfig:
    def test_frozen(self):
        cfg = get_lang("en")
        with pytest.raises(AttributeError):
            cfg.code = "xx"  # type: ignore[misc]

    def test_whisper_code_uses_code_by_default(self):
        cfg = get_lang("en")
        assert cfg.whisper_code == "en"

    def test_whisper_code_override(self):
        cfg = get_lang("en-gb")
        assert cfg.whisper_code == "en"


class TestBackendRouting:
    def test_vi_uses_sherpa(self):
        assert get_lang("vi").stt_backend == "sherpa"

    def test_vi_uses_piper(self):
        assert get_lang("vi").tts_backend == "piper"

    def test_ko_uses_supertonic(self):
        assert get_lang("ko").tts_backend == "supertonic"

    def test_th_uses_pythaitts(self):
        assert get_lang("th").tts_backend == "pythaitts"

    def test_en_uses_whisper(self):
        assert get_lang("en").stt_backend == "whisper"

    def test_en_uses_kokoro(self):
        assert get_lang("en").tts_backend == "kokoro"

    def test_de_uses_piper(self):
        assert get_lang("de").tts_backend == "piper"

    def test_de_uses_whisper(self):
        assert get_lang("de").stt_backend == "whisper"


class TestLangOptions:
    def test_returns_tuples(self):
        opts = lang_options()
        assert isinstance(opts, list)
        for name, code in opts:
            assert isinstance(name, str)
            assert isinstance(code, str)

    def test_length_matches_languages(self):
        assert len(lang_options()) == len(LANGUAGES)


class TestNeedsSTTReload:
    def test_same_backend_no_reload(self):
        assert needs_stt_reload("en", "fr") is False

    def test_different_backend_reload(self):
        assert needs_stt_reload("en", "vi") is True

    def test_same_language_no_reload(self):
        assert needs_stt_reload("en", "en") is False

    def test_piper_languages_no_reload(self):
        assert needs_stt_reload("de", "ru") is False


class TestDefaultVoices:
    @pytest.mark.parametrize(
        "code, voice",
        [
            ("en", "af_heart"),
            ("vi", "vi-vais1000"),
            ("ko", "ko-F1"),
            ("th", "th-default"),
            ("de", "de-thorsten"),
        ],
    )
    def test_default_voice(self, code, voice):
        assert get_lang(code).default_voice == voice


class TestTestPhrases:
    def test_all_languages_have_test_phrase(self):
        for cfg in LANGUAGES.values():
            assert cfg.test_phrase, f"{cfg.code} has empty test phrase"
