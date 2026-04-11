"""Shared fixtures and helpers for EdgeVox tests."""

from __future__ import annotations

import numpy as np
import pytest

# --------------- Reusable fakes ---------------


class FakeSTT:
    def __init__(self):
        self.calls: list[int] = []

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        self.calls.append(len(audio))
        return "hello world"


class FakeLLM:
    def __init__(self):
        self._language = "en"
        self._history: list[dict] = [{"role": "system", "content": "You are Vox."}]

    def chat(self, message: str) -> str:
        self._history.append({"role": "user", "content": message})
        reply = f"Reply to: {message}"
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def chat_stream(self, message: str):
        self._history.append({"role": "user", "content": message})
        tokens = ["Sure", ",", " I", " can", " help", "."]
        yield from tokens
        self._history.append({"role": "assistant", "content": "".join(tokens)})

    def set_language(self, language: str):
        self._language = language

    def reset(self):
        self._history = self._history[:1]


class FakeTTS:
    sample_rate = 24_000

    def synthesize(self, text: str) -> np.ndarray:
        return np.zeros(self.sample_rate, dtype=np.float32)

    def synthesize_stream(self, text: str):
        yield self.synthesize(text)


class FakeVAD:
    """Scripted VAD: returns values from a pre-set script list."""

    def __init__(self, script: list[bool] | None = None):
        self.script = script or []
        self._idx = 0
        self.resets = 0

    def is_speech(self, _frame) -> bool:
        if self._idx < len(self.script):
            val = self.script[self._idx]
            self._idx += 1
            return val
        return False

    def reset(self):
        self._idx = 0
        self.resets += 1


@pytest.fixture
def fake_stt():
    return FakeSTT()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def fake_tts():
    return FakeTTS()


@pytest.fixture
def fake_vad():
    return FakeVAD()


# --------------- GPU mock fixtures ---------------


@pytest.fixture
def mock_gpu_none(monkeypatch):
    """Patch GPU detection to report no GPU, 16GB RAM."""
    monkeypatch.setattr("edgevox.core.gpu.get_nvidia_vram_gb", lambda: None)
    monkeypatch.setattr("edgevox.core.gpu.get_nvidia_gpu_name", lambda: None)
    monkeypatch.setattr("edgevox.core.gpu.get_nvidia_used_mb", lambda: None)
    monkeypatch.setattr("edgevox.core.gpu.has_cuda", lambda: False)
    monkeypatch.setattr("edgevox.core.gpu.has_metal", lambda: False)
    monkeypatch.setattr("edgevox.core.gpu.get_ram_gb", lambda: 16.0)


@pytest.fixture
def mock_gpu_nvidia_8gb(monkeypatch):
    """Patch GPU detection to report 8GB NVIDIA GPU."""
    monkeypatch.setattr("edgevox.core.gpu.get_nvidia_vram_gb", lambda: 8.0)
    monkeypatch.setattr("edgevox.core.gpu.get_nvidia_gpu_name", lambda: "RTX 3070")
    monkeypatch.setattr("edgevox.core.gpu.has_cuda", lambda: True)
    monkeypatch.setattr("edgevox.core.gpu.has_metal", lambda: False)
    monkeypatch.setattr("edgevox.core.gpu.get_ram_gb", lambda: 32.0)
