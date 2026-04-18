"""Voice pipeline — mic → VAD → Whisper → user text.

Wraps EdgeVox's :class:`AudioRecorder` (mic + silero VAD + AEC) and
:class:`WhisperSTT` in a Qt-friendly surface. The recorder runs on its
own daemon threads internally; transcription happens on a
:class:`QThreadPool` worker so the mic loop never blocks on Whisper.

Emits:
    - ``transcript`` — finalised user utterance, ready to feed into
      :meth:`RookBridge.submit_text`.
    - ``level`` — 0..1 RMS so the UI can draw a mic indicator.
    - ``error`` — human-readable problem (permission denied, no
      input device, Whisper load failed).
    - ``loading`` — emitted once at start while STT models initialise,
      cleared afterwards.

The same mic-capture + VAD code powers the main TUI voice loop, so
we inherit its battle-tested behaviour: speech boundary detection,
echo suppression when TTS plays back, interrupt handling.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

if TYPE_CHECKING:
    from edgevox.audio._original import AudioRecorder
    from edgevox.stt.whisper import WhisperSTT

log = logging.getLogger(__name__)


class VoiceWorker(QObject):
    """Qt-friendly mic pipeline.

    Call :meth:`start` once models have finished loading (expensive,
    ~1-3 s on first use). Call :meth:`set_listening` to toggle the
    mic on/off — stops accepting speech while Rook is thinking /
    speaking to avoid the TTS echoing into the mic.
    """

    transcript = Signal(str)
    level = Signal(float)
    error = Signal(str)
    loading = Signal(bool)
    ready = Signal()

    def __init__(self, *, language: str = "en", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._language = language
        self._recorder: AudioRecorder | None = None
        self._stt: WhisperSTT | None = None
        self._listening = False  # True while Rook is idle / expecting user input
        self._pool = QThreadPool.globalInstance()
        self._shutdown = threading.Event()
        self._load_lock = threading.Lock()

    # ----- lifecycle -----

    def start(self) -> None:
        """Kick off mic + STT load on a worker. Safe to call multiple
        times; the first call wins and later calls no-op."""
        with self._load_lock:
            if self._recorder is not None or self._shutdown.is_set():
                return
        self.loading.emit(True)
        self._pool.start(_StartupJob(self))

    def stop(self) -> None:
        """Stop mic capture and release resources. Call on app exit."""
        self._shutdown.set()
        self._listening = False
        if self._recorder is not None:
            try:
                self._recorder.stop()
            except Exception:
                log.exception("recorder.stop() failed")
        self._recorder = None

    def set_listening(self, on: bool) -> None:
        """Gate whether mic-detected speech reaches Whisper. We keep
        the recorder running (for level meters) but drop transcription
        input when off — cheaper than stop/start cycles that re-open
        the audio device."""
        self._listening = on

    def is_listening(self) -> bool:
        return self._listening

    # ----- worker-thread body -----

    def _boot(self) -> None:
        """Build the recorder + STT on a background thread. Runs once."""
        try:
            # Deferred imports so ``VoiceWorker()`` in the main thread
            # stays cheap; we only pay STT init when the user actually
            # wants voice.
            from edgevox.audio._original import AudioRecorder
            from edgevox.stt.whisper import WhisperSTT

            log.info("Loading Whisper STT for voice input...")
            stt = WhisperSTT()
            if self._shutdown.is_set():
                return
            self._stt = stt

            recorder = AudioRecorder(
                on_speech=self._on_speech_segment,
                on_level=self._on_level,
            )
            recorder.start()
            if self._shutdown.is_set():
                recorder.stop()
                return
            self._recorder = recorder
            self.loading.emit(False)
            self.ready.emit()
            log.info("Voice pipeline ready.")
        except Exception as e:
            log.exception("Voice startup failed")
            self.loading.emit(False)
            # Common-case guidance — mic permission is by far the most
            # frequent cause.
            msg = str(e)
            if "Invalid input device" in msg or "no default" in msg.lower():
                self.error.emit("No microphone detected. Connect one and restart the app.")
            elif "denied" in msg.lower() or "permission" in msg.lower():
                self.error.emit("Microphone permission denied — enable it in your OS settings.")
            else:
                self.error.emit(f"Voice setup failed: {msg}")

    # ----- recorder callbacks (audio thread) -----

    def _on_speech_segment(self, audio: np.ndarray) -> None:
        """A VAD-bounded speech chunk is ready. Dispatch to Whisper on
        a pool worker so the mic loop keeps draining."""
        if not self._listening or self._stt is None:
            return
        self._pool.start(_TranscribeJob(self, audio.copy()))

    def _on_level(self, level: float) -> None:
        """Mic RMS for the indicator."""
        # AudioRecorder reports 0..~1 already; clamp for safety.
        self.level.emit(max(0.0, min(1.0, level)))

    def _transcribe(self, audio: np.ndarray) -> None:
        if self._stt is None or self._shutdown.is_set():
            return
        try:
            text = self._stt.transcribe(audio, language=self._language).strip()
        except Exception as e:
            log.exception("Whisper transcribe failed")
            self.error.emit(f"STT error: {e}")
            return
        if text:
            self.transcript.emit(text)


class _StartupJob(QRunnable):
    def __init__(self, worker: VoiceWorker) -> None:
        super().__init__()
        self._worker = worker

    @Slot()
    def run(self) -> None:  # pragma: no cover — event-loop-driven
        self._worker._boot()


class _TranscribeJob(QRunnable):
    def __init__(self, worker: VoiceWorker, audio: np.ndarray) -> None:
        super().__init__()
        self._worker = worker
        self._audio = audio

    @Slot()
    def run(self) -> None:  # pragma: no cover — event-loop-driven
        self._worker._transcribe(self._audio)


__all__ = ["VoiceWorker"]
