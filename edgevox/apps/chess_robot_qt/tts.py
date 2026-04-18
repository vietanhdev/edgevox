"""Text-to-speech pipeline — Kokoro → sounddevice playback.

Keeps Kokoro's ~300 MB ONNX load off the main thread: the backend
warms up inside :class:`_WarmupJob` on a :class:`QThreadPool` worker
so the first chat bubble can render while the model downloads /
loads. Subsequent :meth:`speak` calls synthesise on the same pool.

The worker exposes three Qt signals:

- ``started`` — playback is about to begin; UI should mark the face
  tempo ``speaking`` and gate the mic (AEC) to avoid self-loop.
- ``finished`` — playback ended (completed or interrupted); UI can
  resume listening.
- ``error`` — one-shot human-readable failure string. Voice-only
  failures shouldn't crash the app; we surface them in the status bar.

Kokoro is MIT-licensed — safe to ship inside a MIT app. Falling back
to Piper (also MIT) would be a future extension.
"""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

log = logging.getLogger(__name__)


class TTSWorker(QObject):
    """Qt-friendly Kokoro wrapper."""

    started = Signal()
    finished = Signal()
    error = Signal(str)
    ready = Signal()

    def __init__(self, *, voice: str = "af_heart", lang_code: str = "a", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._voice = voice
        self._lang_code = lang_code
        self._tts = None  # lazy-loaded KokoroTTS
        self._pool = QThreadPool.globalInstance()
        self._shutdown = threading.Event()
        self._ready_lock = threading.Lock()
        self._warming = False
        self._active_lock = threading.Lock()
        # Queue at most ONE pending utterance while the model warms up
        # so the first reply the user hears is the latest one, not some
        # stale "thinking..." line. Later replies overwrite the pending
        # slot. Once the model loads we drain the slot.
        self._pending_text: str | None = None
        self._pending_lock = threading.Lock()

    # ----- lifecycle -----

    def start(self) -> None:
        """Kick off Kokoro load on a background worker."""
        with self._ready_lock:
            if self._tts is not None or self._warming or self._shutdown.is_set():
                return
            self._warming = True
        self._pool.start(_WarmupJob(self))

    def stop(self) -> None:
        self._shutdown.set()

    # ----- public API -----

    def speak(self, text: str) -> None:
        """Schedule text → speech playback on a pool worker.

        If the model is still warming up, park the text in the single
        pending slot — the warmup path drains it when ready. Only the
        *latest* pending text is kept so a slow-booting model never
        replays stale earlier replies.
        """
        text = (text or "").strip()
        if not text or self._shutdown.is_set():
            return
        if self._tts is None:
            with self._pending_lock:
                self._pending_text = text
            return
        self._pool.start(_SpeakJob(self, text))

    # ----- worker bodies -----

    def _warmup(self) -> None:
        try:
            from edgevox.tts.kokoro import KokoroTTS

            log.info("Loading Kokoro TTS for Rook...")
            tts = KokoroTTS(voice=self._voice, lang_code=self._lang_code)
            if self._shutdown.is_set():
                return
            self._tts = tts
            self.ready.emit()
            log.info("Kokoro TTS ready.")
            # Drain a reply that arrived during warmup so the user
            # actually hears Rook's first turn.
            with self._pending_lock:
                pending, self._pending_text = self._pending_text, None
            if pending:
                self._pool.start(_SpeakJob(self, pending))
        except Exception as e:
            log.exception("TTS warmup failed")
            self.error.emit(f"Voice output unavailable: {e}")
        finally:
            self._warming = False

    def _speak(self, text: str) -> None:
        if self._tts is None or self._shutdown.is_set():
            return
        # Serialise playback: the user hears one reply at a time. A
        # reply mid-flight is dropped rather than queued; the UI only
        # ever asks us to speak the latest reply.
        if not self._active_lock.acquire(blocking=False):
            return
        try:
            self.started.emit()
            try:
                audio = self._tts.synthesize(text)
            except Exception as e:
                log.exception("Kokoro synth failed")
                self.error.emit(f"TTS error: {e}")
                return
            if self._shutdown.is_set():
                return
            try:
                # Deferred import: play_audio imports sounddevice which
                # probes audio devices on import in some builds.
                from edgevox.audio import play_audio

                play_audio(audio, sample_rate=self._tts.sample_rate)
            except Exception as e:
                log.exception("Audio playback failed")
                self.error.emit(f"Audio playback: {e}")
        finally:
            self.finished.emit()
            self._active_lock.release()


class _WarmupJob(QRunnable):
    def __init__(self, worker: TTSWorker) -> None:
        super().__init__()
        self._worker = worker

    @Slot()
    def run(self) -> None:  # pragma: no cover — event-loop-driven
        self._worker._warmup()


class _SpeakJob(QRunnable):
    def __init__(self, worker: TTSWorker, text: str) -> None:
        super().__init__()
        self._worker = worker
        self._text = text

    @Slot()
    def run(self) -> None:  # pragma: no cover — event-loop-driven
        self._worker._speak(self._text)


__all__ = ["TTSWorker"]
