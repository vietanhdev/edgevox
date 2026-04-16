"""Concrete pipeline processors wrapping EdgeVox backends.

Each processor adapts an existing backend (STT, LLM, TTS, playback) to the
frame-based pipeline interface defined in ``edgevox.core.frames``.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Generator

from edgevox.audio import play_audio, player
from edgevox.core.frames import (
    AudioFrame,
    EndFrame,
    Frame,
    InterruptFrame,
    MetricsFrame,
    Processor,
    SentenceFrame,
    StopFrame,
    TextFrame,
    TranscriptionFrame,
    TTSAudioFrame,
)
from edgevox.core.pipeline import MAX_CHUNK_CHARS, _find_sentence_break
from edgevox.tts import BaseTTS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------


class STTProcessor(Processor):
    """Speech-to-text: AudioFrame -> TranscriptionFrame.

    Wraps ``BaseSTT.transcribe()`` (batch — full audio segment at once).
    Note: once ``transcribe()`` is running, it cannot be interrupted; the
    interrupt will only take effect after it returns.
    """

    def __init__(self, stt, language: str = "en"):
        self.stt = stt
        self.language = language
        self._interrupted = False

    def process(self, frame: Frame) -> Generator[Frame, None, None]:
        if self._interrupted:
            return
        if isinstance(frame, AudioFrame):
            t0 = time.perf_counter()
            text = self.stt.transcribe(frame.audio, language=self.language)
            t_stt = time.perf_counter() - t0
            audio_duration = len(frame.audio) / frame.sample_rate
            if self._interrupted:
                return
            yield MetricsFrame(metrics={"stt": t_stt, "audio_duration": audio_duration})
            if text and not text.isspace():
                yield TranscriptionFrame(text=text, stt_time=t_stt, audio_duration=audio_duration)
        else:
            yield frame

    def on_interrupt(self):
        self._interrupted = True


# ---------------------------------------------------------------------------
# SafetyMonitor — preempts on stop-words before the LLM is consulted
# ---------------------------------------------------------------------------


class SafetyMonitor(Processor):
    """Checks STT output for stop-words and preempts the pipeline.

    Sits between ``STTProcessor`` and ``LLMProcessor``. When it sees a
    stop-word in a ``TranscriptionFrame``, it:

    1. Does NOT forward the frame — the LLM never sees it.
    2. Emits a :class:`StopFrame` downstream so the agent-layer
       skill dispatcher can cancel any in-flight goals.
    3. Calls ``on_stop()`` if provided, so the application can set
       ``AgentContext.stop`` and wake up any worker threads.

    Inspired by the Brown/CMU "Safety Chip" (ICRA 2024) constraint-
    monitor pattern — safety reflexes must never share critical-path
    latency with the LLM.
    """

    DEFAULT_STOP_WORDS: tuple[str, ...] = (
        "stop",
        "halt",
        "freeze",
        "abort",
        "emergency",
    )

    def __init__(
        self,
        stop_words: tuple[str, ...] | None = None,
        on_stop=None,
    ):
        self._stop_words = tuple(w.lower() for w in (stop_words or self.DEFAULT_STOP_WORDS))
        self._on_stop = on_stop
        self._interrupted = False

    def process(self, frame: Frame) -> Generator[Frame, None, None]:
        if isinstance(frame, TranscriptionFrame):
            tokens = {t.strip(".,!?").lower() for t in frame.text.split()}
            if tokens & set(self._stop_words):
                log.info("SafetyMonitor: stop-word detected in %r", frame.text)
                yield StopFrame(reason=f"stop-word in: {frame.text!r}")
                if self._on_stop is not None:
                    try:
                        self._on_stop()
                    except Exception:
                        log.exception("SafetyMonitor on_stop raised")
                return
        yield frame

    def on_interrupt(self):
        self._interrupted = True


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class LLMProcessor(Processor):
    """LLM chat: TextFrame -> many TextFrames (token stream) + EndFrame.

    Wraps ``LLM.chat_stream()`` which yields tokens as a generator.
    """

    def __init__(self, llm):
        self.llm = llm
        self._interrupted = False

    def process(self, frame: Frame) -> Generator[Frame, None, None]:
        if self._interrupted:
            return
        if isinstance(frame, (TextFrame, TranscriptionFrame)):
            # Pass the input frame through first so downstream can observe it
            # (e.g., UI displays the user's transcription).
            yield frame
            t0 = time.perf_counter()
            first = True
            stream = self.llm.chat_stream(frame.text)
            try:
                for token in stream:
                    if self._interrupted:
                        break
                    if first:
                        yield MetricsFrame(metrics={"ttft": time.perf_counter() - t0})
                        first = False
                    yield TextFrame(text=token)
            finally:
                # Close in a daemon thread — llama-cpp's stream.close() can
                # block if the model is mid-forward-pass; we don't want to
                # freeze the main pipeline waiting for it.
                def _bg_close(s=stream):
                    with contextlib.suppress(Exception):
                        s.close()

                threading.Thread(target=_bg_close, daemon=True).start()
            yield EndFrame()
        else:
            yield frame

    def on_interrupt(self):
        self._interrupted = True


# ---------------------------------------------------------------------------
# Agent-driven LLM processor (routes through LLMAgent.run instead of LLM)
# ---------------------------------------------------------------------------


class AgentProcessor(Processor):
    """Pipeline processor that drives an :class:`~edgevox.agents.Agent`.

    Unlike :class:`LLMProcessor` which streams tokens from an ``LLM``,
    this runs a full agent turn (``agent.run(transcription.text, ctx)``)
    — which can involve tool calls, skill dispatch, handoffs, and
    safety-event cancellation — and yields the final reply as a single
    ``TextFrame`` + ``EndFrame``. Downstream sentence-splitting still
    works naturally because the reply becomes one text chunk.

    Wiring contract:

    - ``agent``: any object implementing the ``Agent`` protocol.
      Typically an ``LLMAgent`` or ``Router`` workflow pre-bound to a
      shared ``LLM`` instance.
    - ``deps``: passed through as ``AgentContext.deps`` (a
      ``SimEnvironment``, ROS2 node, etc.).
    - ``on_event``: optional observability callback that receives every
      ``AgentEvent`` fired during a turn — ``tool_call``,
      ``skill_goal``, ``handoff``, etc. Used by the REPL / TUI to
      render live feedback.

    Interruption: a pipeline-level ``pipeline.interrupt()`` call
    triggers ``on_interrupt`` here, which sets ``ctx.stop`` on the
    current turn's context so in-flight skills cancel immediately.
    """

    def __init__(self, agent, deps=None, on_event=None):
        self.agent = agent
        self.deps = deps
        self.on_event = on_event
        self._interrupted = False
        self._active_ctx = None  # current turn's AgentContext, set per-frame

    def process(self, frame: Frame) -> Generator[Frame, None, None]:
        if self._interrupted:
            return
        if isinstance(frame, (TextFrame, TranscriptionFrame)):
            from edgevox.agents import AgentContext, Session

            # Pass input through first so downstream sees the transcription
            yield frame
            t0 = time.perf_counter()
            ctx = AgentContext(
                session=Session(),
                deps=self.deps,
                on_event=self.on_event,
            )
            self._active_ctx = ctx

            try:
                result = self.agent.run(frame.text, ctx)
            except Exception:
                log.exception("AgentProcessor.run raised")
                yield EndFrame()
                self._active_ctx = None
                return

            yield MetricsFrame(metrics={"ttft": time.perf_counter() - t0})
            if result.reply:
                yield TextFrame(text=result.reply)
            yield EndFrame()
            self._active_ctx = None
        else:
            yield frame

    def on_interrupt(self):
        self._interrupted = True
        if self._active_ctx is not None:
            self._active_ctx.stop.set()


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------


class SentenceSplitter(Processor):
    """Accumulates LLM tokens into complete sentences.

    TextFrame tokens are buffered until a sentence boundary (``.!?``) is
    detected, then yielded as a ``SentenceFrame``.  Reuses the
    abbreviation-aware splitting logic from ``edgevox.core.pipeline``.
    """

    def __init__(self):
        self._buffer: str = ""
        self._interrupted = False

    def process(self, frame: Frame) -> Generator[Frame, None, None]:
        if self._interrupted:
            return
        if isinstance(frame, TextFrame):
            # Pass the token through for observers (e.g., ROS2 bridge)
            yield frame
            self._buffer += frame.text

            # Extract complete sentences
            while True:
                pos = _find_sentence_break(self._buffer)
                if pos is None:
                    break
                sentence = self._buffer[:pos].strip()
                if sentence:
                    yield SentenceFrame(text=sentence)
                self._buffer = self._buffer[pos:].lstrip()

            # Break very long clauses at comma/semicolon
            if len(self._buffer) > MAX_CHUNK_CHARS:
                for sep in ["; ", ", ", ": "]:
                    idx = self._buffer.rfind(sep, MAX_CHUNK_CHARS // 2)
                    if idx > 0:
                        chunk = self._buffer[: idx + len(sep)].strip()
                        if chunk:
                            yield SentenceFrame(text=chunk)
                        self._buffer = self._buffer[idx + len(sep) :]
                        break

        elif isinstance(frame, EndFrame):
            # Flush remaining text
            remaining = self._buffer.strip()
            if remaining:
                yield SentenceFrame(text=remaining)
            self._buffer = ""
            yield frame
        else:
            yield frame

    def on_interrupt(self):
        self._buffer = ""
        self._interrupted = True


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class TTSProcessor(Processor):
    """Text-to-speech: SentenceFrame -> TTSAudioFrame(s).

    Uses ``synthesize_stream()`` when the backend supports true streaming
    (e.g., Kokoro), otherwise falls back to batch ``synthesize()``.
    """

    def __init__(self, tts):
        self.tts = tts
        self._supports_stream = type(tts).synthesize_stream is not BaseTTS.synthesize_stream
        self._interrupted = False

    def process(self, frame: Frame) -> Generator[Frame, None, None]:
        if self._interrupted:
            return
        if isinstance(frame, SentenceFrame):
            # Pass the sentence through so downstream can display it before
            # we start generating audio (reduces perceived latency).
            yield frame
            t0 = time.perf_counter()
            if self._supports_stream:
                for chunk in self.tts.synthesize_stream(frame.text):
                    if self._interrupted:
                        return
                    yield TTSAudioFrame(audio=chunk, sample_rate=self.tts.sample_rate, sentence=frame.text)
            else:
                audio = self.tts.synthesize(frame.text)
                if self._interrupted:
                    return
                yield TTSAudioFrame(audio=audio, sample_rate=self.tts.sample_rate, sentence=frame.text)
            yield MetricsFrame(metrics={"tts_sentence": time.perf_counter() - t0})
        else:
            yield frame

    def on_interrupt(self):
        self._interrupted = True


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------


class PlaybackProcessor(Processor):
    """Plays TTS audio through speakers.  Yields ``InterruptFrame`` if
    playback is interrupted (``play_audio`` returns False).
    """

    def __init__(self):
        self._interrupted = False

    def process(self, frame: Frame) -> Generator[Frame, None, None]:
        if self._interrupted:
            yield InterruptFrame()
            return
        if isinstance(frame, TTSAudioFrame):
            completed = play_audio(frame.audio, sample_rate=frame.sample_rate)
            if not completed or self._interrupted:
                yield InterruptFrame()
                return
        else:
            yield frame

    def on_interrupt(self):
        self._interrupted = True
        player.interrupt()
