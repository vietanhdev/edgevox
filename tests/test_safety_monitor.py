"""Unit tests for the SafetyMonitor pipeline processor and StopFrame."""

from __future__ import annotations

from edgevox.core.frames import (
    Frame,
    InterruptFrame,
    StopFrame,
    TranscriptionFrame,
)
from edgevox.core.processors import SafetyMonitor


def _collect(processor, frames: list[Frame]) -> list[Frame]:
    out: list[Frame] = []
    for f in frames:
        for o in processor.process(f):
            out.append(o)
    return out


class TestSafetyMonitor:
    def test_stop_word_yields_stop_frame(self):
        mon = SafetyMonitor()
        out = _collect(mon, [TranscriptionFrame(text="please stop now")])
        assert len(out) == 1
        assert isinstance(out[0], StopFrame)

    def test_stop_word_suppresses_transcription(self):
        mon = SafetyMonitor()
        out = _collect(mon, [TranscriptionFrame(text="halt")])
        # must NOT contain the original TranscriptionFrame
        assert not any(isinstance(f, TranscriptionFrame) for f in out)

    def test_normal_text_passes_through(self):
        mon = SafetyMonitor()
        frames = [TranscriptionFrame(text="hello there")]
        out = _collect(mon, frames)
        assert len(out) == 1
        assert isinstance(out[0], TranscriptionFrame)
        assert out[0].text == "hello there"

    def test_case_insensitive(self):
        mon = SafetyMonitor()
        out = _collect(mon, [TranscriptionFrame(text="STOP")])
        assert any(isinstance(f, StopFrame) for f in out)

    def test_strips_punctuation(self):
        mon = SafetyMonitor()
        out = _collect(mon, [TranscriptionFrame(text="Stop!")])
        assert any(isinstance(f, StopFrame) for f in out)

    def test_custom_stop_words(self):
        mon = SafetyMonitor(stop_words=("cease",))
        out_stop = _collect(mon, [TranscriptionFrame(text="please cease")])
        assert any(isinstance(f, StopFrame) for f in out_stop)
        out_no = _collect(mon, [TranscriptionFrame(text="stop")])
        # Default word is NOT in the custom list
        assert not any(isinstance(f, StopFrame) for f in out_no)

    def test_on_stop_callback_fires(self):
        fired = []
        mon = SafetyMonitor(on_stop=lambda: fired.append(True))
        _collect(mon, [TranscriptionFrame(text="stop")])
        assert fired == [True]

    def test_on_stop_exception_doesnt_break_pipeline(self):
        def bad():
            raise RuntimeError("boom")

        mon = SafetyMonitor(on_stop=bad)
        # should not raise
        out = _collect(mon, [TranscriptionFrame(text="stop")])
        assert any(isinstance(f, StopFrame) for f in out)

    def test_non_transcription_frame_passes_through(self):
        mon = SafetyMonitor()
        out = _collect(mon, [InterruptFrame()])
        assert len(out) == 1
        assert isinstance(out[0], InterruptFrame)


class TestStopFrame:
    def test_is_distinct_from_interrupt_frame(self):
        assert StopFrame is not InterruptFrame
        s = StopFrame(reason="user said halt")
        assert s.reason == "user said halt"

    def test_default_reason_empty(self):
        assert StopFrame().reason == ""
