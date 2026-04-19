"""Move sound effects — short synthesised piece-click tones.

Per-move audio cues so the user hears their move and Rook's reply
land on the board (the same UX lichess / chess.com ship). Every
tone is synthesised as a decayed sine burst via numpy + sounddevice,
so nothing is bundled as an asset — keeps the app licence clean and
avoids the ~1-2 MB of WAV files we'd otherwise need to ship.

Audio output goes to the PortAudio device picked in Settings (the
same one Kokoro TTS targets); a gentle exponential-decay envelope
keeps the clicks soft under speech when both fire close together.

Muted path: when :class:`MoveSfx` is instantiated with
``enabled=False`` every play call is a no-op. The window hides the
mute-toggle behind ``settings.sfx_muted`` so the user-facing switch
gates TTS *and* these clicks uniformly.
"""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)


class MoveSfx:
    """Synthesise + play short click tones per chess event.

    Fire-and-forget: each ``play_*`` call queues a waveform on a
    background PortAudio stream and returns immediately. If the audio
    device is already busy with TTS the call quietly fails (PortAudio
    raises on overlap) — we swallow the exception rather than block
    the UI thread on a lock.
    """

    _RATE = 22050  # lower than TTS rate on purpose — clicks need no fidelity

    def __init__(
        self,
        *,
        enabled: bool = True,
        output_device: int | None = None,
        volume: float = 0.25,
    ) -> None:
        self._enabled = enabled
        self._output_device = output_device
        self._volume = max(0.0, min(1.0, volume))
        self._play_lock = threading.Lock()
        self._have_audio = False
        self._numpy = None
        self._sounddevice = None
        if enabled:
            self._have_audio = self._probe()

    def _probe(self) -> bool:
        try:
            import numpy as np  # type: ignore
            import sounddevice as sd  # type: ignore
        except Exception:
            log.debug("SFX disabled — numpy or sounddevice unavailable", exc_info=True)
            return False
        self._numpy = np
        self._sounddevice = sd
        return True

    # ----- public API — one method per chess event -----

    def play_move(self) -> None:
        """Routine piece placement click."""
        self._play(self._tone(440.0, 0.04))

    def play_capture(self) -> None:
        """Double-beat click — higher chirp + lower thud."""
        assert self._numpy is not None or not self._have_audio
        np = self._numpy
        if np is None:
            return
        wave = np.concatenate(
            [
                self._tone(660.0, 0.03),
                self._tone(280.0, 0.07),
            ]
        )
        self._play(wave)

    def play_check(self) -> None:
        """Sharp rising two-note — attention cue for the opponent."""
        np = self._numpy
        if np is None:
            return
        wave = np.concatenate(
            [
                self._tone(600.0, 0.05),
                self._tone(880.0, 0.08),
            ]
        )
        self._play(wave)

    def play_castle(self) -> None:
        """Two-tap thud — king + rook landing."""
        np = self._numpy
        if np is None:
            return
        wave = np.concatenate(
            [
                self._tone(500.0, 0.05),
                np.zeros(int(self._RATE * 0.04), dtype=np.float32),
                self._tone(500.0, 0.05),
            ]
        )
        self._play(wave)

    def play_game_end(self, *, rook_won: bool) -> None:
        """Triadic win chime or descending minor lose-sting."""
        np = self._numpy
        if np is None:
            return
        # C major (triumphant) vs C-Ab-Eb descending minor (deflated).
        freqs = (523.25, 659.25, 783.99) if rook_won else (523.25, 415.30, 311.13)
        pieces = [self._tone(f, 0.18) for f in freqs]
        self._play(np.concatenate(pieces))

    # ----- waveform synthesis -----

    def _tone(self, freq_hz: float, duration_s: float):
        """Return a float32 sine burst with exponential decay envelope.

        Exponential decay (``e^(-8t/duration)``) makes the tone sound
        like a discrete click rather than a steady beep — closer to
        the wood-knock aesthetic of physical chess boards.
        """
        np = self._numpy
        if np is None:
            return None
        samples = int(self._RATE * duration_s)
        t = np.linspace(0.0, duration_s, samples, endpoint=False, dtype=np.float32)
        envelope = np.exp(-8.0 * t / max(duration_s, 1e-6))
        wave = np.sin(2.0 * np.pi * freq_hz * t) * envelope * self._volume
        return wave.astype(np.float32)

    def _play(self, wave) -> None:
        if not self._have_audio or wave is None:
            return
        sd = self._sounddevice
        # Serialise play calls so a rapid sequence doesn't thrash the
        # PortAudio callback. Each wave is ~60-200 ms — short enough
        # that holding the lock while starting playback doesn't stall
        # the caller perceptibly.
        with self._play_lock:
            try:
                sd.play(wave, self._RATE, device=self._output_device)
            except Exception:
                log.debug("SFX play failed (non-fatal)", exc_info=True)


def classify_move_sfx(san: str | None, *, is_game_over: bool = False) -> str:
    """Map a SAN move string to a ``MoveSfx`` method name.

    Returned string is one of ``"move"``, ``"capture"``, ``"check"``,
    ``"castle"``, ``"game_end"``. Callers dispatch on the result:

    * ``# `` / ``is_game_over`` → ``game_end`` (caller supplies
      ``rook_won`` flag separately)
    * ``+`` suffix → ``check``
    * ``O-O`` / ``O-O-O`` → ``castle``
    * ``x`` in SAN → ``capture``
    * default → ``move``

    Kept as a pure function so tests can exercise the full SAN →
    sound mapping without loading PortAudio.
    """
    if is_game_over:
        return "game_end"
    if not san:
        return "move"
    clean = san.rstrip("+#")
    if san.endswith("#"):
        return "game_end"
    if san.endswith("+"):
        return "check"
    if clean in ("O-O", "O-O-O"):
        return "castle"
    if "x" in clean:
        return "capture"
    return "move"


__all__ = ["MoveSfx", "classify_move_sfx"]
