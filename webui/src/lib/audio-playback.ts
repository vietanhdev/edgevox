// FIFO WAV player: queue WAV blobs from the server and play them back-to-back.
// We use HTMLAudioElement instead of AudioContext.decodeAudioData because the
// server already encoded a complete WAV per sentence — Audio handles the
// container parsing for us and works on every modern browser.
//
// When a caller registers ``setOnAudioLevel``, we route every audio element
// through a shared AudioContext → MediaElementAudioSourceNode →
// AnalyserNode graph and sample RMS on rAF. That powers the Rook face's
// lip-sync without changing the rest of the pipeline. When no callback
// is set, the graph is never built — zero cost for non-Rook sessions.

export type AudioLevelCallback = (rms: number) => void;

export class WavQueuePlayer {
  private queue: Blob[] = [];
  private playing = false;
  private current: HTMLAudioElement | null = null;
  private onIdle: (() => void) | null = null;

  private levelCb: AudioLevelCallback | null = null;
  private audioCtx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private analyserBuf: Uint8Array | null = null;
  private rafId = 0;

  enqueue(blob: Blob) {
    this.queue.push(blob);
    void this.tick();
  }

  /** Drop any queued/playing audio immediately. */
  flush() {
    this.queue = [];
    if (this.current) {
      this.current.pause();
      this.current.src = "";
      this.current = null;
    }
    this.playing = false;
    this.stopLevelSampler();
    this.levelCb?.(0);
  }

  isBusy() {
    return this.playing || this.queue.length > 0;
  }

  setOnIdle(cb: (() => void) | null) {
    this.onIdle = cb;
  }

  /** Register a 0..1 RMS callback that fires on rAF while audio plays.
   *
   * Safe to call before the first playback — the AudioContext is
   * created lazily on the first enqueued blob, which keeps the
   * browser's autoplay-policy window intact (construction before a
   * user gesture can leave the context in "suspended" state). */
  setOnAudioLevel(cb: AudioLevelCallback | null) {
    this.levelCb = cb;
    if (cb === null) {
      this.stopLevelSampler();
    }
  }

  private ensureAudioGraph() {
    if (this.audioCtx) return;
    const Ctx: typeof AudioContext = (window as typeof window & {
      webkitAudioContext?: typeof AudioContext;
    }).AudioContext ?? (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext!;
    if (!Ctx) return;
    this.audioCtx = new Ctx();
    this.analyser = this.audioCtx.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.3;
    this.analyserBuf = new Uint8Array(this.analyser.fftSize);
  }

  private routeThroughAnalyser(audio: HTMLAudioElement) {
    if (!this.levelCb) return;
    this.ensureAudioGraph();
    if (!this.audioCtx || !this.analyser) return;
    try {
      // Resume the context on first playback. Autoplay policies in
      // Chrome/Safari start contexts ``suspended`` until a user
      // gesture; the connect + play flow is the gesture.
      if (this.audioCtx.state === "suspended") void this.audioCtx.resume();
      const src = this.audioCtx.createMediaElementSource(audio);
      src.connect(this.analyser);
      this.analyser.connect(this.audioCtx.destination);
    } catch (e) {
      // MediaElementAudioSourceNode throws if the element was already
      // routed; silently fall back to the element's own output (no
      // lip-sync for this blob, but audio still plays).
      console.warn("lip-sync: analyser routing failed", e);
    }
  }

  private startLevelSampler() {
    if (!this.levelCb || !this.analyser || !this.analyserBuf) return;
    const analyser = this.analyser;
    const buf = this.analyserBuf;
    const cb = this.levelCb;
    const tick = () => {
      analyser.getByteTimeDomainData(buf as unknown as Uint8Array<ArrayBuffer>);
      let sumSq = 0;
      for (let i = 0; i < buf.length; i++) {
        const v = (buf[i] - 128) / 128;
        sumSq += v * v;
      }
      const rms = Math.sqrt(sumSq / buf.length);
      cb(Math.min(1, rms * 2.5)); // gentle gain so normal speech hits ~0.5
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  }

  private stopLevelSampler() {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
  }

  private async tick() {
    if (this.playing) return;
    const next = this.queue.shift();
    if (!next) {
      this.onIdle?.();
      this.stopLevelSampler();
      this.levelCb?.(0);
      return;
    }
    this.playing = true;
    const url = URL.createObjectURL(next);
    const audio = new Audio(url);
    this.current = audio;
    this.routeThroughAnalyser(audio);
    try {
      await audio.play();
      if (this.levelCb && !this.rafId) this.startLevelSampler();
      await new Promise<void>((resolve) => {
        audio.onended = () => resolve();
        audio.onerror = () => resolve();
      });
    } finally {
      URL.revokeObjectURL(url);
      this.current = null;
      this.playing = false;
      // Don't stop the sampler here — the next blob will keep it
      // rolling. ``tick()`` above handles the "queue empty" case.
      void this.tick();
    }
  }
}
