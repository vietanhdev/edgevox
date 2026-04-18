// Tiny Web Audio SFX synth — no asset bundle, no loading jitter.
//
// Chess UIs live or die on move sounds, and shipping WAV files bloats
// the installer. Here we synthesize each effect from a couple of
// oscillator + envelope primitives. Less than a kilobyte of code,
// indistinguishable from a real click at normal volumes.
//
// A single shared ``AudioContext`` is created on the first user
// gesture so we don't hit Chrome / Safari's autoplay-policy freeze.

let ctx: AudioContext | null = null;
let muted = false;

function getCtx(): AudioContext | null {
  if (ctx) return ctx;
  const Ctx: typeof AudioContext =
    (window as typeof window & { webkitAudioContext?: typeof AudioContext }).AudioContext ??
    (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext!;
  if (!Ctx) return null;
  ctx = new Ctx();
  return ctx;
}

export function setMuted(flag: boolean) {
  muted = flag;
}

export function isMuted() {
  return muted;
}

// Envelope: short attack, fast decay, silent tail. Good for piece clicks.
function tone({
  freq,
  kind = "sine",
  attack = 0.005,
  hold = 0.02,
  decay = 0.08,
  gain = 0.12,
  detune = 0,
}: {
  freq: number;
  kind?: OscillatorType;
  attack?: number;
  hold?: number;
  decay?: number;
  gain?: number;
  detune?: number;
}) {
  if (muted) return;
  const c = getCtx();
  if (!c) return;
  if (c.state === "suspended") void c.resume();
  const t0 = c.currentTime;
  const osc = c.createOscillator();
  osc.type = kind;
  osc.frequency.value = freq;
  osc.detune.value = detune;
  const g = c.createGain();
  g.gain.setValueAtTime(0, t0);
  g.gain.linearRampToValueAtTime(gain, t0 + attack);
  g.gain.setValueAtTime(gain, t0 + attack + hold);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + attack + hold + decay);
  osc.connect(g);
  g.connect(c.destination);
  osc.start(t0);
  osc.stop(t0 + attack + hold + decay + 0.02);
}

// Pink-ish noise burst for captures — crunchier than a pure tone.
function noiseBurst({ duration = 0.07, gain = 0.1 }: { duration?: number; gain?: number } = {}) {
  if (muted) return;
  const c = getCtx();
  if (!c) return;
  if (c.state === "suspended") void c.resume();
  const t0 = c.currentTime;
  const sampleCount = Math.floor(c.sampleRate * duration);
  const buf = c.createBuffer(1, sampleCount, c.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < sampleCount; i++) {
    // Simple pink-noise approximation: running average of white noise
    // weighted to the low end, then windowed with an exp decay.
    data[i] = (Math.random() * 2 - 1) * Math.exp((-i / sampleCount) * 5);
  }
  const src = c.createBufferSource();
  src.buffer = buf;
  const g = c.createGain();
  g.gain.value = gain;
  src.connect(g);
  g.connect(c.destination);
  src.start(t0);
}

// ----- public SFX API -----

/** User or engine moves a non-capturing piece. A short tick. */
export function sfxMove() {
  tone({ freq: 520, kind: "triangle", hold: 0.015, decay: 0.06, gain: 0.09 });
  tone({ freq: 780, kind: "triangle", hold: 0.005, decay: 0.04, gain: 0.05, detune: 6 });
}

/** A capture — two-tone clack + noise burst so it has weight. */
export function sfxCapture() {
  noiseBurst({ duration: 0.08, gain: 0.12 });
  tone({ freq: 300, kind: "square", hold: 0.01, decay: 0.08, gain: 0.07 });
}

/** Check — a tense minor-second beep. */
export function sfxCheck() {
  tone({ freq: 660, kind: "sawtooth", hold: 0.05, decay: 0.12, gain: 0.1 });
  tone({ freq: 700, kind: "sawtooth", hold: 0.05, decay: 0.12, gain: 0.1, detune: 0 });
}

/** Checkmate / game over — descending minor triad. */
export function sfxGameOver() {
  if (muted) return;
  const c = getCtx();
  if (!c) return;
  const base = c.currentTime;
  [659, 523, 392].forEach((f, i) => {
    setTimeout(
      () => tone({ freq: f, kind: "triangle", hold: 0.15, decay: 0.2, gain: 0.11 }),
      i * 160,
    );
  });
  void base;
}

/** Promotion — bright ascending triad. */
export function sfxPromotion() {
  [523, 659, 784].forEach((f, i) => {
    setTimeout(() => tone({ freq: f, kind: "sine", hold: 0.08, decay: 0.12, gain: 0.09 }), i * 80);
  });
}

/** Illegal-move thunk — two low dull taps. */
export function sfxIllegal() {
  tone({ freq: 140, kind: "square", hold: 0.03, decay: 0.08, gain: 0.08 });
  setTimeout(() => tone({ freq: 120, kind: "square", hold: 0.03, decay: 0.08, gain: 0.08 }), 90);
}
