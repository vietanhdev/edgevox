// Rook — the robot face. V2: polished shapes, persona-specific
// silhouettes, soft gradients, smoother tweening.
//
// Driven from three channels:
//   1. ``mood``   → brow angle + mouth curve + eye shape + cheek glow
//   2. ``gaze``   → pupil offset (-1..1 normalised)
//   3. ``mouth``  → 0..1, lip aperture (lip-sync target)
//
// Plus ambient blink + breathing animations and a tempo-driven
// antenna pulse. Every channel tweens on its own rAF so a mood change
// doesn't jump — it eases in over ~250 ms.
//
// SVG only — no external assets, no runtime cost beyond paint.

import { useEffect, useRef, useState } from "react";

export type Mood = "calm" | "curious" | "amused" | "worried" | "triumphant" | "defeated";
export type Persona = "grandmaster" | "casual" | "trash_talker" | string;
export type Tempo = "idle" | "thinking" | "speaking";

export interface RobotFaceProps {
  mood: Mood;
  gazeX: number;
  gazeY: number;
  mouth: number;
  tempo: Tempo;
  persona: Persona;
  size?: number;
}

// Persona palettes — each persona gets a base hue plus deep/light
// shades so the face reads as one material under varied lighting.
interface Palette {
  /** Main accent / highlight — antenna, brows, iris, smile line. */
  accent: string;
  /** Outer halo glow. */
  glow: string;
  /** Head base fill. */
  shell: string;
  /** Head darker shade — under-chin shadow. */
  shellDark: string;
  /** Rim light — top of head, inner eye white. */
  rim: string;
}

const PALETTES: Record<string, Palette> = {
  grandmaster: {
    accent: "#7aa8ff",
    glow: "rgba(122, 168, 255, 0.45)",
    shell: "#18202e",
    shellDark: "#0c131e",
    rim: "#2b3c58",
  },
  casual: {
    accent: "#ffb066",
    glow: "rgba(255, 176, 102, 0.45)",
    shell: "#1d1813",
    shellDark: "#0d0a07",
    rim: "#3a2a1d",
  },
  trash_talker: {
    accent: "#ff5ad1",
    glow: "rgba(255, 90, 209, 0.5)",
    shell: "#1a0f1d",
    shellDark: "#0c0712",
    rim: "#3d1836",
  },
  default: {
    accent: "#34d399",
    glow: "rgba(52, 211, 153, 0.45)",
    shell: "#0a0e14",
    shellDark: "#060a12",
    rim: "#1e3a2e",
  },
};

// Persona silhouette — slight variations in head shape to make the
// three robots feel like distinct characters rather than palette
// swaps. ``headY`` is vertical centre offset; ``headRx/Ry`` are the
// rounded-rect radius; ``antennaShape`` picks between dot, rod, double.
interface Silhouette {
  headRx: number;
  headRy: number;
  headYOffset: number;
  cheekSize: number;
  antennaHeight: number;
  antennaShape: "dot" | "dual" | "coil";
  browWidth: number;
  eyeShape: "round" | "diamond" | "oval";
}

const SILHOUETTES: Record<string, Silhouette> = {
  grandmaster: {
    headRx: 75,
    headRy: 82,
    headYOffset: -2,
    cheekSize: 3,
    antennaHeight: 18,
    antennaShape: "rod",
    browWidth: 28,
    eyeShape: "oval",
  } as any,
  casual: {
    headRx: 72,
    headRy: 76,
    headYOffset: 0,
    cheekSize: 5,
    antennaHeight: 14,
    antennaShape: "dot",
    browWidth: 26,
    eyeShape: "round",
  } as any,
  trash_talker: {
    headRx: 70,
    headRy: 78,
    headYOffset: 0,
    cheekSize: 4,
    antennaHeight: 20,
    antennaShape: "coil",
    browWidth: 30,
    eyeShape: "diamond",
  } as any,
  default: {
    headRx: 72,
    headRy: 78,
    headYOffset: 0,
    cheekSize: 4,
    antennaHeight: 16,
    antennaShape: "dot",
    browWidth: 28,
    eyeShape: "round",
  } as any,
};

interface MoodShape {
  eyeHeight: number;
  browAngle: number;
  browY: number;
  mouthCurve: number;
  mouthOpen: number;
  cheekGlow: number;
  tilt: number;
  eyeShine: number; // 0..1, extra sparkle — triumphant peaks
}

const MOOD_SHAPES: Record<Mood, MoodShape> = {
  calm: {
    eyeHeight: 0.8,
    browAngle: 0,
    browY: 0,
    mouthCurve: 0.12,
    mouthOpen: 0.05,
    cheekGlow: 0.2,
    tilt: 0,
    eyeShine: 0.4,
  },
  curious: {
    eyeHeight: 1.0,
    browAngle: 0.55,
    browY: -2,
    mouthCurve: 0.2,
    mouthOpen: 0.1,
    cheekGlow: 0.35,
    tilt: 3,
    eyeShine: 0.6,
  },
  amused: {
    eyeHeight: 0.5,
    browAngle: 0.45,
    browY: -1,
    mouthCurve: 0.9,
    mouthOpen: 0.18,
    cheekGlow: 0.65,
    tilt: 0,
    eyeShine: 0.8,
  },
  worried: {
    eyeHeight: 1.0,
    browAngle: -0.8,
    browY: 4,
    mouthCurve: -0.3,
    mouthOpen: 0.08,
    cheekGlow: 0.15,
    tilt: -3,
    eyeShine: 0.3,
  },
  triumphant: {
    eyeHeight: 0.35,
    browAngle: 0.9,
    browY: -3,
    mouthCurve: 1.0,
    mouthOpen: 0.32,
    cheekGlow: 1.0,
    tilt: 0,
    eyeShine: 1.0,
  },
  defeated: {
    eyeHeight: 0.45,
    browAngle: -1.0,
    browY: 5,
    mouthCurve: -0.85,
    mouthOpen: 0.04,
    cheekGlow: 0.1,
    tilt: -7,
    eyeShine: 0.15,
  },
};

const MOOD_LERP = 0.14;

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpShape(a: MoodShape, b: MoodShape, t: number): MoodShape {
  return {
    eyeHeight: lerp(a.eyeHeight, b.eyeHeight, t),
    browAngle: lerp(a.browAngle, b.browAngle, t),
    browY: lerp(a.browY, b.browY, t),
    mouthCurve: lerp(a.mouthCurve, b.mouthCurve, t),
    mouthOpen: lerp(a.mouthOpen, b.mouthOpen, t),
    cheekGlow: lerp(a.cheekGlow, b.cheekGlow, t),
    tilt: lerp(a.tilt, b.tilt, t),
    eyeShine: lerp(a.eyeShine, b.eyeShine, t),
  };
}

function useBlink(): boolean {
  const [blinking, setBlinking] = useState(false);
  useEffect(() => {
    let cancelled = false;
    function schedule() {
      const delay = 2200 + Math.random() * 3400;
      setTimeout(() => {
        if (cancelled) return;
        setBlinking(true);
        setTimeout(() => {
          if (cancelled) return;
          setBlinking(false);
          schedule();
        }, 110);
      }, delay);
    }
    schedule();
    return () => {
      cancelled = true;
    };
  }, []);
  return blinking;
}

function useBreathing(): number {
  const [t, setT] = useState(0);
  useEffect(() => {
    let raf = 0;
    let start = 0;
    function tick(now: number) {
      if (!start) start = now;
      setT(((now - start) / 1000) * 0.65);
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  return Math.sin(t * 2 * Math.PI) * 1.5;
}

function useLerpedShape(target: MoodShape): MoodShape {
  const [shape, setShape] = useState(target);
  const shapeRef = useRef(shape);
  shapeRef.current = shape;
  useEffect(() => {
    let raf = 0;
    function tick() {
      const next = lerpShape(shapeRef.current, target, MOOD_LERP);
      shapeRef.current = next;
      setShape(next);
      const delta =
        Math.abs(next.eyeHeight - target.eyeHeight) +
        Math.abs(next.browAngle - target.browAngle) +
        Math.abs(next.browY - target.browY) +
        Math.abs(next.mouthCurve - target.mouthCurve) +
        Math.abs(next.mouthOpen - target.mouthOpen) +
        Math.abs(next.cheekGlow - target.cheekGlow) +
        Math.abs(next.eyeShine - target.eyeShine) +
        Math.abs(next.tilt - target.tilt);
      if (delta > 0.02) raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target]);
  return shape;
}

export function RobotFace({
  mood,
  gazeX,
  gazeY,
  mouth,
  tempo,
  persona,
  size = 360,
}: RobotFaceProps) {
  const target = MOOD_SHAPES[mood] ?? MOOD_SHAPES.calm;
  const shape = useLerpedShape(target);
  const blink = useBlink();
  const breathe = useBreathing();
  const palette = PALETTES[persona] ?? PALETTES.default;
  const sil = SILHOUETTES[persona] ?? SILHOUETTES.default;

  const VB = 220;
  const half = VB / 2;

  const gx = gazeX * 12;
  const gy = -gazeY * 8;
  const bodyY = breathe + sil.headYOffset;

  const cheekOpacity = Math.max(0, shape.mouthCurve) * shape.cheekGlow;
  const ringIntensity =
    0.35 + shape.cheekGlow * 0.5 + (tempo === "speaking" ? 0.3 : 0) + (tempo === "thinking" ? 0.15 : 0);

  const mouthAperture = Math.min(0.5, shape.mouthOpen + mouth * 0.35);
  const eyeH = blink ? 0.05 : Math.max(0.08, shape.eyeHeight);

  // Unique gradient IDs per persona so multiple faces on one page don't
  // share defs (future multi-instance safety).
  const gradId = `rf-grad-${persona}`;
  const glowId = `rf-glow-${persona}`;

  return (
    <div
      data-testid="robot-face"
      data-mood={mood}
      data-tempo={tempo}
      data-persona={persona}
      style={{
        width: size,
        height: size,
        position: "relative",
        display: "inline-block",
      }}
    >
      <svg
        viewBox={`-${half} -${half} ${VB} ${VB}`}
        width={size}
        height={size}
        style={{
          filter: `drop-shadow(0 0 ${14 + ringIntensity * 26}px ${palette.glow})`,
          transition: "filter 160ms ease-out",
        }}
      >
        <defs>
          {/* Head gradient — top rim light + deep chin shadow, sells
             a 3-D volumetric look without 3-D rendering. */}
          <linearGradient id={gradId} x1="0" y1="-80" x2="0" y2="80" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor={palette.rim} />
            <stop offset="40%" stopColor={palette.shell} />
            <stop offset="100%" stopColor={palette.shellDark} />
          </linearGradient>
          {/* Inner radial glow for cheek pads + antenna bulb. */}
          <radialGradient id={glowId}>
            <stop offset="0%" stopColor={palette.accent} stopOpacity="0.85" />
            <stop offset="70%" stopColor={palette.accent} stopOpacity="0.25" />
            <stop offset="100%" stopColor={palette.accent} stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Outer halo ring */}
        <circle
          cx={0}
          cy={bodyY}
          r={98 + ringIntensity * 4}
          fill="none"
          stroke={palette.accent}
          strokeWidth={1.5}
          strokeOpacity={0.3 + ringIntensity * 0.45}
        />

        <g transform={`translate(0, ${bodyY}) rotate(${shape.tilt})`}>
          {/* Neck / base collar — a thin strip under the head so the
             robot reads as "on a body", not floating. */}
          <rect
            x={-22}
            y={72}
            width={44}
            height={8}
            rx={3}
            fill={palette.shellDark}
            stroke={palette.rim}
            strokeWidth={1}
          />

          {/* Head silhouette */}
          <rect
            x={-sil.headRx}
            y={-sil.headRy}
            width={sil.headRx * 2}
            height={sil.headRy * 2}
            rx={sil.headRx * 0.55}
            ry={sil.headRy * 0.55}
            fill={`url(#${gradId})`}
            stroke={palette.rim}
            strokeWidth={2}
          />

          {/* Top-plate seam — a subtle line that hints at a panel break */}
          <line
            x1={-sil.headRx + 16}
            y1={-sil.headRy + 18}
            x2={sil.headRx - 16}
            y2={-sil.headRy + 18}
            stroke={palette.rim}
            strokeWidth={1}
            opacity={0.6}
          />

          {/* Antenna — persona-shaped */}
          <Antenna
            shape={sil.antennaShape}
            height={sil.antennaHeight}
            y={-sil.headRy}
            accent={palette.accent}
            tempo={tempo}
          />

          {/* Brows */}
          <g transform={`translate(0, ${shape.browY})`}>
            <path
              d={browPath(-1, shape.browAngle, sil.browWidth)}
              stroke={palette.accent}
              strokeWidth={4.5}
              strokeLinecap="round"
              fill="none"
              opacity={0.92}
            />
            <path
              d={browPath(1, shape.browAngle, sil.browWidth)}
              stroke={palette.accent}
              strokeWidth={4.5}
              strokeLinecap="round"
              fill="none"
              opacity={0.92}
            />
          </g>

          {/* Eyes */}
          <EyePair
            accent={palette.accent}
            shell={palette.shell}
            eyeHeight={eyeH}
            gx={gx}
            gy={gy}
            shape={sil.eyeShape}
            shine={shape.eyeShine}
          />

          {/* Cheek glow — persona-variable, brightens on positive mood */}
          <circle cx={-44} cy={18} r={sil.cheekSize * 2.5} fill={`url(#${glowId})`} opacity={cheekOpacity} />
          <circle cx={44} cy={18} r={sil.cheekSize * 2.5} fill={`url(#${glowId})`} opacity={cheekOpacity} />

          {/* Mouth */}
          <Mouth
            accent={palette.accent}
            shellDark={palette.shellDark}
            curve={shape.mouthCurve}
            aperture={mouthAperture}
          />
        </g>
      </svg>

      {/* Tempo caption — small, under the face, disappears in idle */}
      <div
        data-testid="tempo-label"
        style={{
          position: "absolute",
          bottom: -14,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: "monospace",
          fontSize: 10,
          color: palette.accent,
          opacity: tempo === "idle" ? 0 : 0.55,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          pointerEvents: "none",
          transition: "opacity 160ms ease-out",
        }}
      >
        {tempo}
      </div>
    </div>
  );
}

// Brow shape — a cubic curve whose midpoint rises/falls with ``angle``.
// side = -1 for left brow, +1 for right.
function browPath(side: 1 | -1, angle: number, width: number): string {
  const inner = side * 10;
  const outer = side * (10 + width);
  const midX = (inner + outer) / 2;
  const baseY = -34;
  const curveY = baseY - side * angle * 8; // outer up when angle > 0 on right
  return `M ${inner} ${baseY} Q ${midX} ${curveY} ${outer} ${baseY - angle * 4}`;
}

// Antenna variants — dot, dual little dots, or coil-like spring.
function Antenna({
  shape,
  height,
  y,
  accent,
  tempo,
}: {
  shape: "dot" | "dual" | "coil" | "rod";
  height: number;
  y: number;
  accent: string;
  tempo: Tempo;
}) {
  const blink = tempo === "thinking";
  if (shape === "dual") {
    return (
      <g>
        <line x1={-8} y1={y} x2={-8} y2={y - height} stroke={accent} strokeWidth={1.5} opacity={0.75} />
        <line x1={8} y1={y} x2={8} y2={y - height} stroke={accent} strokeWidth={1.5} opacity={0.75} />
        <circle cx={-8} cy={y - height - 3} r={3.5} fill={accent} opacity={blink ? 1 : 0.7}>
          {blink && (
            <animate attributeName="opacity" values="0.3;1;0.3" dur="1.1s" repeatCount="indefinite" />
          )}
        </circle>
        <circle cx={8} cy={y - height - 3} r={3.5} fill={accent} opacity={blink ? 1 : 0.7} />
      </g>
    );
  }
  if (shape === "coil") {
    // Little spring to the bulb.
    return (
      <g>
        <path
          d={`M 0 ${y} C -6 ${y - 4}, 6 ${y - 8}, 0 ${y - 12} C -6 ${y - 16}, 6 ${y - 20}, 0 ${y - height}`}
          stroke={accent}
          strokeWidth={1.5}
          fill="none"
          opacity={0.85}
        />
        <circle cx={0} cy={y - height - 4} r={5} fill={accent} opacity={blink ? 1 : 0.8}>
          {blink && (
            <animate attributeName="opacity" values="0.3;1;0.3" dur="1.1s" repeatCount="indefinite" />
          )}
        </circle>
      </g>
    );
  }
  if (shape === "rod") {
    return (
      <g>
        <line x1={0} y1={y} x2={0} y2={y - height} stroke={accent} strokeWidth={2} opacity={0.8} />
        <rect x={-4} y={y - height - 6} width={8} height={3} rx={1} fill={accent} opacity={blink ? 1 : 0.8}>
          {blink && (
            <animate attributeName="opacity" values="0.3;1;0.3" dur="1.1s" repeatCount="indefinite" />
          )}
        </rect>
      </g>
    );
  }
  // default: simple dot
  return (
    <g>
      <line x1={0} y1={y} x2={0} y2={y - height} stroke={accent} strokeWidth={2} opacity={0.75} />
      <circle cx={0} cy={y - height - 3} r={4} fill={accent} opacity={blink ? 1 : 0.75}>
        {blink && <animate attributeName="opacity" values="0.3;1;0.3" dur="1.1s" repeatCount="indefinite" />}
      </circle>
    </g>
  );
}

function EyePair({
  accent,
  shell,
  eyeHeight,
  gx,
  gy,
  shape,
  shine,
}: {
  accent: string;
  shell: string;
  eyeHeight: number;
  gx: number;
  gy: number;
  shape: "round" | "diamond" | "oval";
  shine: number;
}) {
  const ry = 14 * eyeHeight;
  const rx = shape === "oval" ? 15 : shape === "diamond" ? 12 : 14;
  const pupilR = Math.min(6.5, ry * 0.5);

  // Render each eye as an ellipse socket (dark) + white sclera + accent iris + pupil + highlight.
  const Eye = ({ cx }: { cx: number }) => (
    <g>
      {/* Socket shadow — depth cue */}
      <ellipse cx={cx} cy={0} rx={rx + 2.5} ry={ry + 2.5} fill={shell} opacity={0.85} />
      {/* Sclera */}
      <ellipse cx={cx} cy={0} rx={rx} ry={ry} fill="#f5f7fa" opacity={0.96} />
      {/* Iris (accent) */}
      <circle cx={cx + gx} cy={gy} r={pupilR + 1.2} fill={accent} opacity={0.9} />
      {/* Pupil — darker accent dot */}
      <circle cx={cx + gx} cy={gy} r={pupilR * 0.5} fill="#0b1220" />
      {/* Catchlight — fixed offset from iris centre, doesn't move with gaze */}
      <circle cx={cx + gx - pupilR * 0.35} cy={gy - pupilR * 0.45} r={pupilR * 0.28} fill="#ffffff" opacity={0.7 + shine * 0.3} />
      {/* Second subtle shine — triumphant sells it */}
      {shine > 0.6 && (
        <circle cx={cx + gx + pupilR * 0.5} cy={gy + pupilR * 0.4} r={pupilR * 0.15} fill="#ffffff" opacity={shine * 0.8} />
      )}
    </g>
  );

  return (
    <g>
      <Eye cx={-26} />
      <Eye cx={26} />
      {/* Squint overlay — used when eyeHeight < 0.55 */}
      {eyeHeight < 0.55 && (
        <>
          <rect x={-44} y={-ry - 2} width={36} height={14 - ry} fill={shell} />
          <rect x={8} y={-ry - 2} width={36} height={14 - ry} fill={shell} />
        </>
      )}
    </g>
  );
}

function Mouth({
  accent,
  shellDark,
  curve,
  aperture,
}: {
  accent: string;
  shellDark: string;
  curve: number;
  aperture: number;
}) {
  // Mouth baseline centre.
  const y0 = 46;
  // Mouth extents widen slightly when smiling.
  const halfWidth = 28 + Math.max(0, curve) * 2;
  const ctrl = 14 * curve;

  // Lip outer curve.
  const lipPath = `M ${-halfWidth} ${y0} Q 0 ${y0 - ctrl} ${halfWidth} ${y0}`;

  // Open-mouth body: ellipse that follows the smile curve. Clamp so
  // it never escapes the lip.
  const apertureRy = Math.max(1, aperture * 16);
  const openCx = 0;
  const openCy = y0 + 2 - ctrl * 0.22;
  const openRx = Math.min(halfWidth - 4, 18);

  // Teeth hint — a thin bright line at top of the open area when
  // smiling wide. Makes "amused" / "triumphant" feel warmer.
  const showTeeth = curve > 0.4 && aperture > 0.1;

  return (
    <g>
      {/* Outer lip */}
      <path d={lipPath} stroke={accent} strokeWidth={4.5} strokeLinecap="round" fill="none" />
      {/* Open mouth interior */}
      {aperture > 0.03 && (
        <ellipse
          cx={openCx}
          cy={openCy}
          rx={openRx}
          ry={apertureRy}
          fill={shellDark}
          opacity={0.95}
        />
      )}
      {/* Teeth highlight */}
      {showTeeth && (
        <line
          x1={-openRx + 3}
          y1={openCy - apertureRy + 1}
          x2={openRx - 3}
          y2={openCy - apertureRy + 1}
          stroke={accent}
          strokeWidth={1.2}
          opacity={0.55}
        />
      )}
      {/* Chin dimple — tiny mark that sells the face has structure */}
      <line x1={-2} y1={y0 + 14} x2={2} y2={y0 + 14} stroke={accent} strokeWidth={1} opacity={0.25} />
    </g>
  );
}

export default RobotFace;
