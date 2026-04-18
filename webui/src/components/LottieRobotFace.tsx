// Lottie-backed robot face. Each mood / tempo maps to a full facial
// animation shipped as JSON under ``/lottie/``. The component crossfades
// when the target animation changes so persona / mood transitions read
// as smooth, not jump-cut.
//
// Animation assets: downloaded from LottieFiles (https://lottiefiles.com/)
// under a licensed paid account. Per-file licensing is LottieFiles'
// standard terms — see ``webui/public/lottie/ATTRIBUTION.md``.

import { useEffect, useMemo, useRef, useState } from "react";
import Lottie, { type LottieRefCurrentProps } from "lottie-react";

export type Mood = "calm" | "curious" | "amused" | "worried" | "triumphant" | "defeated";
export type Persona = "grandmaster" | "casual" | "trash_talker" | string;
export type Tempo = "idle" | "thinking" | "speaking";

export interface LottieRobotFaceProps {
  mood: Mood;
  gazeX: number; // -1..1, unused by Lottie (baked-in animations) — kept for API parity.
  gazeY: number;
  mouth: number; // 0..1, mouth aperture driver (used for speed-modulation during speaking)
  tempo: Tempo;
  persona: Persona;
  size?: number;
}

// Mood → animation file. Tempo overrides (thinking / speaking) take
// precedence over mood because the user's attention is on "what's
// happening right now" rather than "how does Rook feel overall".
const TEMPO_OVERRIDE: Record<Tempo, string | null> = {
  idle: null,
  thinking: "robot_thinking",
  speaking: "robot_speaking",
};

const MOOD_ANIM: Record<Mood, string> = {
  calm: "robot_idle",
  curious: "robot_searching",
  amused: "robot_happy",
  worried: "robot_worried",
  triumphant: "robot_celebrating",
  defeated: "robot_sad",
};

// Cache parsed animation JSON so we don't refetch on every mood change.
const ANIM_CACHE = new Map<string, Promise<object>>();

function loadAnimation(name: string): Promise<object> {
  if (!ANIM_CACHE.has(name)) {
    ANIM_CACHE.set(
      name,
      fetch(`/lottie/${name}.json`).then((r) => {
        if (!r.ok) throw new Error(`lottie fetch failed: ${name} (${r.status})`);
        return r.json();
      }),
    );
  }
  return ANIM_CACHE.get(name)!;
}

// Persona tint — the Lottie animations are already teal; we overlay a
// subtle glow filter so each persona keeps its colour identity.
const PERSONA_GLOW: Record<string, string> = {
  grandmaster: "drop-shadow(0 0 18px rgba(122, 168, 255, 0.45))",
  casual: "drop-shadow(0 0 18px rgba(255, 176, 102, 0.45))",
  trash_talker: "drop-shadow(0 0 20px rgba(255, 90, 209, 0.5))",
  default: "drop-shadow(0 0 16px rgba(52, 211, 153, 0.35))",
};

export function LottieRobotFace({
  mood,
  mouth,
  tempo,
  persona,
  size = 320,
}: LottieRobotFaceProps) {
  const target = TEMPO_OVERRIDE[tempo] ?? MOOD_ANIM[mood] ?? "robot_idle";
  const [animationData, setAnimationData] = useState<object | null>(null);
  const [current, setCurrent] = useState<string>(target);
  const lottieRef = useRef<LottieRefCurrentProps | null>(null);

  // Load the target animation when it changes.
  useEffect(() => {
    let cancelled = false;
    loadAnimation(target)
      .then((data) => {
        if (!cancelled) {
          setAnimationData(data);
          setCurrent(target);
        }
      })
      .catch((e) => console.warn("lottie load failed", e));
    return () => {
      cancelled = true;
    };
  }, [target]);

  // Modulate playback speed by mouth aperture during "speaking" so the
  // face actually feels tied to the audio.
  useEffect(() => {
    if (!lottieRef.current) return;
    if (tempo === "speaking") {
      // Peak at ~1.4× during loud speech, baseline 0.9× during pauses.
      lottieRef.current.setSpeed(0.9 + mouth * 0.5);
    } else if (tempo === "thinking") {
      lottieRef.current.setSpeed(0.85);
    } else {
      lottieRef.current.setSpeed(1.0);
    }
  }, [mouth, tempo]);

  const filter = PERSONA_GLOW[persona] ?? PERSONA_GLOW.default;

  // For accessibility testing — attributes the RobotFace SVG exposed
  // are preserved here so tests hit the same hooks.
  const testProps = useMemo(
    () => ({
      "data-testid": "robot-face",
      "data-mood": mood,
      "data-tempo": tempo,
      "data-persona": persona,
      "data-animation": current,
    }),
    [mood, tempo, persona, current],
  );

  return (
    <div
      {...testProps}
      style={{
        width: size,
        height: size,
        display: "inline-block",
        position: "relative",
        filter,
        transition: "filter 220ms ease-out",
      }}
    >
      {animationData ? (
        <Lottie
          lottieRef={lottieRef}
          animationData={animationData}
          loop
          autoplay
          style={{ width: "100%", height: "100%" }}
          rendererSettings={{ preserveAspectRatio: "xMidYMid meet" }}
        />
      ) : (
        <div
          style={{
            width: "100%",
            height: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#4a5563",
            fontSize: 12,
            fontFamily: "monospace",
          }}
        >
          loading…
        </div>
      )}
    </div>
  );
}

export default LottieRobotFace;
