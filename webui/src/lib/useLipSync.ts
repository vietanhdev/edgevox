// React hook — subscribe to a WavQueuePlayer's audio level and expose
// it as a reactive mouth-aperture value.
//
// The player emits RMS on rAF; we throttle React updates to ~30 fps
// via a ref + setState-only-on-change pattern, because useState every
// frame would trigger the RobotFace's rAF tween in tight contention
// with the lip-sync rAF and waste budget.

import { useEffect, useRef, useState } from "react";
import type { WavQueuePlayer } from "./audio-playback";

export function useLipSync(player: WavQueuePlayer | null | undefined): number {
  const [value, setValue] = useState(0);
  const lastRef = useRef(0);
  const lastEmitRef = useRef(0);

  useEffect(() => {
    if (!player) return;
    player.setOnAudioLevel((rms) => {
      const now = performance.now();
      // Emit at most every ~33 ms; a mouth that updates 60 Hz isn't
      // visually distinguishable from 30 Hz.
      if (now - lastEmitRef.current < 32) return;
      // Skip tiny jitter (< 3% change) — prevents a stream of
      // near-identical setState calls when the player is idle.
      if (Math.abs(rms - lastRef.current) < 0.03) return;
      lastRef.current = rms;
      lastEmitRef.current = now;
      setValue(rms);
    });
    return () => {
      player.setOnAudioLevel(null);
    };
  }, [player]);

  return value;
}
