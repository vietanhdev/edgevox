// Hold-to-talk hook: gates the mic on while a designated key (space
// by default) is pressed, and auto-submits "text_end" on release so
// the server knows the user finished speaking. Returns ``held`` for
// the UI to render a glowing ring around the robot face.

import { useEffect, useRef, useState } from "react";

export interface PushToTalkControls {
  onStart: () => void | Promise<void>;
  onStop: () => void;
  /** Key to hold. Default: ``" "`` (spacebar). */
  key?: string;
  /** Disable — e.g. while typing into an input. */
  disabled?: boolean;
  /** Minimum hold duration before a press is treated as a real PTT.
   *  Guards against accidental taps. Default 80 ms. */
  minHoldMs?: number;
}

export function usePushToTalk({
  onStart,
  onStop,
  key = " ",
  disabled = false,
  minHoldMs = 80,
}: PushToTalkControls): { held: boolean } {
  const [held, setHeld] = useState(false);
  const pressedAtRef = useRef<number | null>(null);
  const startedRef = useRef(false);

  useEffect(() => {
    if (disabled) return;

    const isKey = (ev: KeyboardEvent) =>
      ev.key === key || (key === " " && ev.code === "Space");

    const shouldIgnoreTarget = (ev: KeyboardEvent) => {
      // Don't fire while focus is in a text field — typing in a chat
      // box shouldn't open the mic.
      const t = ev.target as HTMLElement | null;
      if (!t) return false;
      const tag = t.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return true;
      if (t.isContentEditable) return true;
      return false;
    };

    const down = (ev: KeyboardEvent) => {
      if (ev.repeat || !isKey(ev) || shouldIgnoreTarget(ev)) return;
      ev.preventDefault();
      if (pressedAtRef.current !== null) return;
      pressedAtRef.current = performance.now();
      setHeld(true);
      // Delayed actual ``onStart`` so a fast tap doesn't briefly open
      // the mic + instantly close it (which the server treats as a
      // zero-speech turn).
      const started = pressedAtRef.current;
      setTimeout(() => {
        if (pressedAtRef.current !== started) return; // released already
        startedRef.current = true;
        void onStart();
      }, minHoldMs);
    };

    const up = (ev: KeyboardEvent) => {
      if (!isKey(ev)) return;
      if (pressedAtRef.current === null) return;
      ev.preventDefault();
      pressedAtRef.current = null;
      setHeld(false);
      if (startedRef.current) {
        startedRef.current = false;
        onStop();
      }
    };

    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    // Safety: if the window loses focus while held, stop the mic.
    const blur = () => {
      if (pressedAtRef.current !== null) {
        pressedAtRef.current = null;
        setHeld(false);
        if (startedRef.current) {
          startedRef.current = false;
          onStop();
        }
      }
    };
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
  }, [onStart, onStop, key, disabled, minHoldMs]);

  return { held };
}
