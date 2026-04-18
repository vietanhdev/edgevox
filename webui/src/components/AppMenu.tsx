// AppMenu — hamburger dropdown in the top bar.
//
// Keeps the top bar tidy on narrow viewports + exposes controls that
// don't fit there: persona picker, about, keyboard shortcuts reference.
// Clicking outside or pressing Escape closes the dropdown.

import { useEffect, useRef, useState } from "react";
import { Info, Keyboard, Menu, RefreshCw, Volume2, VolumeX } from "lucide-react";

export type Persona = "grandmaster" | "casual" | "trash_talker";

interface AppMenuProps {
  muted: boolean;
  persona: Persona;
  onNewGame: () => void;
  onToggleMute: () => void;
  onAbout: () => void;
  onShowShortcuts: () => void;
  accent: string;
  disabled?: boolean;
}

export function AppMenu({
  muted,
  persona,
  onNewGame,
  onToggleMute,
  onAbout,
  onShowShortcuts,
  accent,
  disabled,
}: AppMenuProps) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("mousedown", onClick);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onClick);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const choose = (fn: () => void) => () => {
    fn();
    setOpen(false);
  };

  return (
    <div ref={wrapRef} style={{ position: "relative" }}>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-label="menu"
        aria-expanded={open}
        disabled={disabled}
        style={{
          width: 28,
          height: 28,
          border: `1px solid ${open ? accent : "#1e2b3a"}`,
          background: open ? `${accent}14` : "transparent",
          color: open ? accent : "#9aa7b9",
          borderRadius: 6,
          cursor: disabled ? "default" : "pointer",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          transition: "border-color 120ms, color 120ms, background 120ms",
        }}
      >
        <Menu size={15} />
      </button>

      {open && (
        <div
          role="menu"
          style={{
            position: "absolute",
            top: 34,
            right: 0,
            minWidth: 240,
            background: "rgba(12, 18, 28, 0.98)",
            border: "1px solid #1e2b3a",
            borderRadius: 8,
            padding: 6,
            boxShadow: "0 10px 30px rgba(0, 0, 0, 0.5)",
            zIndex: 40,
            animation: "menuFade 140ms ease-out",
          }}
        >
          <MenuItem onClick={choose(onNewGame)} accent={accent}>
            <RefreshCw size={14} /> New game <Shortcut>⌘N</Shortcut>
          </MenuItem>
          <MenuItem onClick={choose(onToggleMute)} accent={accent}>
            {muted ? <VolumeX size={14} /> : <Volume2 size={14} />}{" "}
            {muted ? "Unmute sound effects" : "Mute sound effects"}
            <Shortcut>M</Shortcut>
          </MenuItem>
          <Divider />
          <PersonaRow persona={persona} accent={accent} />
          <Divider />
          <MenuItem onClick={choose(onShowShortcuts)} accent={accent}>
            <Keyboard size={14} /> Keyboard shortcuts
          </MenuItem>
          <MenuItem onClick={choose(onAbout)} accent={accent}>
            <Info size={14} /> About RookApp
          </MenuItem>
        </div>
      )}
      <style>
        {`@keyframes menuFade {
          from { opacity: 0; transform: translateY(-4px); }
          to { opacity: 1; transform: translateY(0); }
        }`}
      </style>
    </div>
  );
}

function MenuItem({
  onClick,
  children,
  accent,
}: {
  onClick: () => void;
  children: React.ReactNode;
  accent: string;
}) {
  return (
    <button
      role="menuitem"
      onClick={onClick}
      style={{
        width: "100%",
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "8px 10px",
        border: "none",
        background: "transparent",
        color: "#dbe4ef",
        fontSize: 13,
        borderRadius: 5,
        cursor: "pointer",
        textAlign: "left",
        transition: "background 100ms",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = `${accent}18`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = "transparent";
      }}
    >
      {children}
    </button>
  );
}

function Shortcut({ children }: { children: React.ReactNode }) {
  return (
    <span
      style={{
        marginLeft: "auto",
        fontFamily: "monospace",
        fontSize: 10,
        color: "#6a7a8d",
        padding: "1px 6px",
        border: "1px solid #2a3442",
        borderRadius: 3,
      }}
    >
      {children}
    </span>
  );
}

function Divider() {
  return (
    <div
      style={{
        height: 1,
        background: "#1e2b3a",
        margin: "4px 6px",
      }}
    />
  );
}

function PersonaRow({ persona, accent }: { persona: Persona; accent: string }) {
  // Persona switching at runtime requires a server restart (the agent
  // factory reads EDGEVOX_CHESS_PERSONA at bind time). For now this
  // row only *displays* the active persona — future work can wire a
  // soft-restart endpoint to make it clickable.
  const label = persona.replace("_", " ");
  return (
    <div
      style={{
        padding: "8px 10px",
        fontSize: 11,
        color: "#8796a8",
        display: "flex",
        alignItems: "center",
        gap: 10,
        fontFamily: "monospace",
        letterSpacing: "0.05em",
      }}
    >
      PERSONA
      <span
        style={{
          marginLeft: "auto",
          color: accent,
          fontWeight: 600,
          textTransform: "capitalize",
        }}
      >
        {label}
      </span>
    </div>
  );
}

export default AppMenu;
