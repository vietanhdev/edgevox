// Custom titlebar for the Tauri desktop window.
//
// Tauri app runs with ``decorations: false`` so the OS never draws
// its own title bar. This component takes that job back: it's a
// draggable strip at the top with a brand mark on the left and
// minimize / maximize / close buttons on the right.
//
// In browser mode (no ``window.__TAURI_INTERNALS__``) this component
// renders nothing — the webpage lives inside whatever the user's
// browser gives them.
//
// Drag is done via ``data-tauri-drag-region`` which the webview
// recognises natively. Window controls invoke the Tauri window API.

import { useEffect, useState } from "react";
import { Minus, Square, X } from "lucide-react";

function isTauri(): boolean {
  return Boolean((window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__);
}

export function TitleBar({ accent = "#34d399" }: { accent?: string }) {
  const [inTauri] = useState(() => isTauri());
  const [maximised, setMaximised] = useState(false);

  useEffect(() => {
    if (!inTauri) return;
    let cancelled = false;
    (async () => {
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      const win = getCurrentWindow();
      const maxed = await win.isMaximized();
      if (!cancelled) setMaximised(maxed);
      const unlisten = await win.onResized(async () => {
        const m = await win.isMaximized();
        if (!cancelled) setMaximised(m);
      });
      return unlisten;
    })();
    return () => {
      cancelled = true;
    };
  }, [inTauri]);

  if (!inTauri) return null;

  const onMinimize = async () => {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    await getCurrentWindow().minimize();
  };
  const onMaximizeToggle = async () => {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    const win = getCurrentWindow();
    if (await win.isMaximized()) await win.unmaximize();
    else await win.maximize();
  };
  const onClose = async () => {
    const { getCurrentWindow } = await import("@tauri-apps/api/window");
    await getCurrentWindow().close();
  };

  return (
    <div
      data-tauri-drag-region
      style={{
        height: 32,
        flex: "0 0 auto",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 0 0 12px",
        background: "rgba(10, 14, 20, 0.85)",
        backdropFilter: "blur(10px)",
        borderBottom: "1px solid #141d2a",
        userSelect: "none",
        // Let the webview's drag-region handler take over.
        WebkitUserSelect: "none",
        WebkitAppRegion: "drag",
      } as React.CSSProperties}
    >
      <div
        data-tauri-drag-region
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          fontSize: 12,
          fontFamily: "ui-sans-serif, system-ui",
          color: "#9aa7b9",
          letterSpacing: "0.03em",
          pointerEvents: "none",
        }}
      >
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: 4,
            background: accent,
          }}
        />
        <span style={{ color: "#dbe4ef", fontWeight: 600 }}>RookApp</span>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "stretch",
          // Buttons explicitly opt OUT of the drag region.
          WebkitAppRegion: "no-drag",
        } as React.CSSProperties}
      >
        <TitleBarBtn onClick={onMinimize} aria-label="minimize">
          <Minus size={13} />
        </TitleBarBtn>
        <TitleBarBtn onClick={onMaximizeToggle} aria-label={maximised ? "unmaximize" : "maximize"}>
          <Square size={11} />
        </TitleBarBtn>
        <TitleBarBtn onClick={onClose} aria-label="close" hoverBg="#e53935" hoverColor="#fff">
          <X size={14} />
        </TitleBarBtn>
      </div>
    </div>
  );
}

function TitleBarBtn({
  onClick,
  children,
  hoverBg,
  hoverColor,
  ...rest
}: {
  onClick: () => void;
  children: React.ReactNode;
  hoverBg?: string;
  hoverColor?: string;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      onClick={onClick}
      style={{
        width: 46,
        height: 32,
        border: "none",
        background: "transparent",
        color: "#9aa7b9",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        transition: "background 80ms, color 80ms",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = hoverBg ?? "rgba(255, 255, 255, 0.06)";
        if (hoverColor) e.currentTarget.style.color = hoverColor;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = "transparent";
        e.currentTarget.style.color = "#9aa7b9";
      }}
    >
      {children}
    </button>
  );
}

export default TitleBar;
