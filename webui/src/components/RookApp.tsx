// RookApp — board-centric layout.
//
// Left column (~2/3): chessboard is the star. Captured-piece trays
// stack above/below the board; material edge pill, check pulse, and
// move-history strip live around it.
//
// Right column (~1/3): Rook face at the top (compact), speech bubble
// with the latest reply beneath it, then a scrollable chat transcript,
// then the text input at the bottom.
//
// Voice: hold SPACE to talk. Text: type in the input or hit "/" to
// focus it. Every turn shows up in the chat as both "you: ..." and
// "rook: ..." lines.
//
// When the server hasn't emitted robot_face yet (e.g. the server is
// running the generic agent), we still render a calm default face
// so the right column never goes blank.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Mic, MicOff, Send, X } from "lucide-react";
import { AppMenu, type Persona as MenuPersona } from "@/components/AppMenu";
import { ChessBoardPanel } from "@/components/ChessBoard";
import { LottieRobotFace } from "@/components/LottieRobotFace";
import type { Mood, Persona, Tempo } from "@/components/RobotFace";
import type { ChessStateMessage } from "@/components/chess-types";
import { createMicCapture } from "@/lib/audio-capture";
import { WavQueuePlayer } from "@/lib/audio-playback";
import {
  deriveMaterial,
  isCheckmateFromSan,
  isInCheckFromSan,
  PIECE_GLYPH,
} from "@/lib/chess-derived";
import {
  setMuted,
  sfxCapture,
  sfxCheck,
  sfxGameOver,
  sfxIllegal,
  sfxMove,
  sfxPromotion,
} from "@/lib/sfx";
import { useLipSync } from "@/lib/useLipSync";
import { usePushToTalk } from "@/lib/usePushToTalk";
import { EdgeVoxWs, type ServerMessage } from "@/lib/ws-client";

async function wsUrl(): Promise<string> {
  const tauriInternals = (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  if (tauriInternals) {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      return await invoke<string>("get_ws_url");
    } catch (e) {
      console.warn("Tauri get_ws_url failed; falling back to same-origin", e);
    }
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}

interface FaceState {
  mood: Mood;
  gazeX: number;
  gazeY: number;
  persona: Persona;
  tempo: Tempo;
  lastLine: string;
}

interface ChatEntry {
  id: string;
  role: "user" | "rook" | "rook-move" | "user-move";
  text: string;
  pending?: boolean;
}

const DEFAULT_FACE: FaceState = {
  mood: "calm",
  gazeX: 0,
  gazeY: 0,
  persona: "casual",
  tempo: "idle",
  lastLine: "",
};

const ONBOARD_KEY = "evox-chess-onboarded";
const MUTE_KEY = "evox-chess-muted";

export default function RookApp() {
  const [face, setFace] = useState<FaceState>(DEFAULT_FACE);
  const [chess, setChess] = useState<ChessStateMessage | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastReply, setLastReply] = useState<string>("");
  const [chatLog, setChatLog] = useState<ChatEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  // Persistent errors stay until dismissed. Transient errors auto-clear.
  const [errorPersistent, setErrorPersistent] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const [muted, setMutedState] = useState(() => {
    try {
      return localStorage.getItem(MUTE_KEY) === "1";
    } catch {
      return false;
    }
  });
  const [showOnboard, setShowOnboard] = useState(() => {
    try {
      return !localStorage.getItem(ONBOARD_KEY);
    } catch {
      return true;
    }
  });
  const [gameOverDismissed, setGameOverDismissed] = useState(false);
  const [viewportW, setViewportW] = useState(() => window.innerWidth);
  const [viewportH, setViewportH] = useState(() => window.innerHeight);

  useEffect(() => {
    const onResize = () => {
      setViewportW(window.innerWidth);
      setViewportH(window.innerHeight);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    setMuted(muted);
  }, [muted]);

  useEffect(() => {
    const prev = document.title;
    document.title = "RookApp";
    return () => {
      document.title = prev;
    };
  }, []);

  const wsRef = useRef<EdgeVoxWs | null>(null);
  const playerRef = useRef<WavQueuePlayer>(new WavQueuePlayer());
  const micRef = useRef<ReturnType<typeof createMicCapture> | null>(null);
  const pendingReplyRef = useRef<string>("");
  const pendingRookIdRef = useRef<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const lastChessRef = useRef<ChessStateMessage | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const mouth = useLipSync(playerRef.current);

  // SFX + engine-move chat chip. One handler so state changes land in
  // a single pass, keeping chat ordering deterministic.
  const onNewChessState = useCallback((msg: ChessStateMessage) => {
    const prev = lastChessRef.current;
    lastChessRef.current = msg;
    if (!msg.last_move_san) return;
    if (prev && prev.ply === msg.ply) return;

    // Chat chip for the engine's move. Heuristic (single-user desktop,
    // user plays white for now): the engine played if:
    // - msg.ply is even (after white+black pair), AND
    // - turn is now white (it's the user's turn again)
    // We avoid chipping user moves because their text is already in
    // the chat. We dedupe by looking at the last chip's SAN.
    const engineJustMoved =
      msg.turn === "white" && msg.ply >= 2 && msg.ply % 2 === 0;
    if (engineJustMoved) {
      const moveCardId = `move-rook-${msg.ply}-${msg.last_move_san}`;
      setChatLog((chat) => {
        if (chat.some((e) => e.id === moveCardId)) return chat;
        const next: ChatEntry[] = [
          ...chat,
          { id: moveCardId, role: "rook-move", text: msg.last_move_san ?? "" },
        ];
        return next.length > 50 ? next.slice(next.length - 50) : next;
      });
    }

    // SFX by move type.
    if (msg.is_game_over || msg.last_move_san.endsWith("#")) {
      sfxGameOver();
      return;
    }
    if (isInCheckFromSan(msg.last_move_san)) {
      sfxCheck();
      return;
    }
    if (msg.last_move_san.includes("=")) {
      sfxPromotion();
      return;
    }
    if (msg.last_move_san.includes("x")) {
      sfxCapture();
      return;
    }
    sfxMove();
  }, []);

  // Auto-scroll chat log.
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chatLog]);

  const addChatEntry = useCallback((entry: ChatEntry) => {
    setChatLog((prev) => {
      const next = [...prev, entry];
      // Cap at 50 entries so long games don't bloat the DOM.
      return next.length > 50 ? next.slice(next.length - 50) : next;
    });
  }, []);

  const updateChatEntry = useCallback((id: string, patch: Partial<ChatEntry>) => {
    setChatLog((prev) => prev.map((e) => (e.id === id ? { ...e, ...patch } : e)));
  }, []);

  const ensurePendingRook = useCallback(() => {
    if (pendingRookIdRef.current) return pendingRookIdRef.current;
    const id = `rook-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    pendingRookIdRef.current = id;
    addChatEntry({ id, role: "rook", text: "", pending: true });
    return id;
  }, [addChatEntry]);

  const finalizePendingRook = useCallback(() => {
    const id = pendingRookIdRef.current;
    if (!id) return;
    updateChatEntry(id, { pending: false });
    pendingRookIdRef.current = null;
  }, [updateChatEntry]);

  const handleJson = useCallback(
    (msg: ServerMessage) => {
      switch (msg.type) {
        case "ready":
          break;
        case "state":
          if (msg.value === "listening") {
            setIsBusy(false);
            setThinkingStart(null);
            setThinkingElapsed(0);
          } else if (msg.value === "thinking") {
            setIsBusy(true);
            setThinkingStart(Date.now());
          } else if (msg.value === "speaking") {
            setIsBusy(true);
            // Stop the timer at the "speaking" boundary — the user
            // gets audio feedback now, so the hourglass has done its
            // job.
            setThinkingStart(null);
          }
          break;
        case "bot_token": {
          // Only llm.chat_stream (legacy) emits per-token frames.
          // LLMAgent.run (the Rook path) skips tokens and emits one
          // bot_text at the end — the ``bot_text`` case below handles
          // that. Keeping this branch for compatibility.
          const id = ensurePendingRook();
          pendingReplyRef.current += msg.text;
          const text = pendingReplyRef.current;
          setLastReply(text);
          updateChatEntry(id, { text });
          break;
        }
        case "bot_text": {
          // Agent mode path — finalize or create the rook's reply.
          const streaming = pendingReplyRef.current.length > 0;
          pendingReplyRef.current = "";
          const reply = (msg.text ?? "").trim();
          if (streaming) {
            // bot_token path already placed the entry; just finalize.
            finalizePendingRook();
          } else if (reply) {
            // Non-streaming path (LLMAgent) — create the rook entry
            // now so the user sees the reply.
            addChatEntry({
              id: `rook-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
              role: "rook",
              text: reply,
            });
            setLastReply(reply);
            pendingRookIdRef.current = null;
          }
          break;
        }
        case "user_text": {
          // STT gives us the transcribed user speech. For typed input,
          // the server also echoes this — but we already optimistically
          // rendered the typed text in ``sendText``, so dedupe: skip
          // when the last entry matches.
          setLastReply("");
          setChatLog((prev) => {
            const last = prev[prev.length - 1];
            if (last?.role === "user" && last.text === msg.text) return prev;
            const next: ChatEntry[] = [
              ...prev,
              {
                id: `user-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
                role: "user",
                text: msg.text,
              },
            ];
            return next.length > 50 ? next.slice(next.length - 50) : next;
          });
          break;
        }
        case "chess_state":
          setChess(msg);
          onNewChessState(msg);
          if (msg.ply === 0) setGameOverDismissed(false);
          break;
        case "robot_face":
          setFace({
            mood: msg.mood,
            gazeX: msg.gaze_x,
            gazeY: -msg.gaze_y,
            persona: msg.persona,
            tempo: msg.tempo,
            lastLine: msg.last_move_san ?? "",
          });
          break;
        case "error":
          setError(msg.message);
          setErrorPersistent(false);
          sfxIllegal();
          setTimeout(() => {
            setError((prev) => (prev === msg.message ? null : prev));
          }, 4000);
          break;
      }
    },
    [addChatEntry, ensurePendingRook, finalizePendingRook, onNewChessState, updateChatEntry],
  );

  const handleAudio = useCallback((blob: Blob) => {
    playerRef.current.enqueue(blob);
  }, []);

  const connect = useCallback(async () => {
    if (wsRef.current?.isOpen()) return;
    let url: string;
    try {
      url = await wsUrl();
    } catch (e) {
      setError(`sidecar: ${(e as Error).message}`);
      return;
    }
    const ws = new EdgeVoxWs(url, {
      onJson: handleJson,
      onAudio: handleAudio,
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
      onError: () => setError("WebSocket error"),
    });
    ws.connect();
    wsRef.current = ws;
  }, [handleJson, handleAudio]);

  useEffect(() => {
    void connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  const [recording, setRecording] = useState(false);
  const [thinkingStart, setThinkingStart] = useState<number | null>(null);
  const [thinkingElapsed, setThinkingElapsed] = useState<number>(0);

  // Tick the thinking-elapsed counter at 10 Hz while active.
  useEffect(() => {
    if (thinkingStart === null) return;
    const id = setInterval(() => {
      setThinkingElapsed((Date.now() - thinkingStart) / 1000);
    }, 100);
    return () => clearInterval(id);
  }, [thinkingStart]);

  const startMic = useCallback(async () => {
    if (!wsRef.current?.isOpen()) await connect();
    if (micRef.current) return;
    try {
      const mic = createMicCapture((pcm) => wsRef.current?.sendAudio(pcm));
      await mic.start();
      micRef.current = mic;
      setRecording(true);
    } catch (e) {
      const msg = (e as Error).message;
      // Friendlier error for the common mic-permission case.
      if (/NotAllowedError|permission/i.test(msg)) {
        setError("microphone permission denied — enable it in your browser's site settings, then click the mic button again");
        setErrorPersistent(true);
      } else if (/NotFoundError/i.test(msg)) {
        setError("no microphone detected");
        setErrorPersistent(true);
      } else {
        setError(`mic: ${msg}`);
        setErrorPersistent(false);
      }
    }
  }, [connect]);

  const stopMic = useCallback(() => {
    micRef.current?.stop();
    micRef.current = null;
    setRecording(false);
  }, []);

  // Click-to-talk: toggle mic with a button. Complements push-to-talk
  // (spacebar) — some users prefer one, some the other; we offer both.
  const toggleMic = useCallback(() => {
    if (recording) stopMic();
    else void startMic();
  }, [recording, startMic, stopMic]);

  const { held } = usePushToTalk({
    onStart: startMic,
    onStop: stopMic,
    // Disable spacebar-hold while toggle-recording is active so the
    // user doesn't accidentally stop mid-utterance.
    disabled: recording,
  });

  const dismissOnboard = useCallback(() => {
    if (!showOnboard) return;
    setShowOnboard(false);
    try {
      localStorage.setItem(ONBOARD_KEY, "1");
    } catch {
      /* ignore */
    }
  }, [showOnboard]);

  const sendText = useCallback(
    (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || !wsRef.current?.isOpen() || isBusy) return;
      wsRef.current.sendControl({ type: "text_input", text: trimmed });
      // Optimistically add to the chat log so the user sees feedback
      // instantly (server will echo a user_text frame that we'd
      // otherwise render on top — we skip the echo since we already
      // placed it).
      addChatEntry({
        id: `user-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        role: "user",
        text: trimmed,
      });
      setInputValue("");
      dismissOnboard();
    },
    [isBusy, dismissOnboard, addChatEntry],
  );

  const newGame = useCallback(() => sendText("new game"), [sendText]);

  // Click-to-move / drag-to-move handler. Sends the UCI as a
  // text_input frame; the server's MoveInterceptHook picks it up,
  // applies it to the env, and triggers the engine reply.
  const handleBoardMove = useCallback(
    (uci: string): boolean => {
      if (!wsRef.current?.isOpen() || isBusy) return false;
      wsRef.current.sendControl({ type: "text_input", text: uci });
      addChatEntry({
        id: `user-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        role: "user",
        text: uci,
      });
      return true;
    },
    [isBusy, addChatEntry],
  );

  const toggleMute = useCallback(() => {
    setMutedState((m) => {
      const next = !m;
      try {
        localStorage.setItem(MUTE_KEY, next ? "1" : "0");
      } catch {
        /* ignore */
      }
      return next;
    });
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const isTyping =
        document.activeElement === inputRef.current ||
        (document.activeElement as HTMLElement | null)?.tagName === "TEXTAREA";
      if (e.key === "Escape" && isBusy) {
        wsRef.current?.sendControl({ type: "interrupt" });
        playerRef.current.flush();
        setIsBusy(false);
        finalizePendingRook();
        e.preventDefault();
      } else if (e.key === "/" && !isTyping) {
        inputRef.current?.focus();
        e.preventDefault();
      } else if (e.key.toLowerCase() === "n" && !isTyping && !isBusy && (e.ctrlKey || e.metaKey)) {
        newGame();
        e.preventDefault();
      } else if (e.key.toLowerCase() === "m" && !isTyping) {
        toggleMute();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isBusy, newGame, toggleMute, finalizePendingRook]);

  const ringColor = useMemo(() => {
    if (face.persona === "grandmaster") return "#6ea8ff";
    if (face.persona === "trash_talker") return "#ff5ad1";
    if (face.persona === "casual") return "#ffb066";
    return "#34d399";
  }, [face.persona]);

  const material = useMemo(() => (chess ? deriveMaterial(chess.fen) : null), [chess]);
  const inCheck = isInCheckFromSan(chess?.last_move_san);
  const checkmate = isCheckmateFromSan(chess?.last_move_san);

  const whoseTurn = useMemo(() => {
    if (!chess) return "waiting…";
    if (chess.is_game_over) return "game over";
    if (chess.turn !== "white") {
      // Rook's turn — include the thinking elapsed if we have it, so
      // the user sees the bot isn't stuck.
      if (thinkingStart !== null && thinkingElapsed > 1) {
        return `thinking · ${thinkingElapsed.toFixed(1)}s`;
      }
      return "rook thinking…";
    }
    return "your turn";
  }, [chess, thinkingStart, thinkingElapsed]);

  // Board size — fills the left column. Constrained by viewport height.
  const boardSize = useMemo(() => {
    const leftColumnWidth = viewportW * 0.62 - 80; // minus side padding
    const heightBudget = viewportH - 220; // top bar + trays + move strip
    return Math.max(360, Math.min(leftColumnWidth, heightBudget, 720));
  }, [viewportW, viewportH]);

  // Face size on the right column: smaller than before so chat has room.
  const faceSize = useMemo(() => {
    const rightColWidth = viewportW * 0.38 - 40;
    return Math.max(160, Math.min(rightColWidth, 240));
  }, [viewportW]);

  // Board orientation follows the user's side so the user always
  // views from their own perspective. We infer the side from the
  // robot_face persona + who's-to-move at the very first move: if
  // Rook's first action is a move (ply 1) then Rook played white and
  // the user plays black. Default to white until we learn otherwise.
  const [userSide, setUserSide] = useState<"white" | "black">("white");
  useEffect(() => {
    if (!chess) return;
    // Heuristic: if ply 1 is already on the board and it's now the
    // user's turn, then the engine played first → user is black.
    if (chess.ply === 1 && chess.turn === "black") {
      // ply=1, turn=black means white just moved. If the user just
      // moved, they'd be playing white. But MoveInterceptHook always
      // applies both user + engine moves in the same turn, so ply=1
      // + turn=black only happens when the user played e4/d4 etc.
      // Don't auto-flip in that case.
    }
    // Explicit signal: the server-side env.user_plays determines the
    // side. We piggyback on the fact that if the robot has the
    // initiative (engine plays white), ply=1 turn=black is reached
    // without any user action — i.e., the priming chess_state + an
    // engine_move at connect. Today our server doesn't expose that;
    // user plays white by default. Leave the state mutable so future
    // servers can opt in.
  }, [chess]);
  const orientation = userSide;

  // Mobile / narrow viewport: stack columns vertically below 900px.
  const isNarrow = viewportW < 900;

  return (
    <div
      style={{
        height: "100vh",
        width: "100vw",
        background: "radial-gradient(circle at 50% 20%, #0f1622 0%, #060a12 75%)",
        color: "#dbe4ef",
        display: "flex",
        flexDirection: "column",
        fontFamily: "ui-sans-serif, system-ui, -apple-system, 'Segoe UI'",
        overflow: "hidden",
      }}
    >
      {/* ==== Top bar ==== */}
      <header
        style={{
          height: 44,
          flex: "0 0 auto",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 18px",
          borderBottom: "1px solid #141d2a",
          background: "rgba(10, 14, 20, 0.6)",
          backdropFilter: "blur(10px)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            fontSize: 12,
            fontFamily: "monospace",
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: 4,
              background: connected ? "#34d399" : "#ef4444",
            }}
          />
          <span style={{ fontWeight: 600, letterSpacing: "0.04em" }}>RookApp</span>
          <span style={{ opacity: 0.5 }}>· {connected ? "online" : "offline"}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <TurnPill label={whoseTurn} highlighted={chess?.turn === "white" && !chess?.is_game_over} color={ringColor} />
          <IconButton
            title={recording ? "Stop listening" : "Talk to Rook (or hold space)"}
            onClick={toggleMic}
            active={recording}
            activeColor={ringColor}
          >
            {recording ? <Mic size={16} /> : <MicOff size={16} />}
          </IconButton>
          <AppMenu
            accent={ringColor}
            muted={muted}
            persona={face.persona as MenuPersona}
            onNewGame={newGame}
            onToggleMute={toggleMute}
            onAbout={() => {
              setError(
                "RookApp — voice chess against Rook. Built on EdgeVox. Face animations from LottieFiles.",
              );
              setErrorPersistent(true);
            }}
            onShowShortcuts={() => {
              setError(
                "Shortcuts: hold SPACE to talk · / to type · ESC interrupts · M mutes · ⌘N new game",
              );
              setErrorPersistent(true);
            }}
          />
        </div>
      </header>

      {error && (
        <div
          role="alert"
          style={{
            position: "absolute",
            top: 52,
            left: "50%",
            transform: "translateX(-50%)",
            background: "#ef4444",
            color: "white",
            padding: "6px 14px 6px 16px",
            borderRadius: 6,
            fontFamily: "monospace",
            fontSize: 12,
            zIndex: 10,
            animation: "shake 140ms ease-in-out 2",
            display: "flex",
            alignItems: "center",
            gap: 10,
            maxWidth: "80%",
            boxShadow: "0 4px 14px rgba(239, 68, 68, 0.3)",
          }}
        >
          <span>{error}</span>
          {errorPersistent && (
            <button
              onClick={() => {
                setError(null);
                setErrorPersistent(false);
              }}
              aria-label="dismiss error"
              style={{
                background: "transparent",
                color: "white",
                border: "none",
                cursor: "pointer",
                padding: "0 2px",
                display: "flex",
                alignItems: "center",
              }}
            >
              <X size={14} />
            </button>
          )}
        </div>
      )}

      {/* ==== Main two-column layout ==== */}
      <main
        style={{
          flex: 1,
          display: "flex",
          flexDirection: isNarrow ? "column" : "row",
          minHeight: 0,
          overflow: "hidden",
        }}
      >
        {/* ---- LEFT: Board ---- */}
        <section
          style={{
            flex: isNarrow ? "0 0 auto" : "0 0 62%",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: isNarrow ? "12px 12px 0" : "20px 24px",
            gap: 8,
            minHeight: 0,
            overflow: "hidden",
          }}
        >
          {/* Top captured tray (rook's captures of your pieces) */}
          <CapturedTray
            pieces={material?.capturedByBlack ?? []}
            edge={material && material.whiteMaterialEdge < 0 ? -material.whiteMaterialEdge : 0}
            label="rook"
            width={boardSize}
          />

          <div style={{ position: "relative" }}>
            {inCheck && !checkmate && (
              <div
                style={{
                  position: "absolute",
                  inset: -14,
                  border: "2px solid rgba(239, 68, 68, 0.7)",
                  borderRadius: 6,
                  boxShadow: "0 0 28px rgba(239, 68, 68, 0.5) inset",
                  animation: "checkpulse 900ms ease-in-out infinite",
                  pointerEvents: "none",
                }}
              />
            )}
            {chess ? (
              <ChessBoardPanel
                state={chess}
                boardWidth={boardSize}
                orientation={orientation}
                onMove={handleBoardMove}
                canMove={chess.turn === "white" && !isBusy}
              />
            ) : (
              <div
                style={{
                  width: boardSize,
                  height: boardSize,
                  border: "1px dashed #1e2b3a",
                  borderRadius: 4,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontFamily: "monospace",
                  fontSize: 13,
                  opacity: 0.5,
                }}
              >
                waiting for server…
              </div>
            )}
          </div>

          {/* Bottom captured tray (your captures of rook's pieces) */}
          <CapturedTray
            pieces={material?.capturedByWhite ?? []}
            edge={material && material.whiteMaterialEdge > 0 ? material.whiteMaterialEdge : 0}
            label="you"
            width={boardSize}
          />

          {/* Move history */}
          <MoveHistory moves={chess?.san_history ?? []} />
        </section>

        {/* ---- RIGHT: Face + chat + input ---- */}
        <aside
          style={{
            flex: isNarrow ? "1 1 auto" : "1 1 38%",
            display: "flex",
            flexDirection: "column",
            borderLeft: isNarrow ? "none" : "1px solid #141d2a",
            borderTop: isNarrow ? "1px solid #141d2a" : "none",
            minWidth: 0,
            padding: "16px",
            gap: 10,
            minHeight: 0,
          }}
        >
          {/* Face + reply bubble */}
          <div
            style={{
              flex: "0 0 auto",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              position: "relative",
            }}
          >
            <div
              style={{
                position: "absolute",
                inset: 0,
                pointerEvents: "none",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                top: 0,
                opacity: held ? 0.95 : 0,
                transition: "opacity 120ms ease-out",
              }}
            >
              <div
                style={{
                  width: faceSize + 50,
                  height: faceSize + 50,
                  borderRadius: "50%",
                  border: `2px solid ${ringColor}`,
                  boxShadow: `0 0 32px ${ringColor}`,
                }}
              />
            </div>
            <LottieRobotFace
              mood={face.mood}
              gazeX={face.gazeX}
              gazeY={face.gazeY}
              mouth={mouth}
              tempo={face.tempo}
              persona={face.persona}
              size={faceSize}
            />

            {/* Speech bubble pointing up at the face */}
            <div
              style={{
                minHeight: 32,
                marginTop: 8,
                padding: lastReply ? "10px 14px" : "0",
                background: lastReply ? "rgba(20, 28, 40, 0.8)" : "transparent",
                border: lastReply ? `1px solid ${ringColor}33` : "none",
                borderRadius: 10,
                fontSize: 13,
                lineHeight: 1.4,
                color: face.tempo === "speaking" ? "#dbe4ef" : "#9aa7b9",
                opacity: lastReply ? 0.92 : 0,
                transition: "opacity 180ms ease-out, padding 120ms",
                width: "100%",
                textAlign: "center",
              }}
            >
              {lastReply}
            </div>
          </div>

          {/* Chat log */}
          <div
            style={{
              flex: 1,
              minHeight: 0,
              overflowY: "auto",
              background: "rgba(10, 14, 22, 0.6)",
              border: "1px solid #141d2a",
              borderRadius: 8,
              padding: "10px 12px",
              display: "flex",
              flexDirection: "column",
              gap: 8,
              fontSize: 13,
              lineHeight: 1.4,
            }}
          >
            {chatLog.length === 0 ? (
              <div
                style={{
                  flex: 1,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#5b6a7d",
                  fontSize: 12,
                  fontStyle: "italic",
                }}
              >
                talk or type a move to start…
              </div>
            ) : (
              chatLog.map((entry) => <ChatBubble key={entry.id} entry={entry} ringColor={ringColor} />)
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input */}
          <form
            onSubmit={(e) => {
              e.preventDefault();
              sendText(inputValue);
            }}
            style={{ display: "flex", gap: 6 }}
          >
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={
                isBusy
                  ? "rook thinking… (esc to interrupt)"
                  : "type a move or question…"
              }
              disabled={!connected || isBusy}
              style={{
                flex: 1,
                padding: "9px 12px",
                background: "#0d1520",
                border: `1px solid ${inputValue ? ringColor : "#1e2b3a"}`,
                borderRadius: 6,
                color: "#dbe4ef",
                fontFamily: "monospace",
                fontSize: 13,
                outline: "none",
                transition: "border-color 120ms",
              }}
            />
            <button
              type="submit"
              disabled={!inputValue.trim() || !connected || isBusy}
              aria-label="send"
              style={{
                padding: "8px 14px",
                background: inputValue.trim() && !isBusy ? ringColor : "#1e2b3a",
                color: inputValue.trim() && !isBusy ? "#0a0e14" : "#556575",
                border: "none",
                borderRadius: 6,
                cursor: inputValue.trim() && !isBusy ? "pointer" : "default",
                transition: "background 120ms",
                display: "flex",
                alignItems: "center",
              }}
            >
              <Send size={14} />
            </button>
          </form>

          {/* Bottom hint */}
          <div
            style={{
              fontFamily: "monospace",
              fontSize: 10,
              opacity: 0.4,
              textAlign: "center",
              letterSpacing: "0.03em",
            }}
          >
            <Kbd>space</Kbd> talk · <Kbd>/</Kbd> type · <Kbd>esc</Kbd> interrupt
          </div>
        </aside>
      </main>

      {/* Game-over banner */}
      {chess?.is_game_over && !gameOverDismissed && (
        <GameOverBanner
          winner={chess.winner}
          reason={chess.game_over_reason}
          ringColor={ringColor}
          onDismiss={() => setGameOverDismissed(true)}
          onNewGame={() => {
            setGameOverDismissed(true);
            newGame();
          }}
        />
      )}

      {/* Onboarding */}
      {showOnboard && connected && (
        <Onboarding ringColor={ringColor} onDismiss={dismissOnboard} />
      )}

      {/* Keyframes */}
      <style>
        {`
          @keyframes checkpulse {
            0%, 100% { transform: scale(1); opacity: 0.55; }
            50% { transform: scale(1.02); opacity: 0.95; }
          }
          @keyframes shake {
            0%, 100% { transform: translateX(-50%); }
            25% { transform: translate(calc(-50% - 4px), 0); }
            75% { transform: translate(calc(-50% + 4px), 0); }
          }
          @keyframes gameOverIn {
            from { transform: translate(-50%, 40px); opacity: 0; }
            to { transform: translate(-50%, 0); opacity: 1; }
          }
          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
          }
          @keyframes typingDot {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
            30% { transform: translateY(-2px); opacity: 1; }
          }
        `}
      </style>
    </div>
  );
}

// ================================================================
// Sub-components
// ================================================================

function ChatBubble({ entry, ringColor }: { entry: ChatEntry; ringColor: string }) {
  // Move chips — compact "played: Nf3" style. Distinct from speech bubbles.
  if (entry.role === "rook-move" || entry.role === "user-move") {
    const isRook = entry.role === "rook-move";
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          justifyContent: "center",
          animation: "fadeIn 180ms ease-out",
          opacity: 0.78,
          fontFamily: "monospace",
          fontSize: 11,
          padding: "2px 0",
        }}
      >
        <span
          style={{
            padding: "2px 8px",
            background: isRook ? `${ringColor}18` : "rgba(52, 211, 153, 0.14)",
            border: `1px solid ${isRook ? `${ringColor}55` : "rgba(52, 211, 153, 0.25)"}`,
            borderRadius: 10,
            color: isRook ? ringColor : "#8be2b6",
            letterSpacing: "0.03em",
          }}
        >
          {isRook ? "rook" : "you"} played <b style={{ fontWeight: 700 }}>{entry.text}</b>
        </span>
      </div>
    );
  }

  const isUser = entry.role === "user";
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start",
        animation: "fadeIn 180ms ease-out",
      }}
    >
      <div
        style={{
          fontSize: 10,
          opacity: 0.5,
          marginBottom: 2,
          letterSpacing: "0.04em",
          textTransform: "uppercase",
          color: isUser ? "#9aa7b9" : ringColor,
        }}
      >
        {isUser ? "you" : "rook"}
      </div>
      <div
        style={{
          maxWidth: "88%",
          padding: "7px 11px",
          background: isUser ? "rgba(52, 211, 153, 0.1)" : "rgba(30, 40, 56, 0.7)",
          border: `1px solid ${isUser ? "rgba(52, 211, 153, 0.2)" : "rgba(40, 55, 75, 0.7)"}`,
          borderRadius: 10,
          color: "#dbe4ef",
          opacity: entry.pending ? 0.75 : 1,
          fontSize: 13,
          lineHeight: 1.4,
          wordBreak: "break-word",
        }}
      >
        {entry.pending && !entry.text ? <TypingDots color={ringColor} /> : entry.text || " "}
      </div>
    </div>
  );
}

function TypingDots({ color }: { color: string }) {
  return (
    <span
      style={{
        display: "inline-flex",
        gap: 3,
        alignItems: "center",
      }}
    >
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          style={{
            width: 4,
            height: 4,
            borderRadius: "50%",
            background: color,
            opacity: 0.6,
            animation: `typingDot 1s ease-in-out ${i * 0.15}s infinite`,
          }}
        />
      ))}
    </span>
  );
}

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd
      style={{
        padding: "1px 5px",
        border: "1px solid #2a3442",
        borderRadius: 3,
        margin: "0 2px",
        fontFamily: "monospace",
        fontSize: 10,
      }}
    >
      {children}
    </kbd>
  );
}

function TurnPill({
  label,
  highlighted,
  color,
}: {
  label: string;
  highlighted: boolean | undefined;
  color: string;
}) {
  return (
    <div
      data-testid="turn-pill"
      style={{
        padding: "4px 10px",
        border: `1px solid ${highlighted ? color : "#1e2b3a"}`,
        borderRadius: 12,
        fontSize: 11,
        fontFamily: "monospace",
        color: highlighted ? color : "#8796a8",
        letterSpacing: "0.04em",
        background: highlighted ? `${color}12` : "transparent",
        transition: "all 180ms ease-out",
      }}
    >
      {label}
    </div>
  );
}

function IconButton({
  children,
  onClick,
  title,
  disabled,
  active,
  activeColor,
}: {
  children: React.ReactNode;
  onClick: () => void;
  title: string;
  disabled?: boolean;
  active?: boolean;
  activeColor?: string;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      disabled={disabled}
      style={{
        width: 28,
        height: 28,
        border: `1px solid ${active && activeColor ? activeColor : "#1e2b3a"}`,
        background: active && activeColor ? `${activeColor}22` : "transparent",
        color: disabled ? "#4a5563" : active && activeColor ? activeColor : "#9aa7b9",
        borderRadius: 6,
        fontSize: 13,
        cursor: disabled ? "default" : "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        transition: "border-color 120ms, color 120ms, background 120ms",
      }}
    >
      {children}
    </button>
  );
}

function CapturedTray({
  pieces,
  edge,
  label,
  width,
}: {
  pieces: string[];
  edge: number;
  label: string;
  width: number;
}) {
  const value: Record<string, number> = {
    Q: 9, q: 9, R: 5, r: 5, B: 3, b: 3, N: 3, n: 3, P: 1, p: 1,
  };
  const sorted = [...pieces].sort((a, b) => (value[b] ?? 0) - (value[a] ?? 0));
  return (
    <div
      style={{
        width,
        display: "flex",
        alignItems: "center",
        gap: 8,
        fontSize: 18,
        color: "#dbe4ef",
        minHeight: 24,
      }}
    >
      <span
        style={{
          fontFamily: "monospace",
          fontSize: 10,
          opacity: 0.5,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          minWidth: 44,
        }}
      >
        {label}
        {edge > 0 ? ` +${edge}` : ""}
      </span>
      <div style={{ flex: 1, letterSpacing: "-1px" }}>
        {sorted.length === 0 ? <span style={{ opacity: 0.2, fontSize: 11 }}>—</span> : sorted.map((p, i) => <span key={`${p}-${i}`}>{PIECE_GLYPH[p] ?? p}</span>)}
      </div>
    </div>
  );
}

function MoveHistory({ moves }: { moves: string[] }) {
  const pairs: [string, string | undefined][] = [];
  for (let i = 0; i < moves.length; i += 2) {
    pairs.push([moves[i], moves[i + 1]]);
  }
  const visible = pairs.slice(-8);
  if (visible.length === 0) {
    return (
      <div
        style={{
          height: 20,
          fontFamily: "monospace",
          fontSize: 11,
          opacity: 0.35,
        }}
      >
        no moves yet
      </div>
    );
  }
  return (
    <div
      style={{
        height: 20,
        display: "flex",
        alignItems: "center",
        gap: 10,
        fontFamily: "monospace",
        fontSize: 11,
        overflowX: "auto",
        color: "#9aa7b9",
        maxWidth: "100%",
      }}
    >
      {visible.map(([w, b], i) => {
        const n = pairs.length - visible.length + i + 1;
        return (
          <span key={n} style={{ whiteSpace: "nowrap" }}>
            <span style={{ opacity: 0.5 }}>{n}.</span> {w}
            {b ? ` ${b}` : ""}
          </span>
        );
      })}
    </div>
  );
}

function GameOverBanner({
  winner,
  reason,
  ringColor,
  onDismiss,
  onNewGame,
}: {
  winner: "white" | "black" | null | undefined;
  reason: string | null | undefined;
  ringColor: string;
  onDismiss: () => void;
  onNewGame: () => void;
}) {
  const headline = winner === null ? "Draw" : winner === "white" ? "You won!" : "Rook won!";
  const sub = reason ?? "game complete";
  return (
    <div
      style={{
        position: "absolute",
        bottom: 28,
        left: "50%",
        transform: "translateX(-50%)",
        background: "rgba(12, 18, 30, 0.96)",
        border: `1px solid ${ringColor}`,
        borderRadius: 10,
        padding: "14px 20px",
        display: "flex",
        alignItems: "center",
        gap: 16,
        animation: "gameOverIn 260ms ease-out both",
        boxShadow: "0 0 30px rgba(52, 211, 153, 0.25)",
        zIndex: 15,
      }}
    >
      <div>
        <div style={{ fontWeight: 600, fontSize: 18, color: ringColor }}>{headline}</div>
        <div style={{ fontSize: 12, opacity: 0.7, marginTop: 2 }}>by {sub}</div>
      </div>
      <button
        onClick={onNewGame}
        style={{
          padding: "6px 14px",
          background: ringColor,
          color: "#0a0e14",
          border: "none",
          borderRadius: 4,
          fontFamily: "monospace",
          fontSize: 12,
          fontWeight: 600,
          cursor: "pointer",
        }}
      >
        rematch
      </button>
      <button
        onClick={onDismiss}
        aria-label="dismiss"
        style={{
          background: "transparent",
          color: "#8796a8",
          border: "none",
          cursor: "pointer",
          padding: "4px 6px",
          display: "flex",
          alignItems: "center",
        }}
      >
        <X size={16} />
      </button>
    </div>
  );
}

function Onboarding({ ringColor, onDismiss }: { ringColor: string; onDismiss: () => void }) {
  return (
    <div
      onClick={onDismiss}
      style={{
        position: "absolute",
        inset: 0,
        background: "rgba(6, 10, 18, 0.9)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: 18,
        zIndex: 20,
        cursor: "pointer",
        backdropFilter: "blur(6px)",
      }}
    >
      <div
        style={{
          fontSize: 32,
          fontWeight: 600,
          color: ringColor,
          letterSpacing: "0.04em",
        }}
      >
        RookApp
      </div>
      <div
        style={{
          maxWidth: 540,
          textAlign: "center",
          fontSize: 15,
          lineHeight: 1.6,
          color: "#dbe4ef",
          padding: "0 24px",
        }}
      >
        Play chess against <b style={{ color: ringColor }}>Rook</b> — a voice robot with a face.
        <br />
        <br />
        Hold <Kbd>space</Kbd> and say <i>"I play e4"</i>, or hit <Kbd>/</Kbd> and type it. Rook plays right back, with commentary.
        <br />
        <Kbd>esc</Kbd> interrupts · <Kbd>m</Kbd> mutes sound · <b>↻</b> starts a new game.
      </div>
      <div style={{ fontSize: 12, color: "#8796a8", opacity: 0.7 }}>
        click anywhere to start
      </div>
    </div>
  );
}
