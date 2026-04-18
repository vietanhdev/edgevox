// In-page WebSocket stub.
//
// The Rook client connects via ``EdgeVoxWs`` which wraps the native
// ``WebSocket``. We replace ``window.WebSocket`` with a stub before the
// page script runs, then each test drives the stub through a global
// ``__rookTest`` control surface exposed on ``window``. That lets us
// assert UI reactions to specific server messages without spinning up
// the real Python server (which needs model downloads).

import type { Page } from "@playwright/test";

// ---- shape of messages we push to the client ----

export interface ChessStateMsg {
  type: "chess_state";
  fen: string;
  ply: number;
  turn: "white" | "black";
  last_move_uci?: string | null;
  last_move_san?: string | null;
  last_move_classification?: string | null;
  san_history?: string[];
  eval_cp?: number | null;
  mate_in?: number | null;
  win_prob_white?: number;
  opening?: string | null;
  is_game_over?: boolean;
  game_over_reason?: string | null;
  winner?: "white" | "black" | null;
}

export interface RobotFaceMsg {
  type: "robot_face";
  mood: "calm" | "curious" | "amused" | "worried" | "triumphant" | "defeated";
  gaze_x: number;
  gaze_y: number;
  persona: string;
  tempo: "idle" | "thinking" | "speaking";
  last_move_san?: string | null;
  is_game_over?: boolean;
}

/**
 * Install the WebSocket stub into the page. Run before navigation.
 *
 * The stub:
 * - Accepts any URL.
 * - Exposes ``window.__rookTest`` with ``push(msg)``, ``state(value)``,
 *   ``sent()`` (returns an array of client → server messages), and
 *   ``open()`` / ``close()`` lifecycle hooks.
 */
export async function installWsStub(page: Page) {
  await page.addInitScript(() => {
    type Listener = (ev: MessageEvent | Event | CloseEvent) => void;

    interface FakeSocket {
      url: string;
      readyState: number;
      onopen: Listener | null;
      onmessage: Listener | null;
      onclose: Listener | null;
      onerror: Listener | null;
      send: (data: string | ArrayBuffer) => void;
      close: () => void;
      binaryType: string;
    }

    const clients: FakeSocket[] = [];
    const sent: Array<string | ArrayBuffer> = [];

    class StubWebSocket implements FakeSocket {
      readonly url: string;
      readyState = 0;
      onopen: Listener | null = null;
      onmessage: Listener | null = null;
      onclose: Listener | null = null;
      onerror: Listener | null = null;
      binaryType = "blob";

      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSING = 2;
      static CLOSED = 3;

      readonly CONNECTING = 0;
      readonly OPEN = 1;
      readonly CLOSING = 2;
      readonly CLOSED = 3;

      constructor(url: string | URL) {
        this.url = typeof url === "string" ? url : url.toString();
        clients.push(this);
        // Open on next tick so listeners can attach first. If the
        // socket was closed before this fires (React StrictMode
        // double-invokes effects in dev, which closes the first
        // socket immediately), skip the transition.
        setTimeout(() => {
          if (this.readyState === 3) return;
          this.readyState = 1;
          this.onopen?.(new Event("open"));
        }, 0);
      }

      send(data: string | ArrayBuffer) {
        sent.push(data);
      }

      close() {
        this.readyState = 3;
        this.onclose?.(new CloseEvent("close"));
        // Remove from the dispatch roster so we don't redeliver
        // messages to a closed socket whose onmessage handler is
        // still attached.
        const i = clients.indexOf(this);
        if (i >= 0) clients.splice(i, 1);
      }
    }

    // Deliver to the LATEST open socket only. React StrictMode in
    // dev double-invokes effects — the first socket is immediately
    // closed and the second is the "real" one. Broadcasting to all
    // open sockets causes duplicate message delivery to a partially-
    // unmounted handler.
    function latestOpen() {
      for (let i = clients.length - 1; i >= 0; i--) {
        if (clients[i].readyState === 1) return clients[i];
      }
      return null;
    }

    (window as unknown as { __rookTest: unknown }).__rookTest = {
      push: (msg: unknown) => {
        const encoded = typeof msg === "string" ? msg : JSON.stringify(msg);
        const c = latestOpen();
        if (c) c.onmessage?.(new MessageEvent("message", { data: encoded }));
      },
      pushAudio: (blob: Blob) => {
        const c = latestOpen();
        if (c) c.onmessage?.(new MessageEvent("message", { data: blob }));
      },
      sent: () => [...sent],
      sentTexts: () => sent.filter((s): s is string => typeof s === "string"),
      clear: () => {
        sent.length = 0;
      },
      closeAll: () => {
        for (const c of clients) c.close();
      },
      clientCount: () => clients.filter((c) => c.readyState === 1).length,
    };

    // Install the stub. Tauri's internals check is guarded so this is
    // safe in browser-only mode.
    (window as unknown as { WebSocket: unknown }).WebSocket = StubWebSocket;

    // Also swallow audio-capture getUserMedia — tests never actually
    // record audio.
    if (navigator.mediaDevices) {
      Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
        configurable: true,
        value: async () => {
          throw new DOMException("mock: no mic", "NotAllowedError");
        },
      });
    }
  });
}

/**
 * Helper: send the priming messages the real server emits on connect.
 */
export async function pushBoot(page: Page) {
  await page.evaluate(() => {
    const api = (window as unknown as { __rookTest: { push: (m: unknown) => void } }).__rookTest;
    api.push({
      type: "chess_state",
      fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      ply: 0,
      turn: "white",
      last_move_uci: null,
      last_move_san: null,
      san_history: [],
      eval_cp: null,
      is_game_over: false,
      winner: null,
    });
    api.push({
      type: "ready",
      session_id: "test-session",
      language: "en",
      languages: ["en"],
      voice: "af_heart",
      voices: ["af_heart"],
      tts_sample_rate: 24000,
      sample_rate: 16000,
      frame_size: 512,
    });
    api.push({ type: "state", value: "listening" });
  });
}

/**
 * Push a robot_face event in one call.
 */
export async function pushFace(page: Page, face: Partial<RobotFaceMsg> & { mood?: string } = {}) {
  await page.evaluate((f) => {
    const api = (window as unknown as { __rookTest: { push: (m: unknown) => void } }).__rookTest;
    api.push({
      type: "robot_face",
      mood: "calm",
      gaze_x: 0,
      gaze_y: 0,
      persona: "casual",
      tempo: "idle",
      ...f,
    });
  }, face);
}

/**
 * Push a chess_state event. Only sparsely-populated fields need to be given.
 */
export async function pushChessState(page: Page, s: Partial<ChessStateMsg>) {
  await page.evaluate((state) => {
    const api = (window as unknown as { __rookTest: { push: (m: unknown) => void } }).__rookTest;
    api.push({
      type: "chess_state",
      fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      ply: 0,
      turn: "white",
      san_history: [],
      is_game_over: false,
      ...state,
    });
  }, s);
}

/**
 * Retrieve text frames the client sent over the WebSocket.
 */
export async function getSentTexts(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const api = (window as unknown as { __rookTest: { sentTexts: () => string[] } }).__rookTest;
    return api.sentTexts();
  });
}

/** Seed localStorage so the onboarding overlay is pre-dismissed. */
export async function skipOnboarding(page: Page) {
  await page.addInitScript(() => {
    try {
      localStorage.setItem("evox-chess-onboarded", "1");
    } catch {
      /* ignore */
    }
  });
}
