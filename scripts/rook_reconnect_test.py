#!/usr/bin/env python
"""Verify Rook stays stable across browser-refresh-style reconnects.

The real user complaint: "I cannot interact with it stably." The
observed symptom was moves failing to apply after a refresh because
the shared ``ChessEnvironment`` had mid-game state left over from a
prior session. This script simulates that pattern:

  1. Connect, play a few moves, disconnect.
  2. Reconnect. The first move MUST apply cleanly (fresh board).
  3. Play a few more moves, disconnect.
  4. Reconnect a third time, play — again, should be fresh.

Passes iff every reconnection starts with a clean board at ply 0 and
the next move applies successfully.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import websockets


async def drain_boot(ws, timeout: float = 3.0) -> dict:
    """Read priming messages until the server goes quiet."""
    priming = {"ready": False, "chess_state": None}
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.8)
        except asyncio.TimeoutError:
            break
        if isinstance(raw, bytes):
            continue
        m = json.loads(raw)
        if m["type"] == "ready":
            priming["ready"] = True
        elif m["type"] == "chess_state":
            priming["chess_state"] = m
    return priming


async def play_one(ws, text: str, patience: float = 30.0) -> dict:
    await ws.send(json.dumps({"type": "text_input", "text": text}))
    last = time.time()
    applied = []
    reply = ""
    while time.time() - last < patience:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=patience - (time.time() - last))
        except asyncio.TimeoutError:
            break
        if isinstance(raw, bytes):
            continue
        m = json.loads(raw)
        last = time.time()
        if m["type"] == "chess_state" and m.get("last_move_san"):
            applied.append((m["last_move_san"], m["ply"]))
        elif m["type"] == "bot_text":
            reply = m["text"]
        elif m["type"] == "state" and m["value"] == "listening":
            break
    return {"applied": applied, "reply": reply}


async def session(ws, moves: list[str]) -> list[dict]:
    boot = await drain_boot(ws)
    if not boot["ready"]:
        raise AssertionError("no ready message")
    cs = boot["chess_state"]
    if cs is None or cs["ply"] != 0:
        raise AssertionError(f"reconnect did not reset: priming ply={cs['ply'] if cs else None}")
    results = []
    for m in moves:
        r = await play_one(ws, m)
        results.append(r)
    return results


async def main(host: str, port: int) -> int:
    uri = f"ws://{host}:{port}/ws"
    print(f"reconnect stability test against {uri}\n")
    failures = []

    # Cycle 1: play some, disconnect.
    print("--- cycle 1: fresh connect → e4, Nf3 ---")
    async with websockets.connect(uri) as ws:
        results = await session(ws, ["play e4", "knight to f3"])
        for i, r in enumerate(results, 1):
            if not r["applied"]:
                failures.append(f"cycle 1 move {i} didn't apply")
            print(f"  move {i}: applied={r['applied']}  reply={r['reply'][:60]!r}")

    # Cycle 2: reconnect, play. Must start fresh.
    print("\n--- cycle 2: reconnect → d4 (should be ply 0) ---")
    async with websockets.connect(uri) as ws:
        results = await session(ws, ["d4", "c4"])
        for i, r in enumerate(results, 1):
            if not r["applied"]:
                failures.append(f"cycle 2 move {i} didn't apply")
            print(f"  move {i}: applied={r['applied']}  reply={r['reply'][:60]!r}")

    # Cycle 3: reconnect mid-game after cycle 2 should still reset.
    print("\n--- cycle 3: reconnect again → e4 ---")
    async with websockets.connect(uri) as ws:
        results = await session(ws, ["e4"])
        for i, r in enumerate(results, 1):
            if not r["applied"]:
                failures.append(f"cycle 3 move {i} didn't apply")
            print(f"  move {i}: applied={r['applied']}  reply={r['reply'][:60]!r}")

    print("\n" + "=" * 60)
    if failures:
        for f in failures:
            print(f"  ✗ {f}")
        return 1
    print("all 3 reconnect cycles stable ✓")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8799)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.host, args.port)))
