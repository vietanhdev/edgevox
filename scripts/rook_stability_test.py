#!/usr/bin/env python
"""Drive a realistic multi-turn game against a running Rook server.

Purpose: surface interaction-stability bugs that don't show up in unit
tests — things like stale env state, tool-call non-determinism, reply
hallucination, intercept-hook misfires.

Usage::

    uv run python scripts/rook_stability_test.py --port 8799

Prints a pass/fail summary for each turn. Returns non-zero exit code
on any failure so this can run in CI once we have a server-start
harness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import websockets


@dataclass
class Turn:
    """One user → server round-trip."""

    text: str
    expect_board_change: bool = True
    expect_tools_or_intercept: bool = True
    note: str = ""
    captured: dict[str, Any] = field(default_factory=dict)


async def run_turn(ws, turn: Turn, patience: float = 45.0) -> None:
    await ws.send(json.dumps({"type": "text_input", "text": turn.text}))
    last = time.time()
    captured = {
        "tools": [],
        "board_moves": [],
        "face": [],
        "reply": "",
        "error": None,
    }
    while time.time() - last < patience:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=patience - (time.time() - last))
        except asyncio.TimeoutError:
            break
        if isinstance(raw, bytes):
            continue
        m = json.loads(raw)
        last = time.time()
        t = m["type"]
        if t == "tool_call":
            captured["tools"].append((m["name"], m["ok"], m.get("error")))
        elif t == "chess_state" and m.get("last_move_san"):
            captured["board_moves"].append((m["last_move_san"], m["ply"]))
        elif t == "robot_face":
            captured["face"].append((m["mood"], m["tempo"]))
        elif t == "bot_text":
            captured["reply"] = m["text"]
        elif t == "state" and m["value"] == "listening":
            break
        elif t == "error":
            captured["error"] = m["message"]
    turn.captured = captured


def check_turn(turn: Turn) -> list[str]:
    """Return a list of failure messages for this turn, or empty if all good."""
    failures = []
    c = turn.captured
    if c["error"]:
        failures.append(f"server error: {c['error']}")
    if turn.expect_board_change and not c["board_moves"]:
        failures.append("no board update — move never applied")
    if not c["reply"].strip():
        failures.append("empty reply")
    reply_lower = c["reply"].lower()
    if "<think>" in reply_lower:
        failures.append("<think> leaked into reply")
    if any(md in c["reply"] for md in ("**", "```", "* ")):
        failures.append("markdown leaked into reply")
    return failures


async def main(host: str, port: int, persona: str) -> int:
    uri = f"ws://{host}:{port}/ws"
    print(f"connecting to {uri} (persona={persona})\n")

    async with websockets.connect(uri) as ws:
        # Drain boot messages.
        t0 = time.time()
        boot = []
        while time.time() - t0 < 3:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                if isinstance(raw, str):
                    m = json.loads(raw)
                    boot.append(m["type"])
            except asyncio.TimeoutError:
                break
        print(f"[boot] {boot}")
        assert "ready" in boot, "no ready message"
        assert "chess_state" in boot, "no chess_state priming"

        turns = [
            Turn("play e4", note="opening move"),
            Turn("knight to f3", note="second move"),
            Turn("bishop to c4", note="develop"),
            Turn("castle kingside", note="castle — may fail if path not clear"),
            Turn("how's the position?", expect_board_change=False, note="meta question"),
            Turn("new game", expect_board_change=False, note="reset flow"),
            Turn("d4", note="first move after reset"),
            Turn("c4", note="second move after reset"),
            Turn("Nc3", note="knight develop"),
        ]

        all_failures: list[tuple[int, Turn, list[str]]] = []
        for i, turn in enumerate(turns, start=1):
            print(f"\n━━ Turn {i}: {turn.text!r} ({turn.note}) ━━")
            start = time.time()
            await run_turn(ws, turn)
            dur = time.time() - start
            c = turn.captured
            moves = c["board_moves"]
            reply_short = c["reply"][:80] + ("…" if len(c["reply"]) > 80 else "")
            print(f"  {dur:.1f}s  moves={moves}  reply={reply_short!r}")
            failures = check_turn(turn)
            if failures:
                for f in failures:
                    print(f"  ✗ {f}")
                all_failures.append((i, turn, failures))
            else:
                print(f"  ✓ ok")

        # --- Summary ---
        print("\n" + "=" * 60)
        print(f"{len(turns)} turns, {len(all_failures)} with failures")
        if all_failures:
            for i, turn, fs in all_failures:
                print(f"  turn {i} ({turn.text!r}): {'; '.join(fs)}")
            return 1
        print("all turns passed ✓")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--persona", default="trash_talker")
    args = parser.parse_args()
    rc = asyncio.run(main(args.host, args.port, args.persona))
    sys.exit(rc)
