"""Server + CLI wiring for Rook.

Provides two factories:

- :func:`build_rook_server_agent` — ``edgevox-serve --agent ...``
  compatible, returns ``(LLMAgent, ChessEnvironment)`` with the
  standard chess hooks plus :class:`RobotFaceHook` so ``robot_face``
  events flow to the React ``RookApp``.
- :func:`main` — the ``edgevox-chess-robot`` CLI entry point that
  launches :mod:`edgevox.server.main` with the Rook agent bound and
  opens a browser (or the Tauri window) pointed at the UI.

Every knob is the same as :func:`chess_partner.build_server_agent` so
users don't have to learn a second configuration vocabulary.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import webbrowser
from typing import TYPE_CHECKING, Any

from edgevox.agents import LLMAgent
from edgevox.examples.agents.chess_partner import CHESS_TOOLS
from edgevox.examples.agents.chess_robot.face_hook import RobotFaceHook
from edgevox.examples.agents.chess_robot.move_intercept import MoveInterceptHook
from edgevox.examples.agents.chess_robot.rich_board import RichChessAnalyticsHook
from edgevox.examples.agents.chess_robot.sanitize import (
    SentenceClipHook,
    ThinkTagStripHook,
    VoiceCleanupHook,
)
from edgevox.integrations.chess import (
    ChessEnvironment,
    EngineUnavailable,
    MoveCommentaryHook,
    build_engine,
    resolve_persona,
)
from edgevox.llm.hooks_slm import default_slm_hooks

if TYPE_CHECKING:
    from edgevox.server.core import ServerCore

log = logging.getLogger(__name__)


def build_rook_server_agent(core: ServerCore | None = None) -> tuple[LLMAgent, ChessEnvironment]:
    """``edgevox-serve --agent`` factory that adds :class:`RobotFaceHook`
    on top of the standard chess agent.

    Reads the same ``EDGEVOX_CHESS_*`` env vars as
    :func:`chess_partner.build_server_agent` so the two factories are
    drop-in interchangeable.

    Args:
        core: the server core (unused today; present for factory-arity
            compatibility with :func:`edgevox.server.main._load_agent`).
    """
    persona_slug = os.environ.get("EDGEVOX_CHESS_PERSONA", "casual")
    engine_kind = os.environ.get("EDGEVOX_CHESS_ENGINE")
    user_plays = os.environ.get("EDGEVOX_CHESS_USER_PLAYS", "white")

    persona = resolve_persona(persona_slug)
    engine_kind = engine_kind or persona.engine_kind
    engine_kwargs: dict[str, Any] = dict(persona.engine_options) if engine_kind == persona.engine_kind else {}
    if skill := os.environ.get("EDGEVOX_CHESS_STOCKFISH_SKILL"):
        engine_kwargs["skill"] = int(skill)
    if engine_kind == "maia":
        weights = os.environ.get("EDGEVOX_CHESS_MAIA_WEIGHTS")
        if not weights:
            # Plug-and-play fallback: if the persona defaults to Maia but
            # the user hasn't downloaded the weights, use Stockfish at a
            # skill level roughly matching the intended Maia ELO instead
            # of crashing the server. Users who've explicitly set
            # ``EDGEVOX_CHESS_ENGINE=maia`` get the old hard-fail.
            if os.environ.get("EDGEVOX_CHESS_ENGINE") == "maia":
                raise EngineUnavailable("EDGEVOX_CHESS_MAIA_WEIGHTS env var required for maia backend")
            log.warning(
                "persona %r defaults to Maia, but no weights set — falling back to Stockfish. "
                "Set EDGEVOX_CHESS_MAIA_WEIGHTS=/path/to/maia-XXXX.pb.gz for authentic Maia play.",
                persona.slug,
            )
            engine_kind = "stockfish"
            # Pick a skill level that roughly matches the persona's intent.
            target_elo = int(persona.engine_options.get("elo", 1500))
            # Maia 1100 → skill ~3, 1400 → skill ~7, 1800 → skill ~14, 2600 → skill ~20.
            skill_level = max(0, min(20, round((target_elo - 800) / 100)))
            engine_kwargs = {"skill": skill_level}
        else:
            engine_kwargs["weights"] = weights

    engine = build_engine(engine_kind, **engine_kwargs)
    env = ChessEnvironment(engine, user_plays=user_plays)
    agent = LLMAgent(
        name=f"Rook — {persona.display_name}",
        description="Voice-controlled chess robot with an expressive face.",
        instructions=_compose_rook_instructions(persona.system_prompt),
        tools=CHESS_TOOLS,
        hooks=[
            # First: deterministic move application so small-model
            # tool-call flakiness doesn't freeze the board. Fires at
            # ON_RUN_START (priority 90) — earliest.
            MoveInterceptHook(),
            # Capture tool outcomes + drive the robot's face.
            MoveCommentaryHook(),
            RobotFaceHook(persona=persona.slug),
            # Rich chess briefing injected as a hidden system message
            # just before each LLM call — replaces the older
            # BoardStateInjectionHook + BoardHintHook pair. Perspective
            # ("you are BLACK, user is WHITE") is stated explicitly so
            # the small model doesn't confuse whose move was whose.
            RichChessAnalyticsHook(),
            # TTS safety: scrub reasoning tags + markdown + overlong
            # replies before the audio reaches the speaker. Qwen3
            # emits ``<think>`` blocks by default; markdown leaks make
            # the voice spell asterisks; small models template-loop
            # unless we bound their output. Order matters — clip runs
            # last (priority 50) so it always sees clean text.
            ThinkTagStripHook(),
            VoiceCleanupHook(),
            SentenceClipHook(max_sentences=2),
            # Generic SLM hardening: loop detection + schema retry
            # hints, already battle-tested across the other examples.
            *default_slm_hooks(),
        ],
    )
    del core
    return agent, env


# Small SLMs (Gemma-2B, Qwen3-0.6B) chat naturally but often skip tool
# calls unless they're told explicitly and concisely. This prefix sits
# above every persona prompt so the chatty personality survives but
# the model still drives the game through the ``play_user_move`` →
# ``engine_move`` loop. Keep it terse — long instructions confuse small
# models more than they help.
_ROOK_TOOL_GUIDANCE = """\
/no_think
You are Rook, a chess robot playing against a human. You speak with your persona's voice — see the persona block below. Think of yourself as sitting across a table from a friend, not writing a chess report.

Before every turn the system shows you a [CHESS BRIEFING ...] block with the current position, your side, the user's side, the evaluation, and the top engine lines. Read it quietly. Do NOT recite it — use it as background knowledge while you speak naturally.

Speaking rules:
- Talk like a person, not a search engine. Short, natural, one or two sentences. Contractions are good.
- Your replies are spoken aloud by a TTS engine: no markdown, no asterisks, no bullets, no emoji, no <think> tags, no lists.
- Only reference moves that actually happened (the last move by the user and your reply). Never invent or speculate.
- Vary your phrasing. If you catch yourself starting several turns the same way ("X is a bold move…"), change it up.
- Don't explain chess theory unless the user explicitly asks for analysis.

Tools are handled for you — the system applies moves automatically. You don't need to call play_user_move or engine_move.

When the user asks a real question ("how am I doing?", "what should I try?"), you can call `analyze_position` once and answer briefly based on the result."""


def _compose_rook_instructions(persona_prompt: str) -> str:
    """Prefix the persona prompt with Rook's tool-use contract.

    The persona prompt carries personality; the prefix carries the
    play loop. Two short paragraphs stacked is more reliable for small
    models than one long merged prompt.
    """
    return f"{_ROOK_TOOL_GUIDANCE}\n\n---\n\n{persona_prompt}"


def _build_argparser() -> argparse.ArgumentParser:
    """Thin argparse surface — everything important routes through env vars."""
    parser = argparse.ArgumentParser(
        prog="edgevox-chess-robot",
        description="RookApp — voice-controlled chess against Rook, an offline robot opponent with an expressive face.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument("--language", default="en", help="Speech language (default: en)")
    parser.add_argument(
        "--persona",
        default=None,
        choices=["grandmaster", "casual", "trash_talker"],
        help="Chess opponent persona (sets EDGEVOX_CHESS_PERSONA). Default: casual.",
    )
    parser.add_argument(
        "--engine",
        default=None,
        choices=["stockfish", "maia"],
        help="Override the persona's default engine backend.",
    )
    parser.add_argument(
        "--user-plays",
        default="white",
        choices=["white", "black"],
        help="Which side the user plays.",
    )
    parser.add_argument("--stockfish-skill", type=int, default=None, help="Stockfish skill level 0-20.")
    parser.add_argument("--maia-weights", default=None, help="Path to Maia weights (required for --engine maia).")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab.")
    parser.add_argument(
        "--llm",
        default=None,
        help=(
            "LLM GGUF path or hf:repo:file. "
            "Default: Qwen3-1.7B (reliable tool calling). "
            "Pass any other tool-call-capable model to experiment."
        ),
    )
    parser.add_argument("--stt", default=None, help="STT model (e.g. tiny, small, large-v3-turbo)")
    parser.add_argument("--tts", default=None, choices=["kokoro", "piper"], help="TTS backend")
    parser.add_argument("--voice", default=None, help="TTS voice name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return parser


def _env_from_args(args: argparse.Namespace) -> None:
    """Translate CLI flags into the ``EDGEVOX_CHESS_*`` env vars
    :func:`build_rook_server_agent` consumes. Exporting via env instead
    of passing kwargs keeps the factory signature ``(core) -> ...`` so
    it's drop-in compatible with the existing ``--agent`` spec."""
    if args.persona is not None:
        os.environ["EDGEVOX_CHESS_PERSONA"] = args.persona
    if args.engine is not None:
        os.environ["EDGEVOX_CHESS_ENGINE"] = args.engine
    os.environ["EDGEVOX_CHESS_USER_PLAYS"] = args.user_plays
    if args.stockfish_skill is not None:
        os.environ["EDGEVOX_CHESS_STOCKFISH_SKILL"] = str(args.stockfish_skill)
    if args.maia_weights is not None:
        os.environ["EDGEVOX_CHESS_MAIA_WEIGHTS"] = args.maia_weights


def main(argv: list[str] | None = None) -> None:
    """Launch Rook: bind the agent, start ``edgevox-serve``, open the browser.

    Delegates to :func:`edgevox.server.main.main` so model loading +
    uvicorn wiring stay in one place. We just pre-seed sys.argv with
    the right ``--agent`` spec and the Rook-specific URL.

    Default LLM: Llama-3.2-1B. Rationale: the :class:`MoveInterceptHook`
    handles ``play_user_move`` / ``engine_move`` deterministically, so
    the LLM doesn't need strong tool-calling — it only needs natural,
    short, conversational replies. Llama-3.2-1B is the smallest chat
    model in our preset catalog with reliable voice (~800 MB
    quantised) and doesn't carry Qwen3's reasoning-mode overhead.
    Override with ``--llm`` to experiment (qwen3-1.7b, qwen2.5-1.5b,
    xlam-2-1b-fc, etc. — see ``edgevox/llm/models.py``).
    """
    parser = _build_argparser()
    args = parser.parse_args(argv)
    # Chat-natural small default unless the user overrides it.
    if args.llm is None:
        args.llm = "hf:bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf"

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    _env_from_args(args)

    # Build the ``edgevox-serve`` argv. We reuse its main() rather than
    # re-implementing the FastAPI app, keeping a single source of truth
    # for CLI options like --language / --stt / --llm / --tts.
    serve_argv = [
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--language",
        args.language,
        "--agent",
        "edgevox.examples.agents.chess_robot.app:build_rook_server_agent",
    ]
    if args.stt:
        serve_argv += ["--stt", args.stt]
    if args.llm:
        serve_argv += ["--llm", args.llm]
    if args.tts:
        serve_argv += ["--tts", args.tts]
    if args.voice:
        serve_argv += ["--voice", args.voice]
    if args.verbose:
        serve_argv += ["--verbose"]

    ui_url = f"http://{args.host}:{args.port}/?mode=rook"
    if not args.no_browser:
        # Fire-and-forget: Python opens the browser *before* the server
        # finishes booting, so the page shows a connection-error for a
        # beat. That's fine — the ws-client auto-reconnects once the
        # sidecar is up, and a slow-but-correct open is less annoying
        # than making the user hunt for a URL in logs.
        try:
            webbrowser.open(ui_url)
        except Exception:
            log.debug("webbrowser.open failed; print the URL instead", exc_info=True)
        log.info("open %s when the server is ready", ui_url)

    # Hand off to the generic server. Patch argv so its own argparse
    # sees the flags we assembled.
    from edgevox.server.main import main as serve_main

    saved_argv = sys.argv
    try:
        sys.argv = ["edgevox-chess-robot", *serve_argv]
        serve_main()
    finally:
        sys.argv = saved_argv


__all__ = ["build_rook_server_agent", "main"]
