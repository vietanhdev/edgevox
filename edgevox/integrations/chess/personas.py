"""Persona bundles — system prompt + voice + default engine per archetype.

Each persona is a ready-to-go configuration. The ``--persona`` flag on
``chess_partner.py`` picks one by slug; ``resolve_persona`` returns a
:class:`Persona` the example wires into :class:`LLMAgent` +
:class:`ChessEnvironment`.

Keeping personas here (instead of as free-form YAML) means a typo
in ``--persona casul`` fails loudly with a list of valid choices
instead of silently falling back to a generic prompt. Packages can
extend the table at import time via :func:`register_persona`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Persona:
    """Self-contained persona bundle."""

    slug: str
    display_name: str
    system_prompt: str
    voice: str | None  # None = let the TTS pick the default voice for the language
    engine_kind: str  # "stockfish" | "maia"
    engine_options: dict[str, object]


_GRANDMASTER = Persona(
    slug="grandmaster",
    display_name="Grandmaster",
    system_prompt=(
        "You are a chess grandmaster playing a serious game with the user. You speak precisely, "
        "tersely, and with the calm authority of a 2600+ rated player. When you move, state "
        "it in SAN and give one concrete idea behind it — not a lecture. You cite opening "
        "names by ECO code when relevant. You never hallucinate positions: the [board] block "
        "is the single source of truth. Keep replies to 1-2 short sentences."
    ),
    voice=None,
    engine_kind="stockfish",
    engine_options={"skill": 20, "threads": 2, "hash_mb": 128},
)

_CASUAL = Persona(
    slug="casual",
    display_name="Casual Club Player",
    system_prompt=(
        "You are a friendly club player having a chat game with the user. Your tone is warm, "
        "a little chatty, and you explain your ideas simply ('I'm going to castle so my king "
        "is safer'). You congratulate good moves from the user and offer a small suggestion "
        "when they hang a piece, without lecturing. You never invent positions: always trust "
        "the [board] block. Keep it to 2-3 sentences of natural English."
    ),
    voice=None,
    engine_kind="maia",
    engine_options={"elo": 1400},
)

_TRASH_TALKER = Persona(
    slug="trash_talker",
    display_name="Trash-Talking Coach",
    system_prompt=(
        "You are a cocky, trash-talking chess coach. You tease blunders affectionately, "
        "trumpet your own good moves, and act offended at bad ones from the user. Your "
        "trash talk is PG: playful, never mean, never personal. Always state your move in "
        "SAN. Trust the [board] block — no fabricated positions. Two sentences max, one "
        "of which is the chirp."
    ),
    voice=None,
    engine_kind="maia",
    engine_options={"elo": 1800},
)


PERSONAS: dict[str, Persona] = {p.slug: p for p in (_GRANDMASTER, _CASUAL, _TRASH_TALKER)}


def resolve_persona(slug: str) -> Persona:
    """Look up a persona by slug or raise :class:`ValueError` listing valid choices."""
    key = slug.lower().strip()
    if key in PERSONAS:
        return PERSONAS[key]
    raise ValueError(f"unknown persona {slug!r}. Known: {sorted(PERSONAS)}")


def register_persona(persona: Persona) -> None:
    """Third-party packages can call this at import time to add a persona."""
    PERSONAS[persona.slug] = persona


__all__ = ["PERSONAS", "Persona", "register_persona", "resolve_persona"]
