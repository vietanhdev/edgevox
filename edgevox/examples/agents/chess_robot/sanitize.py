"""Reply sanitisation for Rook.

Three cleanups run at :data:`AFTER_LLM` before the reply reaches TTS:

- :class:`ThinkTagStripHook` — removes ``<think>...</think>`` blocks
  (and truncated ``<think>`` prefixes when the model ran out of tokens
  before closing the tag). Required for Qwen3-family models which
  default to reasoning mode — without stripping, the robot literally
  reads the model's internal monologue out loud.
- :class:`VoiceCleanupHook` — strips markdown artefacts (asterisks,
  backticks, leading list dashes) that TTS engines spell out.
- :class:`SentenceClipHook` — trims the reply to the first N sentences
  so the SLM can't run on past its point. Small models tend to repeat
  or template-loop when they overshoot; capping their output defends
  against that without changing generation parameters.

All hooks are idempotent and safe to stack — if nothing matches they
return ``None`` and the payload flows through unchanged.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

from edgevox.agents.hooks import AFTER_LLM, HookResult

if TYPE_CHECKING:
    from edgevox.agents.base import AgentContext


# ``<think>...</think>`` — greedy-match spans, but also handle the
# truncated case where the model never emits a closing tag (ran out of
# tokens). In that case we drop everything from ``<think>`` to EOF,
# which would otherwise be spoken as stream-of-consciousness.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_TRUNCATED_RE = re.compile(r"<think>.*$", re.DOTALL)

# Markdown artefacts a TTS engine reads verbatim.
_MARKDOWN_RE = re.compile(
    r"("
    r"\*\*|"  # bold
    r"__|"  # bold alt
    r"`+|"  # inline code / fences
    r"^\s*[-*+]\s|"  # list bullet at line start
    r"^\s*#+\s"  # header at line start
    r")",
    re.MULTILINE,
)

# Matched pairs of single ``*foo*`` / ``_foo_`` (italic/emphasis) —
# removed after the block patterns above so we don't eat bold ``**``.
# Keep the inner text; drop the asterisks/underscores.
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)([^\*\n]+?)(?<!\s)\*(?!\*)")
_UNDERSCORE_ITALIC_RE = re.compile(r"(?<!_)_(?!\s)([^_\n]+?)(?<!\s)_(?!_)")


# If the model's entire output was a <think> block, we substitute a
# short persona-agnostic filler so the speaker isn't mute. Random pick
# prevents the same phrase twice in a row when the model stalls on
# consecutive turns.
_FALLBACK_FILLERS = (
    "Hmm, let me look at that.",
    "Interesting position.",
    "Your move.",
    "Let's see how that plays.",
    "I see what you're doing.",
    "Okay, let's continue.",
)


class ThinkTagStripHook:
    """Remove ``<think>…</think>`` reasoning blocks before TTS.

    Qwen3 defaults to reasoning mode; even with ``/no_think`` in the
    system prompt, some prompts occasionally still emit them. This
    hook is a belt-and-braces cleanup at ``AFTER_LLM`` so nothing
    leaks to the speaker.
    """

    points = frozenset({AFTER_LLM})
    priority = 70  # run early in AFTER_LLM so other hooks see clean content

    def __call__(self, point: str, ctx: AgentContext, payload: dict) -> HookResult | None:
        content = payload.get("content") or ""
        if "<think>" not in content:
            return None
        cleaned = _THINK_BLOCK_RE.sub("", content)
        cleaned = _THINK_TRUNCATED_RE.sub("", cleaned)
        cleaned = cleaned.strip()
        if cleaned == content.strip():
            return None
        if not cleaned:
            # Model produced only a think block — substitute a short
            # filler so the robot doesn't go silent. Random pick avoids
            # "Let me think about that one" appearing twice in a row.
            cleaned = random.choice(_FALLBACK_FILLERS)
        new_payload = dict(payload)
        new_payload["content"] = cleaned
        return HookResult.replace(new_payload, reason="stripped <think> block")


class VoiceCleanupHook:
    """Strip markdown artefacts that TTS engines read literally.

    Keeps commentary readable-aloud: bold, italic, backticks, list
    bullets, headers. Tool calls are unaffected — we only touch
    ``content``.
    """

    points = frozenset({AFTER_LLM})
    priority = 65

    def __call__(self, point: str, ctx: AgentContext, payload: dict) -> HookResult | None:
        content = payload.get("content") or ""
        if not content:
            return None
        cleaned = _MARKDOWN_RE.sub("", content)
        cleaned = _ITALIC_RE.sub(r"\1", cleaned)
        cleaned = _UNDERSCORE_ITALIC_RE.sub(r"\1", cleaned)
        cleaned = cleaned.strip()
        if cleaned == content.strip():
            return None
        new_payload = dict(payload)
        new_payload["content"] = cleaned
        return HookResult.replace(new_payload, reason="markdown stripped for TTS")


class SentenceClipHook:
    """Clip the reply to the first N sentences.

    Small models (1-2B) often run on past their useful output,
    template-looping or repeating stock phrases ("I'll bring my bishop
    to c5 — solid" every turn). This hook bounds the damage: keep the
    first ``max_sentences`` sentences and drop the rest. The cutoff
    sits in the ``AFTER_LLM`` tier after think / markdown scrubbing
    (priority 50 so it runs last), so we always clip the clean text.

    A "sentence" ends at ``.``, ``!``, or ``?`` followed by whitespace
    or end-of-string. We keep the terminator so the clip reads
    naturally.
    """

    points = frozenset({AFTER_LLM})
    priority = 50

    # Match a sentence terminator followed by whitespace OR end of string.
    _SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, *, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def __call__(self, point: str, ctx: AgentContext, payload: dict) -> HookResult | None:
        content = (payload.get("content") or "").strip()
        if not content:
            return None
        parts = self._SPLIT_RE.split(content)
        if len(parts) <= self.max_sentences:
            return None
        clipped = " ".join(parts[: self.max_sentences]).strip()
        if clipped == content:
            return None
        return HookResult.replace(
            {**payload, "content": clipped},
            reason=f"clipped to {self.max_sentences} sentences",
        )


__all__ = ["SentenceClipHook", "ThinkTagStripHook", "VoiceCleanupHook"]
