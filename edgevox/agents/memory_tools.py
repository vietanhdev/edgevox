"""LLM-callable tools that let an agent curate its own memory.

The :class:`MemoryStore` protocol already exposes ``add_fact`` /
``forget_fact`` / ``get_fact`` on the Python side, but exposing those
as ``Tool`` objects lets the LLM itself decide to persist or revoke a
fact during a turn — the "memory-as-tools" pattern popularised by
Anthropic's memory tool and the Claude Agent SDK.

Usage::

    from edgevox.agents.memory import JSONMemoryStore
    from edgevox.agents.memory_tools import memory_tools

    store = JSONMemoryStore("./memory.json")
    agent = LLMAgent(..., tools=memory_tools(store))

The factory builds three fresh :class:`Tool` instances bound to the
supplied store. Bind two different stores to two different agents and
nothing leaks between them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from edgevox.llm.tools import Tool

if TYPE_CHECKING:
    from edgevox.agents.memory import MemoryStore


def memory_tools(
    store: MemoryStore,
    *,
    include: tuple[str, ...] = ("remember_fact", "forget_fact", "recall_fact"),
) -> list[Tool]:
    """Build the memory-curation tools bound to ``store``.

    Args:
        store: The :class:`MemoryStore` the tools should read from and
            write to. Thread-safety is inherited from the store.
        include: Subset of tool names to expose. Defaults to all three;
            drop ``"forget_fact"`` to keep the LLM from revoking its
            own memory, drop ``"recall_fact"`` if you already inject
            memory via a hook.

    Returns:
        A list of :class:`Tool` objects ready to hand to
        ``LLMAgent(tools=...)`` or register manually.
    """
    available = {
        "remember_fact": _build_remember(store),
        "forget_fact": _build_forget(store),
        "recall_fact": _build_recall(store),
    }
    missing = set(include) - available.keys()
    if missing:
        raise ValueError(f"unknown memory tool(s): {sorted(missing)}")
    return [available[name] for name in include]


def _build_remember(store: MemoryStore) -> Tool:
    def remember_fact(key: str, value: str, scope: str = "global") -> str:
        """Persist a fact the user just told you so you can recall it later.

        Args:
            key: Short identifier, e.g. ``user.name`` or ``kitchen.fridge_temp``.
            value: The fact itself, as a string.
            scope: Logical scope — ``global``, ``user``, ``env:<name>``.
        """
        store.add_fact(key, value, scope=scope, source="llm")
        return f"remembered {key!r} = {value!r} (scope={scope})"

    return Tool(
        name="remember_fact",
        description=remember_fact.__doc__.strip().splitlines()[0],
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Short identifier for the fact."},
                "value": {"type": "string", "description": "The fact's value as a string."},
                "scope": {
                    "type": "string",
                    "description": "Logical scope: 'global', 'user', or 'env:<name>'.",
                    "default": "global",
                },
            },
            "required": ["key", "value"],
        },
        func=remember_fact,
    )


def _build_forget(store: MemoryStore) -> Tool:
    def forget_fact(key: str, scope: str = "global") -> str:
        """Revoke a previously-remembered fact — use when the user corrects you.

        Args:
            key: The fact key to forget.
            scope: Scope the fact lives in.
        """
        removed = store.forget_fact(key, scope=scope)
        if removed:
            return f"forgot {key!r} (scope={scope})"
        return f"no fact {key!r} in scope={scope} — nothing to forget"

    return Tool(
        name="forget_fact",
        description=forget_fact.__doc__.strip().splitlines()[0],
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The fact key to forget."},
                "scope": {
                    "type": "string",
                    "description": "Scope the fact lives in.",
                    "default": "global",
                },
            },
            "required": ["key"],
        },
        func=forget_fact,
    )


def _build_recall(store: MemoryStore) -> Tool:
    def recall_fact(key: str, scope: str = "global") -> str:
        """Look up a previously-remembered fact by key.

        Args:
            key: The fact key to look up.
            scope: Scope the fact lives in.
        """
        value = store.get_fact(key, scope=scope)
        if value is None:
            return f"no fact {key!r} remembered in scope={scope}"
        return value

    return Tool(
        name="recall_fact",
        description=recall_fact.__doc__.strip().splitlines()[0],
        parameters={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The fact key to look up."},
                "scope": {
                    "type": "string",
                    "description": "Scope the fact lives in.",
                    "default": "global",
                },
            },
            "required": ["key"],
        },
        func=recall_fact,
    )


__all__ = ["memory_tools"]
