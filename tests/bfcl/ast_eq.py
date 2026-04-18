"""AST-style equivalence between tool-call lists.

BFCL v3 compares model output to a gold reference by parsing both into
an abstract syntactic representation: ``[(tool_name, sorted_kwargs),
…]``. Two predictions are equivalent iff:

1. Same set of tool names in the same order (parallel calls in the
   same hop are *order-independent*; passing them through ``frozenset``
   handles that).
2. For each call, the kwarg keys match exactly.
3. For each kwarg, the values match by Python equality after
   normalising obvious wire-format quirks (string-encoded JSON,
   trailing whitespace, etc.).

This module ships the predicate so unit tests can invoke it directly:

.. code-block:: python

    from tests.bfcl.ast_eq import calls_equivalent
    assert calls_equivalent(predicted_calls, gold_calls)
"""

from __future__ import annotations

import json
from typing import Any


def _coerce_args(raw: Any) -> dict:
    """Decode an OpenAI-style ``arguments`` field to a dict.

    OpenAI tool calls carry ``arguments`` as a JSON-encoded string;
    SGLang detectors may already pre-parse it. Either form is
    accepted.
    """
    if isinstance(raw, dict):
        return {str(k): v for k, v in raw.items()}
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {"__raw__": raw}
        return decoded if isinstance(decoded, dict) else {"__raw__": raw}
    return {}


def _normalise_call(call: dict) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """Turn an OpenAI-shaped tool_call dict into ``(name, sorted_kwargs)``.

    The kwargs are sorted by key so order-of-emission differences
    don't break equivalence.
    """
    fn = (call or {}).get("function") or {}
    name = fn.get("name") or ""
    args = _coerce_args(fn.get("arguments"))
    return (name, tuple(sorted(args.items())))


def _normalise_value(v: Any) -> Any:
    """Lossless normalisation for tolerable wire-format drift.

    - Strip whitespace from string values.
    - Coerce ``True``/``False`` token mismatches (no-op in Python).
    - Recurse through nested dicts / lists.
    """
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, list):
        return [_normalise_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _normalise_value(val) for k, val in v.items()}
    return v


def calls_equivalent(
    predicted: list[dict] | None,
    gold: list[dict] | None,
    *,
    parallel_unordered: bool = True,
) -> bool:
    """Return True iff ``predicted`` matches ``gold`` under BFCL v3 semantics.

    ``parallel_unordered=True`` (the default, matching BFCL's "parallel"
    category): treat the two call lists as multisets — order does not
    matter as long as each (name, kwargs) appears the same number of
    times. Set ``False`` for sequential / multi-turn comparisons where
    order is meaningful.
    """
    p = list(predicted or [])
    g = list(gold or [])
    if len(p) != len(g):
        return False
    p_norm = [
        (name, {k: _normalise_value(v) for k, v in dict(kw).items()}) for name, kw in (_normalise_call(c) for c in p)
    ]
    g_norm = [
        (name, {k: _normalise_value(v) for k, v in dict(kw).items()}) for name, kw in (_normalise_call(c) for c in g)
    ]
    if parallel_unordered:
        # Compare as a multiset — convert each call to a hashable
        # canonical form (JSON, sorted keys).
        canon = lambda items: sorted(  # noqa: E731 — local helper
            json.dumps((n, kw), sort_keys=True, default=str) for n, kw in items
        )
        return canon(p_norm) == canon(g_norm)
    return p_norm == g_norm
