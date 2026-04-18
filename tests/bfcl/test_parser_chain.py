"""BFCL v3-style regression: the parser chain must round-trip the
shipped fixtures into AST-equivalent tool calls.

Run only this slice for fast local iteration:

    uv run pytest tests/bfcl -q

Each fixture entry:

- ``id`` — stable identifier for diffing across runs.
- ``category`` — one of ``simple``, ``parallel``, ``multiple``,
  ``irrelevance``, ``relevance``. Drives whether
  :func:`calls_equivalent` runs in unordered or ordered mode.
- ``model_output`` — the raw assistant ``content`` string we feed
  through :func:`parse_tool_calls_from_content`.
- ``tools`` — OpenAI tool schema list (drives detector behaviour).
- ``gold`` — expected OpenAI-shaped tool_call dicts; ``[]`` for
  irrelevance/relevance turns where no call should fire.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgevox.llm.llamacpp import parse_tool_calls_from_content
from tests.bfcl.ast_eq import calls_equivalent

FIXTURES_PATH = Path(__file__).parent / "fixtures.json"
FIXTURES = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "fixture",
    FIXTURES,
    ids=[f["id"] for f in FIXTURES],
)
def test_parser_chain_matches_gold(fixture):
    detectors = ("qwen25", "hermes")  # default chain for chatml-style outputs
    calls, _cleaned, _fallback = parse_tool_calls_from_content(
        fixture["model_output"],
        preset_parsers=detectors,
        tool_schemas=fixture["tools"],
    )
    parallel = fixture.get("category") in ("parallel", "multiple")
    assert calls_equivalent(
        calls,
        fixture["gold"],
        parallel_unordered=parallel,
    ), f"AST mismatch for {fixture['id']}: predicted={calls!r} gold={fixture['gold']!r}"


def test_ast_equivalence_predicate_basic():
    """Sanity-check the predicate itself."""
    a = [{"function": {"name": "f", "arguments": '{"x": 1, "y": 2}'}}]
    b = [{"function": {"name": "f", "arguments": '{"y": 2, "x": 1}'}}]
    assert calls_equivalent(a, b)
    c = [{"function": {"name": "f", "arguments": '{"x": 1}'}}]
    assert not calls_equivalent(a, c)


def test_parallel_unordered_mode():
    """In ``parallel`` mode the call order doesn't matter."""
    a = [
        {"function": {"name": "f", "arguments": '{"x": 1}'}},
        {"function": {"name": "g", "arguments": '{"y": 2}'}},
    ]
    b = list(reversed(a))
    assert calls_equivalent(a, b, parallel_unordered=True)
    assert not calls_equivalent(a, b, parallel_unordered=False)


def test_irrelevance_returns_empty_calls():
    """Irrelevance fixtures must produce zero tool calls — the parser
    chain doesn't false-positive on plain prose."""
    fix = next(f for f in FIXTURES if f["id"] == "irrelevance-001")
    calls, _cleaned, _fallback = parse_tool_calls_from_content(
        fix["model_output"],
        preset_parsers=("qwen25", "hermes"),
        tool_schemas=fix["tools"],
    )
    assert calls == []
