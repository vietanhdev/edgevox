"""Regression: Qwen3-Instruct emits tool calls inside ``<think>`` blocks.

The parser chain used to run ``_strip_thinking`` before the SGLang +
chatml detectors, which silently dropped any tool call the model placed
inside its reasoning block — a confirmed bug in upstream llama.cpp
(https://github.com/ggml-org/llama.cpp/issues/20837).

These tests lock in the post-fix order: detectors run against the **raw**
content first, and only fall back to the stripped content when nothing
matched. The user-facing reply is still derived from the stripped text
so chain-of-thought never reaches TTS.
"""

from __future__ import annotations

import json

from edgevox.llm.llamacpp import parse_tool_calls_from_content


def test_chatml_tool_call_inside_think_block_recovered():
    raw = '<think>\nThe user wants the time.\n<tool_call>{"name": "get_time", "arguments": {}}</tool_call>\n</think>'
    calls, cleaned, fallback = parse_tool_calls_from_content(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_time"
    # The visible reply must not leak the <think> block.
    assert "<think>" not in cleaned
    assert fallback is True


def test_sglang_preset_detector_sees_raw_content():
    """SGLang detectors (hermes/qwen25) must also run on the raw
    content so a tool call embedded in a think block is recovered."""
    raw = '<think>reasoning...\n<tool_call>\n{"name": "set_temp", "arguments": {"c": 22}}\n</tool_call>\n</think>'
    calls, _cleaned, _fallback = parse_tool_calls_from_content(
        raw,
        preset_parsers=("qwen25", "hermes"),
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "set_temp",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {"c": {"type": "integer"}},
                        "required": ["c"],
                    },
                },
            }
        ],
    )
    assert len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"c": 22}


def test_no_think_block_still_parses():
    """When there's no think block at all, the raw and scrubbed passes
    are identical and parsing still works (no regression)."""
    raw = '<tool_call>{"name": "noop", "arguments": {}}</tool_call>'
    calls, _cleaned, fallback = parse_tool_calls_from_content(raw)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "noop"
    assert fallback is True


def test_reply_outside_think_kept_after_strip():
    """A tool call inside think + trailing user-facing reply: the reply
    survives, the <think> block is stripped from cleaned output."""
    raw = '<think>\nLet me check.\n<tool_call>{"name": "get_time", "arguments": {}}</tool_call>\n</think>\nOne moment.'
    calls, cleaned, _fallback = parse_tool_calls_from_content(raw)
    # Tool call recovered from inside the think block.
    assert calls and calls[0]["function"]["name"] == "get_time"
    # Chain-of-thought gone.
    assert "<think>" not in cleaned
    assert "Let me check" not in cleaned


def test_plain_prose_with_no_tool_call_returns_empty():
    """Parser chain must not false-positive on ordinary prose, with or
    without a think block."""
    raw = "<think>thinking</think>Hello there."
    calls, cleaned, fallback = parse_tool_calls_from_content(raw)
    assert calls == []
    assert cleaned == "Hello there."
    assert fallback is False


def test_plain_call_regex_skips_code_fences():
    """``grasp(x=1)`` inside a fenced code block must not be dispatched
    as a tool call — that's example code, not a call."""
    raw = "Here's an example:\n```python\ngrasp(x=1)\n```"
    calls, _cleaned, _fallback = parse_tool_calls_from_content(raw, known_tools={"grasp"})
    assert calls == []
